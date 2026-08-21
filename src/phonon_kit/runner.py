from __future__ import annotations

import shutil
import traceback
from pathlib import Path
from typing import Any

from .analysis import analyze_method
from .compare import compare_methods
from .config import Config, DeepMDMethod, VaspMethod, load_config
from .errors import IncompleteResultsError, RunStateError
from .providers.deepmd import DeepMDProvider, relax_structure
from .providers.vasp import VaspProvider
from .state import RunPaths, StateStore, choose_run, matching_run
from .snapshot import SNAPSHOT_CONFIG, create_config_snapshot
from .structure import generate_displacements, normalize_structure
from .util import load_json, utc_now
from .validation import validate_runtime


def selected_method_names(config: Config, only: list[str] | None) -> list[str]:
    selected = list(config.enabled_methods) if not only else list(only)
    if len(selected) != len(set(selected)):
        raise ValueError("--only 中的方法名不能重复")
    invalid = [name for name in selected if name not in config.enabled_methods]
    if invalid:
        raise ValueError(f"方法未定义或未启用: {', '.join(invalid)}")
    return selected


def _record_error(store: StateStore, method: str | None, exc: Exception) -> None:
    state = store.load()
    state.setdefault("errors", []).append({
        "time": utc_now(),
        "method": method,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    })
    store.save(state)


def _canonical_structure(config: Config, paths: RunPaths, store: StateStore) -> Path:
    state = store.load()
    output = paths.canonical / "POSCAR"
    if state.get("stages", {}).get("structure_normalized") and output.is_file():
        return output
    metadata = normalize_structure(config.structure.file, output)
    from .util import atomic_write_json

    atomic_write_json(paths.canonical / "structure.json", metadata)
    store.update_stage("structure_normalized", metadata)
    return output


def _relax_if_needed(config: Config, paths: RunPaths, store: StateStore, canonical: Path) -> Path:
    if not config.relaxation.enabled:
        store.update_stage("relaxation", {"enabled": False, "status": "skipped"})
        return canonical
    relaxed = paths.canonical / "POSCAR-relaxed"
    state = store.load()
    stage = state.get("stages", {}).get("relaxation", {})
    if stage.get("status") == "completed" and relaxed.is_file():
        return relaxed
    method = config.methods[config.relaxation.method]
    assert isinstance(method, DeepMDMethod)
    workdir = paths.work / "relaxation"
    summary = relax_structure(
        method,
        canonical,
        workdir,
        paths.logs,
        variable_cell=config.relaxation.variable_cell,
        pressure_gpa=config.relaxation.pressure_gpa,
        fmax=config.relaxation.fmax_ev_angstrom,
        max_steps=config.relaxation.max_steps,
        trajectory_interval=config.relaxation.trajectory_interval,
    )
    shutil.copy2(workdir / "POSCAR-relaxed", relaxed)
    store.update_stage("relaxation", {"enabled": True, "status": "completed", **summary})
    return relaxed


def _displacements(config: Config, paths: RunPaths, store: StateStore, canonical: Path) -> Path:
    yaml_path = paths.displacements / "phonopy_disp.yaml"
    state = store.load()
    if state.get("stages", {}).get("displacements") and yaml_path.is_file():
        return yaml_path
    manifest = generate_displacements(config, canonical, paths.displacements)
    store.update_stage("displacements", manifest)
    return yaml_path


def _provider(config: Config, paths: RunPaths, name: str, phonopy_yaml: Path):
    method = config.methods[name]
    kwargs = {
        "method": method,
        "phonopy_yaml": phonopy_yaml,
        "workdir": paths.method_work(name),
        "resultdir": paths.method_results(name),
        "logdir": paths.logs,
    }
    if isinstance(method, DeepMDMethod):
        return DeepMDProvider(**kwargs)
    if isinstance(method, VaspMethod):
        return VaspProvider(**kwargs)
    raise TypeError(method)


def _finalize_status(store: StateStore, selected: list[str]) -> dict[str, Any]:
    state = store.load()
    statuses = [state.get("methods", {}).get(name, {}).get("status") for name in selected]
    if statuses and all(status == "completed" for status in statuses):
        state["status"] = "completed"
        state["completed_at"] = utc_now()
    elif any(status == "failed" for status in statuses):
        state["status"] = "failed"
    elif any(status in {"submitted", "waiting"} for status in statuses):
        state["status"] = "waiting_dft"
    else:
        state["status"] = "running"
    store.save(state)
    return state


def execute_run(config: Config, paths: RunPaths, *, wait: bool = False, validate: bool = False) -> dict[str, Any]:
    store = StateStore(paths)
    state = store.load()
    selected = list(state["selected_methods"])
    if state.get("status") == "completed":
        return state
    store.update(status="running")
    try:
        if validate:
            report = validate_runtime(config, selected, real_inference=True, logdir=paths.logs)
            from .util import atomic_write_json

            atomic_write_json(paths.root / "validation.json", report)
        canonical = _canonical_structure(config, paths, store)
        canonical = _relax_if_needed(config, paths, store, canonical)
        phonopy_yaml = _displacements(config, paths, store, canonical)
    except Exception as exc:
        _record_error(store, None, exc)
        store.update(status="failed")
        raise

    for name in selected:
        method_state = store.load().get("methods", {}).get(name, {})
        if method_state.get("status") == "completed":
            continue
        provider = _provider(config, paths, name, phonopy_yaml)
        store.update_method(name, status="running", started_at=method_state.get("started_at", utc_now()))
        try:
            status = provider.run_or_submit(wait=wait)
            if status != "ready":
                ready = total = None
                if isinstance(provider, VaspProvider):
                    ready, total = provider.readiness()
                store.update_method(name, status="submitted", ready=ready, total=total)
                continue
            forces, energies = provider.collect()
            store.update_method(name, status="analyzing", n_displacements=len(forces), energies_available=energies is not None)
            summary = analyze_method(config, name, phonopy_yaml, paths.method_results(name))
            store.update_method(name, status="completed", completed_at=utc_now(), summary=summary)
        except IncompleteResultsError as exc:
            if isinstance(config.methods[name], VaspMethod):
                ready, total = provider.readiness()
                store.update_method(name, status="waiting", ready=ready, total=total, message=str(exc))
            else:
                _record_error(store, name, exc)
                store.update_method(name, status="failed", message=str(exc))
        except Exception as exc:
            _record_error(store, name, exc)
            store.update_method(name, status="failed", message=str(exc))

    try:
        comparison = compare_methods(config, selected, paths.results)
        if comparison is not None:
            store.update_stage("comparison", {"status": "completed", "result": comparison})
    except Exception as exc:
        _record_error(store, "comparison", exc)
        store.update_stage("comparison", {"status": "failed", "message": str(exc)})
    state = _finalize_status(store, selected)
    if state["status"] == "failed":
        failed = [name for name in selected if state["methods"][name]["status"] == "failed"]
        raise RuntimeError(f"运行中有方法失败: {', '.join(failed)}；请查看 {paths.logs} 和 state.json")
    return state


def run_config(config: Config, *, only: list[str] | None = None, force_new: bool = False, wait: bool = False) -> tuple[RunPaths, dict[str, Any], bool]:
    selected = selected_method_names(config, only)
    paths, state, created = choose_run(config, selected, force_new=force_new)
    snapshot_path = paths.root / SNAPSHOT_CONFIG
    run_config = (
        create_config_snapshot(config, paths.root)
        if created or not snapshot_path.is_file()
        else load_config(snapshot_path)
    )
    if state.get("status") == "completed" and not force_new:
        return paths, state, False
    final = execute_run(run_config, paths, wait=wait, validate=created)
    return paths, final, created


def resume_config(config: Config, *, wait: bool = False, paths: RunPaths | None = None) -> tuple[RunPaths, dict[str, Any]]:
    if paths is None:
        paths, state = matching_run(config)
    else:
        state = StateStore(paths).load()
    run_config = load_config(paths.root / SNAPSHOT_CONFIG)
    if state.get("status") == "completed":
        return paths, state
    return paths, execute_run(run_config, paths, wait=wait, validate=False)


def collect_config(config: Config, *, paths: RunPaths | None = None) -> tuple[RunPaths, dict[str, Any]]:
    if paths is None:
        paths, state = matching_run(config)
    else:
        state = StateStore(paths).load()
    config = load_config(paths.root / SNAPSHOT_CONFIG)
    store = StateStore(paths)
    selected = list(state["selected_methods"])
    yaml_path = paths.displacements / "phonopy_disp.yaml"
    if not yaml_path.is_file():
        raise RunStateError("尚未生成 phonopy_disp.yaml，不能收集 VASP 结果")
    found = False
    for name in selected:
        if not isinstance(config.methods[name], VaspMethod):
            continue
        found = True
        provider = _provider(config, paths, name, yaml_path)
        try:
            forces, energies = provider.collect()
            store.update_method(name, status="analyzing", n_displacements=len(forces), energies_available=energies is not None)
            summary = analyze_method(config, name, yaml_path, paths.method_results(name))
            store.update_method(name, status="completed", completed_at=utc_now(), summary=summary)
        except IncompleteResultsError as exc:
            ready, total = provider.readiness()
            store.update_method(name, status="waiting", ready=ready, total=total, message=str(exc))
            raise
    if not found:
        raise RunStateError("当前运行未选择 VASP 方法")
    comparison = compare_methods(config, selected, paths.results)
    if comparison:
        store.update_stage("comparison", {"status": "completed", "result": comparison})
    return paths, _finalize_status(store, selected)


def replot_config(config: Config, *, paths: RunPaths | None = None) -> tuple[RunPaths, dict[str, Any]]:
    if paths is None:
        paths, state = matching_run(config)
    else:
        state = StateStore(paths).load()
    config = load_config(paths.root / SNAPSHOT_CONFIG)
    selected = list(state["selected_methods"])
    yaml_path = paths.displacements / "phonopy_disp.yaml"
    for name in selected:
        resultdir = paths.method_results(name)
        if (resultdir / "forces.npy").is_file():
            analyze_method(config, name, yaml_path, resultdir)
    compare_methods(config, selected, paths.results)
    return paths, StateStore(paths).load()


def status_text(paths: RunPaths, state: dict[str, Any]) -> str:
    lines = [
        f"run: {paths.root}",
        f"status: {state.get('status', 'unknown')}",
        f"updated: {state.get('updated_at', '-')}",
    ]
    displacement = state.get("stages", {}).get("displacements")
    if isinstance(displacement, dict):
        lines.append(f"displacements: {displacement.get('n_displacements', '?')}")
    for name in state.get("selected_methods", []):
        item = state.get("methods", {}).get(name, {})
        detail = item.get("status", "unknown")
        if item.get("total") is not None:
            detail += f" ({item.get('ready', 0)}/{item['total']} VASP results)"
        lines.append(f"method {name}: {detail}")
    comparison = state.get("stages", {}).get("comparison")
    if comparison:
        lines.append(f"comparison: {comparison.get('status', 'unknown')}")
    if state.get("errors"):
        latest = state["errors"][-1]
        lines.append(f"latest error: {latest.get('method') or 'workflow'}: {latest.get('message')}")
    return "\n".join(lines)

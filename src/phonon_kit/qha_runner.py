from __future__ import annotations

import shutil
import traceback
from pathlib import Path
from typing import Any

import yaml

from .config import DeepMDMethod
from .errors import IncompleteResultsError, RetryableExternalError, RunStateError
from .providers.deepmd import DeepMDProvider
from .qha_analysis import analyze_phase_diagram, analyze_phase_qha, analyze_qha_volume, compare_qha_methods
from .qha_config import QHAConfig, QHAVaspMethod
from .qha_deepmd import evaluate_qha_static, relax_qha_volume
from .qha_state import (
    QHARunPaths,
    QHAStateStore,
    choose_qha_run,
    matching_qha_run,
    volume_id,
)
from .qha_structure import generate_qha_displacements, scale_structure
from .qha_validation import validate_qha_runtime
from .qha_vasp import (
    collect_phonon_tasks,
    collect_relaxation_tasks,
    collect_static_tasks,
    prepare_phonon_tasks,
    prepare_relaxation_tasks,
    prepare_static_tasks,
    submit_stage,
)
from .util import atomic_write_json, atomic_write_text, load_json, utc_now


def selected_qha_methods(config: QHAConfig, only: list[str] | None) -> list[str]:
    selected = list(config.enabled_methods) if not only else list(only)
    if len(selected) != len(set(selected)):
        raise ValueError("--only 中的方法名不能重复")
    invalid = [name for name in selected if name not in config.enabled_methods]
    if invalid:
        raise ValueError(f"方法未定义或未启用: {', '.join(invalid)}")
    return selected


def _snapshot(config: QHAConfig, paths: QHARunPaths) -> None:
    atomic_write_text(
        paths.root / "config.resolved.yaml",
        yaml.safe_dump(config.resolved_dict(), sort_keys=False, allow_unicode=True),
    )


def _record_error(store: QHAStateStore, method: str | None, exc: Exception) -> None:
    state = store.load()
    state.setdefault("errors", []).append({
        "time": utc_now(),
        "method": method,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    })
    store.save(state)


def _deepmd_volume(
    config: QHAConfig,
    method: DeepMDMethod,
    phase_name: str,
    index: int,
    ratio: float,
    paths: QHARunPaths,
    store: QHAStateStore,
) -> dict[str, Any]:
    phase = config.phases[phase_name]
    vid = volume_id(index, ratio)
    volume_root = paths.volume_work(method.name, phase_name, vid)
    resultdir = paths.phase_results(method.name, phase_name) / "volumes" / vid
    summary_path = resultdir / "volume_summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        store.update_volume(method.name, phase_name, vid, status="completed", summary=summary)
        return summary

    scaled = volume_root / "POSCAR-scaled"
    if not scaled.is_file():
        scaling = scale_structure(phase.structure, scaled, ratio)
    else:
        scaling = load_json(scaled.with_suffix(scaled.suffix + ".json"))
    store.update_volume(method.name, phase_name, vid, status="relaxing", target_volume_angstrom3=scaling["target_volume_angstrom3"])
    relax_dir = volume_root / "relaxation"
    relax_summary_path = relax_dir / "summary.json"
    if relax_summary_path.is_file() and (relax_dir / "POSCAR-relaxed").is_file():
        relaxation = load_json(relax_summary_path)
        if not relaxation.get("converged"):
            relaxation = relax_qha_volume(
                method,
                scaled,
                relax_dir,
                paths.logs / method.name / phase_name / vid,
                config.volume_relaxation,
                float(scaling["target_volume_angstrom3"]),
            )
    else:
        relaxation = relax_qha_volume(
            method,
            scaled,
            relax_dir,
            paths.logs / method.name / phase_name / vid,
            config.volume_relaxation,
            float(scaling["target_volume_angstrom3"]),
        )
    relaxed = relax_dir / "POSCAR-relaxed"

    store.update_volume(method.name, phase_name, vid, status="static")
    static_dir = volume_root / "static"
    static_summary_path = static_dir / "static-summary.json"
    if static_summary_path.is_file():
        static = load_json(static_summary_path)
    else:
        static = evaluate_qha_static(
            method,
            relaxed,
            static_dir,
            paths.logs / method.name / phase_name / vid,
        )
        atomic_write_json(static_summary_path, static)

    store.update_volume(method.name, phase_name, vid, status="phonon")
    phonon_dir = volume_root / "phonon"
    phonopy_yaml = phonon_dir / "phonopy_disp.yaml"
    if not phonopy_yaml.is_file():
        generate_qha_displacements(relaxed, phase.phonon, phonon_dir)
    provider = DeepMDProvider(
        method=method,
        phonopy_yaml=phonopy_yaml,
        workdir=phonon_dir / "deepmd",
        resultdir=resultdir,
        logdir=paths.logs / method.name / phase_name / vid,
    )
    provider.run_or_submit(wait=True)
    provider.collect()
    summary = analyze_qha_volume(
        config,
        method.name,
        phase_name,
        ratio,
        phonopy_yaml,
        float(static["energy_ev"]),
        relaxation,
        resultdir,
    )
    store.update_volume(method.name, phase_name, vid, status="completed", summary=summary)
    return summary


def _run_deepmd_method(
    config: QHAConfig,
    method: DeepMDMethod,
    paths: QHARunPaths,
    store: QHAStateStore,
) -> None:
    store.update_method(method.name, status="running", stage="volume_phonons")
    for phase_name, phase in config.phases.items():
        store.update_phase(method.name, phase_name, status="running")
        for index, ratio in enumerate(phase.volume_ratios):
            _deepmd_volume(config, method, phase_name, index, ratio, paths, store)
        store.update_phase(method.name, phase_name, status="analyzing_qha")
        summary = analyze_phase_qha(config, method.name, phase_name, paths)
        store.update_phase(method.name, phase_name, status="completed", summary=summary)
    store.update_method(method.name, status="analyzing_phase_diagram", stage="phase_diagram")
    diagram = analyze_phase_diagram(config, method.name, paths)
    store.update_method(method.name, status="completed", stage="completed", completed_at=utc_now(), summary=diagram)


def _mark_vasp_relaxations(
    config: QHAConfig,
    method: QHAVaspMethod,
    summaries: dict[tuple[str, str], dict[str, Any]],
    store: QHAStateStore,
) -> None:
    for phase_name, phase in config.phases.items():
        for index, ratio in enumerate(phase.volume_ratios):
            vid = volume_id(index, ratio)
            store.update_volume(method.name, phase_name, vid, status="relaxed", relaxation=summaries[(phase_name, vid)])


def _analyze_vasp_method(
    config: QHAConfig,
    method: QHAVaspMethod,
    paths: QHARunPaths,
    store: QHAStateStore,
    relaxations: dict[tuple[str, str], dict[str, Any]],
    energies: dict[tuple[str, str], float],
) -> None:
    for phase_name, phase in config.phases.items():
        store.update_phase(method.name, phase_name, status="analyzing_volumes")
        for index, ratio in enumerate(phase.volume_ratios):
            vid = volume_id(index, ratio)
            volume_root = paths.volume_work(method.name, phase_name, vid)
            resultdir = paths.phase_results(method.name, phase_name) / "volumes" / vid
            summary_path = resultdir / "volume_summary.json"
            if summary_path.is_file():
                summary = load_json(summary_path)
            else:
                summary = analyze_qha_volume(
                    config,
                    method.name,
                    phase_name,
                    ratio,
                    volume_root / "phonon" / "phonopy_disp.yaml",
                    energies[(phase_name, vid)],
                    relaxations[(phase_name, vid)],
                    resultdir,
                )
            store.update_volume(method.name, phase_name, vid, status="completed", summary=summary)
        store.update_phase(method.name, phase_name, status="analyzing_qha")
        phase_summary = analyze_phase_qha(config, method.name, phase_name, paths)
        store.update_phase(method.name, phase_name, status="completed", summary=phase_summary)
    diagram = analyze_phase_diagram(config, method.name, paths)
    store.update_method(method.name, status="completed", stage="completed", completed_at=utc_now(), summary=diagram)


def _run_vasp_method(
    config: QHAConfig,
    method: QHAVaspMethod,
    paths: QHARunPaths,
    store: QHAStateStore,
    *,
    wait: bool,
    allow_submit: bool,
) -> None:
    store.update_method(method.name, status="running", stage="volume_relax")
    relax_tasks = prepare_relaxation_tasks(config, method, paths)
    try:
        relaxations = collect_relaxation_tasks(config, method, paths)
    except IncompleteResultsError as exc:
        if not allow_submit:
            store.update_method(method.name, status="waiting", stage="volume_relax", message=str(exc))
            return
        submission = submit_stage(method, paths, "volume_relax", relax_tasks, wait=wait)
        if submission["status"] != "ready":
            ready = sum((task / "vasprun.xml").is_file() for task in relax_tasks)
            store.update_method(method.name, status="submitted", stage="volume_relax", ready=ready, total=len(relax_tasks), submission=submission)
            return
        relaxations = collect_relaxation_tasks(config, method, paths)
    _mark_vasp_relaxations(config, method, relaxations, store)

    static_tasks = prepare_static_tasks(config, method, paths)
    phonon_tasks = prepare_phonon_tasks(config, method, paths)
    static_ready = phonon_ready = False
    try:
        energies = collect_static_tasks(config, method, paths)
        static_ready = True
    except IncompleteResultsError as static_error:
        energies = {}
        if not allow_submit:
            store.update_method(method.name, status="waiting", stage="static", message=str(static_error))
        else:
            submission = submit_stage(method, paths, "static", static_tasks, wait=wait)
            static_ready = submission["status"] == "ready"
            if static_ready:
                energies = collect_static_tasks(config, method, paths)
    try:
        collect_phonon_tasks(config, method, paths)
        phonon_ready = True
    except IncompleteResultsError as phonon_error:
        if not allow_submit:
            if static_ready:
                store.update_method(method.name, status="waiting", stage="phonon", message=str(phonon_error))
        else:
            submission = submit_stage(method, paths, "phonon", phonon_tasks, wait=wait)
            phonon_ready = submission["status"] == "ready"
            if phonon_ready:
                collect_phonon_tasks(config, method, paths)
    if not static_ready or not phonon_ready:
        ready_static = sum((task / "vasprun.xml").is_file() for task in static_tasks)
        ready_phonon = sum((task / "vasprun.xml").is_file() for task in phonon_tasks)
        store.update_method(
            method.name,
            status="submitted" if allow_submit else "waiting",
            stage="static_and_phonon",
            ready={"static": ready_static, "phonon": ready_phonon},
            total={"static": len(static_tasks), "phonon": len(phonon_tasks)},
        )
        return
    _analyze_vasp_method(config, method, paths, store, relaxations, energies)


def _finalize(config: QHAConfig, paths: QHARunPaths, store: QHAStateStore) -> dict[str, Any]:
    state = store.load()
    selected = list(state["selected_methods"])
    statuses = [state["methods"][name].get("status") for name in selected]
    completed = [name for name, status in zip(selected, statuses) if status == "completed"]
    try:
        comparison = compare_qha_methods(config, completed, paths)
        if comparison is not None:
            state["comparison"] = {"status": "completed", "summary": comparison}
    except Exception as exc:
        _record_error(store, "comparison", exc)
        state = store.load()
        state["comparison"] = {"status": "failed", "message": str(exc)}
    if statuses and all(status == "completed" for status in statuses):
        state["status"] = "completed"
        state["completed_at"] = utc_now()
    elif any(status == "failed" for status in statuses):
        state["status"] = "failed"
    else:
        state["status"] = "waiting"
    store.save(state)
    return state


def execute_qha(
    config: QHAConfig,
    paths: QHARunPaths,
    *,
    wait: bool = False,
    validate: bool = False,
    collect_only: bool = False,
) -> dict[str, Any]:
    store = QHAStateStore(paths)
    state = store.load()
    if state.get("status") == "completed":
        return state
    store.update(status="running")
    _snapshot(config, paths)
    selected = list(state["selected_methods"])
    validation_path = paths.root / "validation.json"
    if not collect_only and (validate or not validation_path.is_file()):
        try:
            report = validate_qha_runtime(config, selected, real_inference=True)
            atomic_write_json(validation_path, report)
        except Exception as exc:
            _record_error(store, None, exc)
            store.update(status="failed")
            raise
    for name in selected:
        method = config.methods[name]
        if store.load()["methods"][name].get("status") == "completed":
            continue
        try:
            if isinstance(method, DeepMDMethod):
                if not collect_only:
                    _run_deepmd_method(config, method, paths, store)
            else:
                _run_vasp_method(config, method, paths, store, wait=wait, allow_submit=not collect_only)
        except RetryableExternalError as exc:
            _record_error(store, name, exc)
            store.update_method(name, status="retryable", message=str(exc))
        except Exception as exc:
            _record_error(store, name, exc)
            store.update_method(name, status="failed", message=str(exc))
    final = _finalize(config, paths, store)
    if final["status"] == "failed":
        failed = [name for name in selected if final["methods"][name].get("status") == "failed"]
        raise RuntimeError(f"QHA 运行中有方法失败: {', '.join(failed)}；请查看 {paths.logs} 和 state.json")
    return final


def run_qha_config(
    config: QHAConfig,
    *,
    only: list[str] | None = None,
    force_new: bool = False,
    wait: bool = False,
) -> tuple[QHARunPaths, dict[str, Any], bool]:
    selected = selected_qha_methods(config, only)
    paths, state, created = choose_qha_run(config, selected, force_new=force_new)
    if state.get("status") == "completed" and not force_new:
        return paths, state, False
    return paths, execute_qha(config, paths, wait=wait, validate=created), created


def resume_qha_config(config: QHAConfig, *, wait: bool = False) -> tuple[QHARunPaths, dict[str, Any]]:
    paths, state = matching_qha_run(config)
    if state.get("status") == "completed":
        return paths, state
    return paths, execute_qha(config, paths, wait=wait, validate=False)


def collect_qha_config(config: QHAConfig) -> tuple[QHARunPaths, dict[str, Any]]:
    paths, state = matching_qha_run(config)
    if not any(isinstance(config.methods[name], QHAVaspMethod) for name in state["selected_methods"]):
        raise RunStateError("当前 QHA 运行未选择 VASP 方法")
    return paths, execute_qha(config, paths, validate=False, collect_only=True)


def replot_qha_config(config: QHAConfig) -> tuple[QHARunPaths, dict[str, Any]]:
    paths, state = matching_qha_run(config)
    completed: list[str] = []
    for name in state["selected_methods"]:
        if all((paths.phase_results(name, phase) / "qha_grid.npz").is_file() for phase in config.phases):
            analyze_phase_diagram(config, name, paths)
            completed.append(name)
    compare_qha_methods(config, completed, paths)
    return paths, QHAStateStore(paths).load()


def qha_status_text(paths: QHARunPaths, state: dict[str, Any]) -> str:
    lines = [
        f"qha run: {paths.root}",
        f"status: {state.get('status', 'unknown')}",
        f"updated: {state.get('updated_at', '-')}",
    ]
    for name in state.get("selected_methods", []):
        method = state.get("methods", {}).get(name, {})
        detail = f"method {name}: {method.get('status', 'unknown')}"
        if method.get("stage"):
            detail += f"; stage={method['stage']}"
        if method.get("ready") is not None:
            detail += f"; ready={method['ready']}/{method.get('total')}"
        lines.append(detail)
        for phase_name, phase in method.get("phases", {}).items():
            volumes = phase.get("volumes", {})
            complete = sum(item.get("status") == "completed" for item in volumes.values())
            lines.append(f"  phase {phase_name}: {phase.get('status', 'pending')} ({complete}/{len(volumes)} volumes)")
    if state.get("comparison", {}).get("status") == "completed":
        lines.append("method comparison: completed")
    if state.get("errors"):
        latest = state["errors"][-1]
        lines.append(f"latest error: {latest.get('method') or 'workflow'}: {latest.get('message')}")
    return "\n".join(lines)

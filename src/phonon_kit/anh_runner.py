from __future__ import annotations

import re
import traceback
from pathlib import Path
from typing import Any

from .anh_analysis import (
    analyze_anh_method,
    force_constants_are_valid,
    plot_method,
    produce_force_constants,
    result_is_complete,
)
from .anh_compare import compare_anh_methods
from .anh_config import AnhConfig, load_anh_config
from .anh_deepmd import run_anh_forces
from .anh_snapshot import SNAPSHOT_CONFIG, create_anh_snapshot
from .anh_state import AnhRunPaths, AnhStateStore, choose_anh_run, latest_anh_run
from .anh_structure import generate_anh_displacements
from .errors import RunStateError
from .providers.deepmd import validate_method
from .structure import normalize_structure
from .util import atomic_write_json, load_json, utc_now


def _record_error(store: AnhStateStore, method: str | None, exc: Exception) -> None:
    state = store.load()
    state.setdefault("errors", []).append({
        "time": utc_now(), "method": method, "type": type(exc).__name__,
        "message": str(exc), "traceback": traceback.format_exc(),
    })
    state["status"] = "failed"
    store.save(state)


def _canonical(config: AnhConfig, paths: AnhRunPaths, store: AnhStateStore) -> Path:
    output = paths.canonical / "POSCAR"
    if output.is_file() and store.load().get("stages", {}).get("structure_normalized"):
        return output
    metadata = normalize_structure(config.structure.file, output)
    atomic_write_json(paths.canonical / "structure.json", metadata)
    store.update_stage("structure_normalized", metadata)
    return output


def _displacements(config: AnhConfig, paths: AnhRunPaths, store: AnhStateStore, canonical: Path) -> Path:
    yaml_path = paths.displacements / "phono3py_disp.yaml"
    if yaml_path.is_file() and store.load().get("stages", {}).get("displacements"):
        return yaml_path
    manifest = generate_anh_displacements(config, canonical, paths.displacements)
    store.update_stage("displacements", manifest)
    return yaml_path


def run_anh_config(
    config: AnhConfig,
    *,
    force_new: bool = False,
    paths: AnhRunPaths | None = None,
) -> tuple[AnhRunPaths, dict[str, Any], bool]:
    if paths is not None:
        if force_new:
            raise RunStateError("指定运行目录时不能同时使用 --new")
        store = AnhStateStore(paths)
        store.load()
        config = load_anh_config(paths.root / SNAPSHOT_CONFIG)
        created = False
    else:
        paths, _, created = choose_anh_run(config, force_new=force_new)
        store = AnhStateStore(paths)
        if created:
            config = create_anh_snapshot(config, paths.root)
        else:
            config = load_anh_config(paths.root / SNAPSHOT_CONFIG)
    state = store.load()
    if state.get("status") == "completed":
        return paths, state, created
    state["status"] = "running"; store.save(state)
    current_method: str | None = None
    try:
        canonical = _canonical(config, paths, store)
        yaml_path = _displacements(config, paths, store, canonical)
        for name, method in config.enabled_methods.items():
            current_method = name
            resultdir = paths.method_results(name)
            resultdir.mkdir(parents=True, exist_ok=True)
            if result_is_complete(resultdir, config.anharmonic.lifetime_temperature_k):
                store.update_method(name, status="completed")
                continue
            method_state = store.load().get("methods", {}).get(name, {})
            if not method_state.get("validated"):
                report = validate_method(method, canonical, paths.method_work(name), paths.logs)
                store.update_method(name, status="validated", validated=True, validation=report)
            workdir = paths.method_work(name)
            checkpoint_dir = workdir / "checkpoints"
            force_summary = run_anh_forces(
                method, yaml_path, checkpoint_dir, paths.logs / f"anh-{name}.log",
                subtract_residual_forces=config.anharmonic.subtract_residual_forces,
            )
            store.update_method(name, status="forces_completed", force_progress=force_summary)
            if not force_constants_are_valid(resultdir):
                fc_summary = produce_force_constants(config, yaml_path, workdir, resultdir)
                store.update_method(name, status="force_constants_completed", force_constants=fc_summary)
            store.update_method(name, status="scattering")
            summary = analyze_anh_method(config, name, yaml_path, workdir, resultdir)
            store.update_method(name, status="completed", summary=summary)
        comparison = compare_anh_methods(
            list(config.enabled_methods), paths.results, config.anharmonic.lifetime_temperature_k
        )
        store.update_stage("comparison", {"status": "completed" if comparison else "skipped", "summary": comparison})
        final = store.load(); final["status"] = "completed"; store.save(final)
        return paths, store.load(), created
    except Exception as exc:
        _record_error(store, current_method, exc)
        raise


def replot_anh(config: AnhConfig, *, paths: AnhRunPaths | None = None) -> tuple[AnhRunPaths, dict[str, Any]]:
    if paths is None:
        latest = latest_anh_run(config)
        if latest is None: raise RunStateError("尚无三阶运行")
        paths, state = latest
    else:
        state = AnhStateStore(paths).load()
    snapshot = load_anh_config(paths.root / SNAPSHOT_CONFIG)
    for name in state.get("selected_methods", []):
        resultdir = paths.method_results(name)
        if (resultdir / "summary.json").is_file(): plot_method(resultdir)
    compare_anh_methods(state.get("selected_methods", []), paths.results, snapshot.anharmonic.lifetime_temperature_k)
    return paths, AnhStateStore(paths).load()


def anh_status_text(paths: AnhRunPaths, state: dict[str, Any]) -> str:
    lines = [f"run: {paths.root}", f"status: {state.get('status', 'unknown')}", f"updated: {state.get('updated_at', '-')}"]
    displacement = state.get("stages", {}).get("displacements", {})
    if displacement:
        fc3_detail = f"fc3 displacements: {displacement.get('fc3_displacements', '?')}"
        if "fc3_displacement_angstrom" in displacement:
            fc3_detail += f" at {displacement['fc3_displacement_angstrom']} Å"
        lines.append(fc3_detail)
        if displacement.get("fc2_uses_separate_displacements"):
            fc2_detail = f"fc2 displacements: {displacement.get('fc2_displacements', '?')} (separate)"
            if "fc2_displacement_angstrom" in displacement:
                fc2_detail += f" at {displacement['fc2_displacement_angstrom']} Å"
            lines.append(fc2_detail)
        else:
            lines.append("fc2 displacements: reused from fc3 dataset")
    for name in state.get("selected_methods", []):
        method = state.get("methods", {}).get(name, {})
        work = paths.method_work(name)
        fc3_done = len(list((work / "checkpoints" / "fc3").glob("disp-*.npz")))
        fc2_done = len(list((work / "checkpoints" / "fc2").glob("disp-*.npz")))
        gamma_done = len([p for p in (work / "scattering").glob("kappa-m*-g*.hdf5") if re.search(r"-g\d+", p.stem)])
        detail = f"method {name}: {method.get('status', 'pending')}"
        if displacement: detail += f"; forces fc3 {fc3_done}/{displacement.get('fc3_displacements', '?')}"
        if displacement.get("fc2_displacements", 0): detail += f", fc2 {fc2_done}/{displacement['fc2_displacements']}"
        manifest = work / "scattering" / "manifest.json"
        if manifest.is_file():
            total_gp = len(load_json(manifest).get("grid_points", [])); detail += f"; q points {gamma_done}/{total_gp}"
        lines.append(detail)
    if state.get("errors"):
        latest = state["errors"][-1]; lines.append(f"latest error: {latest.get('method') or 'workflow'}: {latest.get('message')}")
    return "\n".join(lines)

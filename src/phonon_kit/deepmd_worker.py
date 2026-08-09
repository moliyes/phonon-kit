from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .structure import phonopy_to_ase, read_structure, write_vasp_grouped
from .util import atomic_write_json, ensure_finite


def make_calculator(model: str, head: str | None):
    from deepmd.calculator import DP

    kwargs: dict[str, Any] = {"model": model}
    if head:
        kwargs["head"] = head
    return DP(**kwargs)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        with open(tmp_name, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    atoms = read_structure(Path(payload["structure"]))
    calc = make_calculator(payload["model"], payload.get("head"))
    type_map = list(calc.dp.get_type_map())
    missing = sorted(set(atoms.get_chemical_symbols()) - set(type_map))
    if missing:
        raise RuntimeError(f"模型 type map 缺少元素: {', '.join(missing)}")
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    stress = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    ensure_finite([energy], "energy")
    ensure_finite(forces, "forces")
    ensure_finite(stress, "stress")
    return {
        "model": payload["model"],
        "head": payload.get("head"),
        "type_map": type_map,
        "n_atoms": len(atoms),
        "energy_ev": energy,
        "force_max_ev_angstrom": float(np.linalg.norm(forces, axis=1).max()),
        "stress_ev_angstrom3": stress.tolist(),
    }


def calculate_forces(payload: dict[str, Any]) -> dict[str, Any]:
    from phonopy import load

    phonon = load(payload["phonopy_yaml"], produce_fc=False)
    cells = phonon.supercells_with_displacements
    checkpoint_dir = Path(payload["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    calc = make_calculator(payload["model"], payload.get("head"))
    completed = 0
    for index, cell in enumerate(cells, start=1):
        checkpoint = checkpoint_dir / f"disp-{index:04d}.npz"
        if checkpoint.is_file():
            try:
                with np.load(checkpoint) as data:
                    if data["forces"].shape == (len(cell), 3) and np.all(np.isfinite(data["forces"])):
                        completed += 1
                        continue
            except Exception:
                pass
        atoms = phonopy_to_ase(cell)
        atoms.calc = calc
        forces = np.asarray(atoms.get_forces(), dtype=float)
        energy = float(atoms.get_potential_energy())
        ensure_finite(forces, f"disp-{index:04d} forces")
        ensure_finite([energy], f"disp-{index:04d} energy")
        _atomic_npz(checkpoint, forces=forces, energy=np.asarray(energy))
        completed += 1
        print(f"forces {completed}/{len(cells)}", flush=True)
    result = {"n_displacements": len(cells), "completed": completed}
    atomic_write_json(checkpoint_dir / "worker-summary.json", result)
    return result


def relax(payload: dict[str, Any]) -> dict[str, Any]:
    from ase.filters import FrechetCellFilter
    from ase.io import Trajectory
    from ase.optimize import LBFGS

    workdir = Path(payload["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    checkpoint = workdir / "checkpoint.vasp"
    source = checkpoint if checkpoint.is_file() else Path(payload["structure"])
    atoms = read_structure(source)
    atoms.calc = make_calculator(payload["model"], payload.get("head"))
    trajectory_path = workdir / "relax.traj"
    trajectory = Trajectory(str(trajectory_path), mode="a", atoms=atoms)
    if len(trajectory) == 0:
        trajectory.write(atoms)

    variable_cell = bool(payload["variable_cell"])
    pressure_ev_angstrom3 = float(payload["pressure_gpa"]) / 160.21766208
    target = FrechetCellFilter(atoms, scalar_pressure=pressure_ev_angstrom3) if variable_cell else atoms
    optimizer = LBFGS(
        target,
        restart=str(workdir / "lbfgs.restart"),
        logfile=str(workdir / "relax.log"),
    )

    def save_progress() -> None:
        trajectory.write(atoms)
        write_vasp_grouped(atoms, checkpoint)

    optimizer.attach(save_progress, interval=int(payload["trajectory_interval"]))
    converged = bool(optimizer.run(fmax=float(payload["fmax"]), steps=int(payload["max_steps"])))
    trajectory.write(atoms)
    trajectory.close()
    final_path = workdir / "POSCAR-relaxed"
    write_vasp_grouped(atoms, final_path)
    forces = np.asarray(atoms.get_forces(), dtype=float)
    stress = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    result = {
        "converged": converged,
        "steps": int(optimizer.get_number_of_steps()),
        "energy_ev": float(atoms.get_potential_energy()),
        "fmax_ev_angstrom": float(np.linalg.norm(forces, axis=1).max()),
        "stress_ev_angstrom3": stress.tolist(),
        "volume_angstrom3": float(atoms.get_volume()),
        "output": str(final_path),
    }
    atomic_write_json(workdir / "summary.json", result)
    if not converged:
        raise RuntimeError(
            f"DPA 弛豫在 {payload['max_steps']} 步内未收敛；末态已保存在 {final_path}"
        )
    return result


def qha_relax(payload: dict[str, Any]) -> dict[str, Any]:
    """Relax ions and cell shape while projecting out volume changes."""
    from ase.filters import FrechetCellFilter
    from ase.io import Trajectory
    from ase.optimize import LBFGS

    workdir = Path(payload["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    checkpoint = workdir / "checkpoint.vasp"
    source = checkpoint if checkpoint.is_file() else Path(payload["structure"])
    atoms = read_structure(source)
    target_volume = float(payload["target_volume_angstrom3"])
    tolerance = float(payload["volume_tolerance_relative"])
    if abs(atoms.get_volume() - target_volume) / target_volume > tolerance:
        raise RuntimeError(
            f"QHA 固定体积起点不匹配: {atoms.get_volume():.10f} vs {target_volume:.10f} A^3"
        )
    atoms.calc = make_calculator(payload["model"], payload.get("head"))
    trajectory_path = workdir / "relax.traj"
    trajectory = Trajectory(str(trajectory_path), mode="a", atoms=atoms)
    if len(trajectory) == 0:
        trajectory.write(atoms)

    cell_filter = FrechetCellFilter(atoms, constant_volume=True)
    optimizer = LBFGS(
        cell_filter,
        restart=str(workdir / "lbfgs.restart"),
        logfile=str(workdir / "relax.log"),
    )

    def save_progress() -> None:
        trajectory.write(atoms)
        write_vasp_grouped(atoms, checkpoint)

    optimizer.attach(save_progress, interval=int(payload["trajectory_interval"]))
    fmax = float(payload["fmax"])
    stress_limit_gpa = float(payload["max_deviatoric_stress_gpa"])
    stress_force_scale = stress_limit_gpa / 160.21766208 * target_volume / len(atoms)
    optimizer_threshold = min(fmax, stress_force_scale)
    optimizer_converged = bool(
        optimizer.run(fmax=optimizer_threshold, steps=int(payload["max_steps"]))
    )
    trajectory.write(atoms)
    trajectory.close()
    final_path = workdir / "POSCAR-relaxed"
    write_vasp_grouped(atoms, final_path)
    forces = np.asarray(atoms.get_forces(), dtype=float)
    stress = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    normal = stress[:3]
    mean_normal = float(np.mean(normal))
    deviatoric = stress.copy()
    deviatoric[:3] -= mean_normal
    max_force = float(np.linalg.norm(forces, axis=1).max())
    max_deviatoric_gpa = float(np.max(np.abs(deviatoric)) * 160.21766208)
    actual_volume = float(atoms.get_volume())
    volume_error = abs(actual_volume - target_volume) / target_volume
    converged = bool(
        optimizer_converged
        and max_force <= fmax
        and max_deviatoric_gpa <= stress_limit_gpa
        and volume_error <= tolerance
    )
    result = {
        "converged": converged,
        "optimizer_converged": optimizer_converged,
        "steps": int(optimizer.get_number_of_steps()),
        "energy_ev": float(atoms.get_potential_energy()),
        "fmax_ev_angstrom": max_force,
        "stress_ev_angstrom3": stress.tolist(),
        "max_deviatoric_stress_gpa": max_deviatoric_gpa,
        "target_volume_angstrom3": target_volume,
        "volume_angstrom3": actual_volume,
        "volume_error_relative": volume_error,
        "output": str(final_path),
    }
    atomic_write_json(workdir / "summary.json", result)
    if not converged:
        reasons: list[str] = []
        if not optimizer_converged:
            reasons.append("优化器未收敛")
        if max_force > fmax:
            reasons.append(f"最大力 {max_force:.6g} > {fmax:.6g} eV/A")
        if max_deviatoric_gpa > stress_limit_gpa:
            reasons.append(f"最大偏应力 {max_deviatoric_gpa:.6g} > {stress_limit_gpa:.6g} GPa")
        if volume_error > tolerance:
            reasons.append(f"相对体积误差 {volume_error:.3g} > {tolerance:.3g}")
        raise RuntimeError("QHA 固定体积弛豫未通过: " + "; ".join(reasons))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["validate", "forces", "relax", "qha-relax"])
    parser.add_argument("payload", type=Path)
    args = parser.parse_args(argv)
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    try:
        if args.operation == "validate":
            result = validate(payload)
        elif args.operation == "forces":
            result = calculate_forces(payload)
        elif args.operation == "relax":
            result = relax(payload)
        else:
            result = qha_relax(payload)
        output = payload.get("result_file")
        if output:
            atomic_write_json(Path(output), result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

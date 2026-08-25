from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .deepmd_worker import _atomic_npz, make_calculator
from .structure import phonopy_to_ase
from .util import atomic_write_json, ensure_finite


def _checkpoint_valid(path: Path, n_atoms: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as data:
            energy = float(data["energy"])
            return (
                data["forces"].shape == (n_atoms, 3)
                and np.all(np.isfinite(data["forces"]))
                and np.isfinite(energy)
            )
    except Exception:
        return False


def _evaluate(calc: Any, cell: Any, checkpoint: Path) -> None:
    atoms = phonopy_to_ase(cell)
    atoms.calc = calc
    forces = np.asarray(atoms.get_forces(), dtype=float)
    energy = float(atoms.get_potential_energy())
    ensure_finite(forces, checkpoint.stem + " forces")
    ensure_finite([energy], checkpoint.stem + " energy")
    _atomic_npz(checkpoint, forces=forces, energy=np.asarray(energy))


def run(payload: dict[str, Any]) -> dict[str, Any]:
    from phono3py import load

    ph3 = load(payload["phono3py_yaml"], produce_fc=False, is_nac=False)
    root = Path(payload["checkpoint_dir"])
    fc3_dir = root / "fc3"
    fc2_dir = root / "fc2"
    residual_dir = root / "residual"
    for directory in (fc3_dir, fc2_dir, residual_dir):
        directory.mkdir(parents=True, exist_ok=True)
    calc = make_calculator(payload["model"], payload.get("head"))

    fc3_cells = ph3.supercells_with_displacements
    fc2_cells = ph3.phonon_supercells_with_displacements if ph3.phonon_supercell_matrix is not None else []
    total = len(fc3_cells) + len(fc2_cells)
    if payload.get("subtract_residual_forces"):
        total += 2 if ph3.phonon_supercell_matrix is not None else 1
    completed = 0

    if payload.get("subtract_residual_forces"):
        residual_fc3 = residual_dir / "fc3.npz"
        if not _checkpoint_valid(residual_fc3, len(ph3.supercell)):
            _evaluate(calc, ph3.supercell, residual_fc3)
        completed += 1
        print(f"anh forces {completed}/{total} residual-fc3", flush=True)
        if ph3.phonon_supercell_matrix is not None:
            residual_fc2 = residual_dir / "fc2.npz"
            if not _checkpoint_valid(residual_fc2, len(ph3.phonon_supercell)):
                _evaluate(calc, ph3.phonon_supercell, residual_fc2)
            completed += 1
            print(f"anh forces {completed}/{total} residual-fc2", flush=True)

    for index, cell in enumerate(fc3_cells, start=1):
        checkpoint = fc3_dir / f"disp-{index:05d}.npz"
        if not _checkpoint_valid(checkpoint, len(cell)):
            _evaluate(calc, cell, checkpoint)
        completed += 1
        print(f"anh forces {completed}/{total} fc3-{index:05d}", flush=True)

    for index, cell in enumerate(fc2_cells, start=1):
        checkpoint = fc2_dir / f"disp-{index:05d}.npz"
        if not _checkpoint_valid(checkpoint, len(cell)):
            _evaluate(calc, cell, checkpoint)
        completed += 1
        print(f"anh forces {completed}/{total} fc2-{index:05d}", flush=True)

    result = {
        "fc3_displacements": len(fc3_cells),
        "fc2_displacements": len(fc2_cells),
        "residual_evaluations": (2 if ph3.phonon_supercell_matrix is not None else 1)
        if payload.get("subtract_residual_forces") else 0,
        "completed": completed,
        "total": total,
    }
    atomic_write_json(root / "worker-summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    args = parser.parse_args(argv)
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    result = run(payload)
    atomic_write_json(Path(payload["result_file"]), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

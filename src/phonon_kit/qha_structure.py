from __future__ import annotations

import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ase.formula import Formula

from .qha_config import QHAPhononConfig
from .structure import ase_to_phonopy, phonopy_to_ase, read_structure, write_vasp_grouped
from .util import atomic_write_json


def composition_metadata(path: Path) -> dict[str, Any]:
    atoms = read_structure(path)
    counts = Counter(atoms.get_chemical_symbols())
    divisor = math.gcd(*counts.values())
    reduced = {symbol: count // divisor for symbol, count in sorted(counts.items())}
    formula = Formula.from_dict(reduced).format("periodic")
    return {
        "composition": dict(sorted(counts.items())),
        "reduced_composition": reduced,
        "reduced_key": tuple(reduced.items()),
        "formula_unit": formula,
        "formula_units_in_input_cell": divisor,
        "atoms_per_formula_unit": sum(reduced.values()),
        "n_atoms": len(atoms),
        "volume_angstrom3": float(atoms.get_volume()),
    }


def scale_structure(source: Path, destination: Path, ratio: float) -> dict[str, Any]:
    atoms = read_structure(source)
    source_volume = float(atoms.get_volume())
    scale = float(ratio) ** (1.0 / 3.0)
    atoms.set_cell(np.asarray(atoms.cell) * scale, scale_atoms=True)
    atoms.wrap()
    write_vasp_grouped(atoms, destination)
    result = {
        "ratio": float(ratio),
        "linear_scale": scale,
        "source_volume_angstrom3": source_volume,
        "target_volume_angstrom3": source_volume * float(ratio),
        "actual_volume_angstrom3": float(atoms.get_volume()),
        "n_atoms": len(atoms),
    }
    atomic_write_json(destination.with_suffix(destination.suffix + ".json"), result)
    return result


def make_phonon(structure: Path, settings: QHAPhononConfig, *, displacements: bool):
    from phonopy import Phonopy
    from phonopy.structure.cells import PrimitiveMatrixAutoDefaultWarning

    atoms = read_structure(structure)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrimitiveMatrixAutoDefaultWarning)
        phonon = Phonopy(
            ase_to_phonopy(atoms),
            supercell_matrix=np.diag(settings.supercell),
            primitive_matrix="auto",
            symprec=settings.symmetry_tolerance,
        )
    if displacements:
        phonon.generate_displacements(distance=settings.displacement_angstrom)
    return phonon


def estimate_displacements(structure: Path, settings: QHAPhononConfig) -> dict[str, Any]:
    phonon = make_phonon(structure, settings, displacements=True)
    cells = phonon.supercells_with_displacements
    if not cells:
        raise RuntimeError(f"Phonopy 没有为 {structure} 生成位移超胞")
    return {
        "n_displacements": len(cells),
        "n_atoms_unitcell": len(phonon.unitcell),
        "n_atoms_primitive": len(phonon.primitive),
        "n_atoms_supercell": len(cells[0]),
        "primitive_volume_angstrom3": float(phonon.primitive.volume),
    }


def generate_qha_displacements(structure: Path, settings: QHAPhononConfig, outdir: Path) -> dict[str, Any]:
    phonon = make_phonon(structure, settings, displacements=True)
    cells = phonon.supercells_with_displacements
    if not cells:
        raise RuntimeError(f"Phonopy 没有为 {structure} 生成位移超胞")
    outdir.mkdir(parents=True, exist_ok=True)
    yaml_path = outdir / "phonopy_disp.yaml"
    phonon.save(filename=str(yaml_path))
    write_vasp_grouped(phonopy_to_ase(phonon.unitcell), outdir / "POSCAR-unitcell")
    write_vasp_grouped(phonopy_to_ase(phonon.supercell), outdir / "SPOSCAR")
    for index, cell in enumerate(cells, start=1):
        write_vasp_grouped(
            phonopy_to_ase(cell),
            outdir / f"POSCAR-{index:04d}",
            outdir / f"atom-map-{index:04d}.json",
        )
    manifest = {
        "phonopy_yaml": str(yaml_path),
        "source_structure": str(structure),
        "supercell": list(settings.supercell),
        "mesh": list(settings.mesh),
        "displacement_angstrom": settings.displacement_angstrom,
        "symmetry_tolerance": settings.symmetry_tolerance,
        "n_displacements": len(cells),
        "n_atoms_unitcell": len(phonon.unitcell),
        "n_atoms_primitive": len(phonon.primitive),
        "n_atoms_supercell": len(cells[0]),
        "unitcell_volume_angstrom3": float(phonon.unitcell.volume),
        "primitive_volume_angstrom3": float(phonon.primitive.volume),
        "resolved_primitive_matrix": np.asarray(phonon.primitive_matrix, dtype=float).tolist(),
    }
    atomic_write_json(outdir / "manifest.json", manifest)
    return manifest

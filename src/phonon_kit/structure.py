from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np

from .config import Config
from .util import atomic_write_json


def read_structure(path: Path):
    from ase.io import read

    atoms = read(str(path), index=-1)
    if len(atoms) == 0:
        raise ValueError(f"结构不含原子: {path}")
    if atoms.cell.rank != 3 or atoms.get_volume() <= 0:
        raise ValueError(f"声子计算要求三维非奇异晶胞: {path}")
    atoms.pbc = True
    return atoms


def write_vasp_grouped(atoms, path: Path, mapping_path: Path | None = None) -> dict[str, Any]:
    """Write a VASP POSCAR grouped by first-seen species and persist atom mapping.

    ``written_to_canonical[i]`` gives the canonical atom index corresponding to
    row i in the written POSCAR. This makes VASP-force reordering explicit.
    """
    from ase.io import write

    symbols = atoms.get_chemical_symbols()
    order: list[str] = []
    for symbol in symbols:
        if symbol not in order:
            order.append(symbol)
    indices = [index for symbol in order for index, item in enumerate(symbols) if item == symbol]
    sorted_atoms = atoms[indices]
    sorted_atoms.set_cell(atoms.cell)
    sorted_atoms.pbc = atoms.pbc
    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), sorted_atoms, format="vasp", direct=True, vasp5=True, sort=False)
    inverse = [0] * len(indices)
    for written, canonical in enumerate(indices):
        inverse[canonical] = written
    mapping = {
        "written_to_canonical": indices,
        "canonical_to_written": inverse,
        "symbols_written": sorted_atoms.get_chemical_symbols(),
        "symbols_canonical": symbols,
    }
    if mapping_path is not None:
        atomic_write_json(mapping_path, mapping)
    return mapping


def normalize_structure(source: Path, destination: Path) -> dict[str, Any]:
    atoms = read_structure(source)
    write_vasp_grouped(atoms, destination)
    return {
        "source": str(source),
        "normalized": str(destination),
        "formula": atoms.get_chemical_formula(mode="hill"),
        "n_atoms": len(atoms),
        "volume_angstrom3": float(atoms.get_volume()),
        "symbols": atoms.get_chemical_symbols(),
    }


def ase_to_phonopy(atoms):
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=np.asarray(atoms.cell, dtype=float),
        scaled_positions=np.asarray(atoms.get_scaled_positions(), dtype=float),
    )


def phonopy_to_ase(cell):
    from ase import Atoms

    return Atoms(
        symbols=list(cell.symbols),
        cell=np.asarray(cell.cell, dtype=float),
        scaled_positions=np.asarray(cell.scaled_positions, dtype=float),
        pbc=True,
    )


def generate_displacements(config: Config, canonical_poscar: Path, outdir: Path) -> dict[str, Any]:
    from phonopy import Phonopy
    from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
    from phonopy.structure.cells import PrimitiveMatrixAutoDefaultWarning

    atoms = read_structure(canonical_poscar)
    unitcell = ase_to_phonopy(atoms)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrimitiveMatrixAutoDefaultWarning)
        phonon = Phonopy(
            unitcell,
            supercell_matrix=np.diag(config.phonon.supercell),
            primitive_matrix="auto",
            symprec=config.phonon.symmetry_tolerance,
        )
    phonon.generate_displacements(distance=config.phonon.displacement_angstrom)
    supercells = phonon.supercells_with_displacements
    if not supercells:
        raise RuntimeError("Phonopy 没有生成任何位移超胞")
    outdir.mkdir(parents=True, exist_ok=True)
    yaml_path = outdir / "phonopy_disp.yaml"
    phonon.save(filename=str(yaml_path))
    write_vasp_grouped(phonopy_to_ase(phonon.unitcell), outdir / "POSCAR-unitcell")
    write_vasp_grouped(phonopy_to_ase(phonon.supercell), outdir / "SPOSCAR")
    for index, cell in enumerate(supercells, start=1):
        write_vasp_grouped(
            phonopy_to_ase(cell),
            outdir / f"POSCAR-{index:04d}",
            outdir / f"atom-map-{index:04d}.json",
        )

    bands, labels, connections = get_band_qpoints_by_seekpath(
        phonon.primitive,
        npoints=config.phonon.band.points_per_segment,
        is_const_interval=True,
    )
    band_path = {
        "bands": [np.asarray(band, dtype=float).tolist() for band in bands],
        "labels": list(labels),
        "path_connections": [bool(value) for value in connections],
        "points_per_segment": config.phonon.band.points_per_segment,
    }
    atomic_write_json(outdir / "band_path.json", band_path)
    manifest = {
        "phonopy_yaml": str(yaml_path),
        "source_structure": str(canonical_poscar),
        "supercell": list(config.phonon.supercell),
        "displacement_angstrom": config.phonon.displacement_angstrom,
        "symmetry_tolerance": config.phonon.symmetry_tolerance,
        "n_displacements": len(supercells),
        "n_atoms_unitcell": len(phonon.unitcell),
        "n_atoms_supercell": len(supercells[0]),
        "resolved_primitive_matrix": np.asarray(phonon.primitive_matrix, dtype=float).tolist(),
    }
    atomic_write_json(outdir / "manifest.json", manifest)
    return manifest


def load_displacement_phonon(path: Path):
    from phonopy import load

    return load(str(path), produce_fc=False)

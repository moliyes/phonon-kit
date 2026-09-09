from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from phonon_kit.structure import normalize_structure, write_vasp_grouped
from phonon_kit.util import load_json


def test_explicit_vasp_atom_mapping(tmp_path: Path):
    atoms = Atoms(
        symbols=["Si", "O", "Si", "O"],
        scaled_positions=[[0, 0, 0], [0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.8, 0.8, 0.8]],
        cell=np.eye(3) * 5,
        pbc=True,
    )
    write_vasp_grouped(atoms, tmp_path / "POSCAR", tmp_path / "map.json")
    mapping = load_json(tmp_path / "map.json")
    assert mapping["written_to_canonical"] == [0, 2, 1, 3]
    raw = np.arange(12).reshape(4, 3)
    canonical = np.empty_like(raw)
    canonical[np.asarray(mapping["written_to_canonical"])] = raw
    assert np.array_equal(canonical[0], raw[0])
    assert np.array_equal(canonical[2], raw[1])


def test_normalize_snaps_cell_boundary_coordinates(tmp_path: Path):
    source = tmp_path / "source.vasp"
    source.write_text(
        "Si\n1\n3 0 0\n0 3 0\n0 0 3\nSi\n1\nDirect\n1.0 0.5 0.0\n",
        encoding="utf-8",
    )
    destination = tmp_path / "POSCAR"
    normalize_structure(source, destination)
    coordinate_line = destination.read_text(encoding="utf-8").splitlines()[-1].split()
    assert float(coordinate_line[0]) == 1.0
    assert float(coordinate_line[2]) == 0.0

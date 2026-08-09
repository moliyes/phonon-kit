from __future__ import annotations

from pathlib import Path

import numpy as np

from phonon_kit.initializer import POSCAR
from phonon_kit.qha_config import QHAPhononConfig
from ase import Atoms

from phonon_kit.qha_structure import (
    canonicalize_primitive_structure,
    composition_metadata,
    estimate_displacements,
    generate_qha_displacements,
    scale_structure,
)
from phonon_kit.structure import read_structure, write_vasp_grouped


def test_qha_volume_scaling_and_formula_units(tmp_path: Path):
    source = tmp_path / "POSCAR"
    source.write_text(POSCAR, encoding="utf-8")
    destination = tmp_path / "scaled" / "POSCAR"
    before = read_structure(source).get_volume()
    metadata = scale_structure(source, destination, 0.91)
    after = read_structure(destination).get_volume()
    assert np.isclose(after / before, 0.91)
    assert np.isclose(metadata["linear_scale"], 0.91 ** (1 / 3))
    composition = composition_metadata(source)
    assert composition["formula_units_in_input_cell"] == 2
    assert composition["atoms_per_formula_unit"] == 3


def test_qha_displacement_estimate(tmp_path: Path):
    source = tmp_path / "POSCAR"
    source.write_text(POSCAR, encoding="utf-8")
    estimate = estimate_displacements(source, QHAPhononConfig(supercell=(1, 1, 1), mesh=(4, 4, 4)))
    assert estimate["n_displacements"] > 0
    assert estimate["n_atoms_unitcell"] == 6


def test_qha_uses_one_canonical_primitive_basis_for_all_volumes(tmp_path: Path):
    conventional = Atoms(
        "Si2",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0]],
        cell=[[4, 0, 0], [0, 4, 0], [0, 0, 5]],
        pbc=True,
    )
    source = tmp_path / "POSCAR-conventional"
    write_vasp_grouped(conventional, source)
    settings = QHAPhononConfig(supercell=(1, 1, 1), mesh=(4, 4, 4))
    reference = tmp_path / "reference" / "POSCAR-primitive"
    metadata = canonicalize_primitive_structure(source, reference, settings)
    assert metadata["source_n_atoms"] == 2
    assert metadata["primitive_n_atoms"] == 1

    primitive_counts = []
    for index, ratio in enumerate((0.9, 1.0, 1.1)):
        scaled = tmp_path / f"scaled-{index}" / "POSCAR"
        scale_structure(reference, scaled, ratio)
        manifest = generate_qha_displacements(scaled, settings, tmp_path / f"phonon-{index}")
        primitive_counts.append(manifest["n_atoms_primitive"])
    assert primitive_counts == [1, 1, 1]

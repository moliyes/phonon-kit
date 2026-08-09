from __future__ import annotations

from pathlib import Path

import numpy as np

from phonon_kit.initializer import POSCAR
from phonon_kit.qha_config import QHAPhononConfig
from phonon_kit.qha_structure import composition_metadata, estimate_displacements, scale_structure
from phonon_kit.structure import read_structure


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

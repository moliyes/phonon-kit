from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from phonon_kit import deepmd_worker
from phonon_kit.initializer import POSCAR
from phonon_kit.structure import read_structure


class ZeroCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3)),
            "stress": np.zeros(6),
        }


def test_deepmd_qha_relax_preserves_volume(tmp_path: Path, monkeypatch):
    structure = tmp_path / "POSCAR"
    structure.write_text(POSCAR, encoding="utf-8")
    target_volume = read_structure(structure).get_volume()
    monkeypatch.setattr(deepmd_worker, "make_calculator", lambda model, head: ZeroCalculator())
    result = deepmd_worker.qha_relax({
        "structure": str(structure),
        "model": "fake.pth",
        "head": None,
        "workdir": str(tmp_path / "relax"),
        "target_volume_angstrom3": target_volume,
        "fmax": 0.01,
        "max_deviatoric_stress_gpa": 0.1,
        "max_steps": 10,
        "trajectory_interval": 10,
        "volume_tolerance_relative": 1.0e-8,
    })
    assert result["converged"]
    assert result["volume_error_relative"] < 1.0e-8
    assert (tmp_path / "relax" / "POSCAR-relaxed").is_file()

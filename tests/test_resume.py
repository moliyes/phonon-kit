from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from phonon_kit import deepmd_worker
from phonon_kit.config import RelaxationConfig, load_config
from phonon_kit.runner import _canonical_structure, _relax_if_needed
from phonon_kit.state import StateStore, choose_run
from phonon_kit.structure import generate_displacements


class CountingCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]
    calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        type(self).calls += 1
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3)),
            "stress": np.zeros(6),
        }


def test_deepmd_worker_reuses_per_displacement_checkpoints(config_path: Path, tmp_path: Path, monkeypatch):
    config = load_config(config_path)
    displacement_dir = tmp_path / "displacements"
    manifest = generate_displacements(config, config.structure.file, displacement_dir)
    monkeypatch.setattr(deepmd_worker, "make_calculator", lambda model, head: CountingCalculator())
    payload = {
        "phonopy_yaml": str(displacement_dir / "phonopy_disp.yaml"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "model": "unused.pth",
        "head": None,
    }
    CountingCalculator.calls = 0
    deepmd_worker.calculate_forces(payload)
    assert CountingCalculator.calls == manifest["n_displacements"]
    CountingCalculator.calls = 0
    deepmd_worker.calculate_forces(payload)
    assert CountingCalculator.calls == 0
    (tmp_path / "checkpoints" / "disp-0001.npz").unlink()
    deepmd_worker.calculate_forces(payload)
    assert CountingCalculator.calls == 1


def test_relaxed_structure_is_shared_and_not_repeated(config_path: Path, monkeypatch):
    config = load_config(config_path)
    config = replace(config, relaxation=RelaxationConfig(enabled=True, method="dp"))
    paths, _, _ = choose_run(config, ["dp"])
    store = StateStore(paths)
    canonical = _canonical_structure(config, paths, store)
    calls = []

    def fake_relax(method, structure, workdir, logdir, **kwargs):
        calls.append(method.name)
        workdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(structure, workdir / "POSCAR-relaxed")
        return {"converged": True, "output": str(workdir / "POSCAR-relaxed")}

    monkeypatch.setattr("phonon_kit.runner.relax_structure", fake_relax)
    first = _relax_if_needed(config, paths, store, canonical)
    second = _relax_if_needed(config, paths, store, canonical)
    assert first == second == paths.canonical / "POSCAR-relaxed"
    assert calls == ["dp"]


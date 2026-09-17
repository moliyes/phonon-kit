from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from phonon_kit.analysis import analyze_method, result_is_analyzed
from phonon_kit.config import BandConfig, UnfoldingConfig, load_config
from phonon_kit.structure import generate_displacements


def test_harmonic_analysis_end_to_end(config_path: Path, tmp_path: Path):
    config = load_config(config_path)
    displacement_dir = tmp_path / "displacements"
    manifest = generate_displacements(config, config.structure.file, displacement_dir)
    from phonopy import load

    phonon = load(str(displacement_dir / "phonopy_disp.yaml"), produce_fc=False)
    force_sets = []
    dataset = phonon.dataset
    for displaced in dataset["first_atoms"]:
        force = np.zeros((manifest["n_atoms_supercell"], 3), dtype=float)
        force[int(displaced["number"])] = -10.0 * np.asarray(displaced["displacement"])
        force_sets.append(force)
    resultdir = tmp_path / "result"
    resultdir.mkdir()
    np.save(resultdir / "forces.npy", np.asarray(force_sets))
    summary = analyze_method(config, "harmonic", displacement_dir / "phonopy_disp.yaml", resultdir)
    assert summary["n_displacements"] == manifest["n_displacements"]
    assert result_is_analyzed(resultdir)
    assert (resultdir / "thermal_properties.png").is_file()
    assert not list(resultdir.glob("*.pdf"))


def test_harmonic_analysis_runs_optional_unfolding(config_path: Path, tmp_path: Path, monkeypatch):
    config = load_config(config_path)
    reference = tmp_path / "reference.vasp"
    reference.write_text(config.structure.file.read_text(encoding="utf-8"), encoding="utf-8")
    phonon_config = replace(
        config.phonon,
        band=BandConfig(path=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)), points_per_segment=3),
        unfolding=UnfoldingConfig(reference, ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
    )
    config = replace(config, phonon=phonon_config)
    displacement_dir = tmp_path / "displacements"
    manifest = generate_displacements(config, config.structure.file, displacement_dir)
    from phonopy import load

    phonon = load(str(displacement_dir / "phonopy_disp.yaml"), produce_fc=False)
    force_sets = []
    for displaced in phonon.dataset["first_atoms"]:
        force = np.zeros((manifest["n_atoms_supercell"], 3), dtype=float)
        force[int(displaced["number"])] = -10.0 * np.asarray(displaced["displacement"])
        force_sets.append(force)
    resultdir = tmp_path / "result"
    resultdir.mkdir()
    np.save(resultdir / "forces.npy", np.asarray(force_sets))
    seen = []

    def fake_unfold(run_config, run_phonon, output):
        seen.append((run_config, run_phonon, output))
        return {"n_qpoints": 3, "maximum_weight_sum_error": 0.0}

    monkeypatch.setattr("phonon_kit.unfolding.run_unfolding", fake_unfold)
    summary = analyze_method(config, "harmonic", displacement_dir / "phonopy_disp.yaml", resultdir)
    assert len(seen) == 1
    assert summary["unfolding"]["n_qpoints"] == 3

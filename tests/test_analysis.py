from __future__ import annotations

from pathlib import Path

import numpy as np

from phonon_kit.analysis import analyze_method, result_is_analyzed
from phonon_kit.config import load_config
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

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from phonon_kit.analysis import analyze_method
from phonon_kit.config import load_config


@pytest.mark.integration
def test_reanalyze_portable_legacy_result(config_path: Path, tmp_path: Path):
    base = Path("/home/begin/phonon_workflow_portable_8.9/structures/task.000007_vol095/3_phonon_calc")
    displacement = base / "displacements_2x2x2" / "phonopy_disp.yaml"
    old_forces = base / "dft_reference" / "forces.npy"
    if not displacement.is_file() or not old_forces.is_file():
        pytest.skip("portable legacy fixture is not available")
    config = load_config(config_path)
    config = replace(config, phonon=replace(config.phonon, supercell=(2, 2, 2)))
    result = tmp_path / "legacy-result"
    result.mkdir()
    shutil.copy2(old_forces, result / "forces.npy")
    summary = analyze_method(config, "legacy-dft", displacement, result)
    assert np.load(result / "forces.npy").shape == (36, 48, 3)
    assert summary["n_displacements"] == 36
    assert summary["n_atoms_supercell"] == 48
    assert (result / "phonon_band.png").is_file()


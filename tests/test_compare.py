from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from phonon_kit.compare import compare_methods
from phonon_kit.config import DispatcherConfig, VaspMethod, load_config
from phonon_kit.util import atomic_write_json


def test_dpa_dft_metrics_and_png(config_path: Path, tmp_path: Path):
    config = load_config(config_path)
    executor = DispatcherConfig("dpdispatcher", tmp_path / "machine.json", tmp_path / "resources.json", "vasp_std")
    dft = VaspMethod("dft", "vasp", tmp_path / "vasp", executor)
    config = replace(config, methods={"dp": config.methods["dp"], "dft": dft})
    results = tmp_path / "results"
    dft_dir = results / "dft"
    dpa_dir = results / "dp"
    dft_dir.mkdir(parents=True)
    dpa_dir.mkdir(parents=True)
    reference_forces = np.arange(12, dtype=float).reshape(2, 2, 3) / 10
    predicted_forces = reference_forces + 0.01
    distances = np.asarray([0.0, 0.5, 1.0])
    reference_frequencies = np.asarray([[0.0, 1.0], [0.2, 1.2], [0.0, 1.1]])
    predicted_frequencies = reference_frequencies + 0.02
    np.save(dft_dir / "forces.npy", reference_forces)
    np.save(dpa_dir / "forces.npy", predicted_forces)
    np.savez(dft_dir / "band_data.npz", distances=distances, frequencies=reference_frequencies)
    np.savez(dpa_dir / "band_data.npz", distances=distances, frequencies=predicted_frequencies)
    atomic_write_json(dft_dir / "summary.json", {"min_frequency_thz": 0.0})
    atomic_write_json(dpa_dir / "summary.json", {"min_frequency_thz": 0.02})
    result = compare_methods(config, ["dp", "dft"], results)
    assert result is not None
    assert result["models"][0]["force_mae_mev_angstrom"] == pytest.approx(10.0)
    assert (results / "comparison" / "metrics.csv").is_file()
    assert (results / "comparison" / "phonon_band_compare.png").is_file()
    assert (results / "comparison" / "force_comparison.png").is_file()
    assert (results / "comparison" / "frequency_error.png").is_file()

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from phonon_kit.config import load_config
from phonon_kit.cli import main
from phonon_kit.structure import generate_displacements
from phonon_kit.unfolding import CHECKPOINT_FILE, spectral_map, unfold_result
from phonon_kit.util import atomic_write_json

from conftest import write_config


def _prepared_result(tmp_path: Path, *, three_vertices: bool = False):
    config_path = write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("  supercell: [1, 1, 1]", "  supercell: [2, 1, 1]")
    text = text.replace("  primitive: auto", "  primitive: P")
    config_path.write_text(text, encoding="utf-8")
    base_config = load_config(config_path)
    displacement_dir = tmp_path / "displacements"
    generate_displacements(base_config, base_config.structure.file, displacement_dir)
    shutil.copy2(displacement_dir / "SPOSCAR", tmp_path / "SPOSCAR-Ref.vasp")

    from phonopy import load

    phonon = load(str(displacement_dir / "phonopy_disp.yaml"), produce_fc=False)
    force_constants = np.zeros((2, 2, 3, 3), dtype=float)
    spring = np.diag([2.0, 3.0, 5.0])
    force_constants[0, 0] = spring
    force_constants[1, 1] = spring
    force_constants[0, 1] = -spring
    force_constants[1, 0] = -spring
    phonon.force_constants = force_constants
    resultdir = tmp_path / "result"
    resultdir.mkdir()
    phonon.save(filename=str(resultdir / "phonopy_params.yaml"), settings={"force_constants": True})

    vertices = "[[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]]" if three_vertices else "[[0, 0, 0], [0.5, 0, 0]]"
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("    path: auto", f"    path: {vertices}")
    text = text.replace("    points_per_segment: 11", "    points_per_segment: 3")
    text = text.replace(
        "  mesh: [5, 5, 5]",
        "  mesh: [5, 5, 5]\n"
        "  unfolding:\n"
        "    reference_supercell: SPOSCAR-Ref.vasp\n"
        "    supercell_matrix: [[2, 0, 0], [0, 1, 0], [0, 0, 1]]\n"
        "    mapping_tolerance_angstrom: 0.01",
    )
    config_path.write_text(text, encoding="utf-8")
    return load_config(config_path), resultdir


def test_unfolding_small_supercell_and_cache(tmp_path: Path):
    config, resultdir = _prepared_result(tmp_path)
    first = unfold_result(config, resultdir)
    assert first["n_qpoints"] == 3
    assert first["n_modes"] == 6
    assert first["expected_weight_sum_per_qpoint"] == 3.0
    assert first["maximum_weight_sum_error"] < 1.0e-10
    with np.load(resultdir / "unfolding_data.npz", allow_pickle=False) as data:
        assert data["frequencies"].shape == (3, 6)
        assert np.allclose(data["weights"].sum(axis=1), 3.0)
    assert (resultdir / "phonon_unfolded.png").is_file()
    assert not (resultdir / CHECKPOINT_FILE).exists()
    second = unfold_result(config, resultdir)
    assert second["cache_key"] == first["cache_key"]
    assert second["plot"]["color_scale"] == "linear"


def test_spectral_map_keeps_integrated_weight():
    frequencies = np.asarray([[0.0, 1.0, 2.0], [0.2, 1.2, 2.2]])
    weights = np.asarray([[1.0, 0.5, 1.5], [0.8, 1.2, 1.0]])
    grid, intensity = spectral_map(frequencies, weights)
    assert intensity.shape == (2, len(grid))
    assert np.allclose(np.sum(intensity, axis=1), np.sum(weights, axis=1), atol=1.0e-4)


def test_unfolding_resumes_at_segment_boundary(tmp_path: Path, monkeypatch):
    config, resultdir = _prepared_result(tmp_path, three_vertices=True)
    from phonopy.unfolding import core

    real = core.Unfolding
    calls = 0

    class InterruptSecondSegment:
        def __init__(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("intentional interruption")
            self._inner = real(*args, **kwargs)

        def prepare(self):
            return self._inner.prepare()

        def __iter__(self):
            return iter(self._inner)

        @property
        def frequencies(self):
            return self._inner.frequencies

        @property
        def unfolding_weights(self):
            return self._inner.unfolding_weights

    monkeypatch.setattr(core, "Unfolding", InterruptSecondSegment)
    with pytest.raises(RuntimeError, match="intentional interruption"):
        unfold_result(config, resultdir)
    with np.load(resultdir / CHECKPOINT_FILE, allow_pickle=False) as checkpoint:
        assert int(checkpoint["completed_segments"].item()) == 1

    resumed_calls = 0

    class CountRemaining(real):
        def __init__(self, *args, **kwargs):
            nonlocal resumed_calls
            resumed_calls += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(core, "Unfolding", CountRemaining)
    summary = unfold_result(config, resultdir)
    assert resumed_calls == 1
    assert summary["n_segments"] == 2
    assert not (resultdir / CHECKPOINT_FILE).exists()


def test_unfold_command_backfills_old_run_with_config_override(tmp_path: Path):
    config, resultdir = _prepared_result(tmp_path)
    run = tmp_path / "runs" / "old-001"
    target = run / "results" / "dp"
    target.parent.mkdir(parents=True)
    shutil.move(str(resultdir), target)
    (run / "config.resolved.yaml").write_text("legacy snapshot placeholder\n", encoding="utf-8")
    atomic_write_json(run / "state.json", {"selected_methods": ["dp"]})
    assert main(["unfold", str(run), "--config", str(config.path)]) == 0
    assert (target / "unfolding_data.npz").is_file()

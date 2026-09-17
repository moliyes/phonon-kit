from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .util import atomic_write_json, load_json, sha256_file, sha256_json


DATA_FILE = "unfolding_data.npz"
PLOT_FILE = "phonon_unfolded.png"
SUMMARY_FILE = "unfolding_summary.json"
CHECKPOINT_FILE = ".unfolding_checkpoint.npz"
PLOT_BROADENING_THZ = 0.04
PLOT_FREQUENCY_STEP_THZ = 0.01


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _path_arrays(
    config: Config, reference_cell: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    band = config.phonon.band
    if band.path == "auto":
        raise ValueError("unfolding 要求显式 phonon.band.path")
    vertices = np.asarray(band.path, dtype=float)
    segments = [
        np.linspace(vertices[index], vertices[index + 1], band.points_per_segment)
        for index in range(len(vertices) - 1)
    ]
    reciprocal = np.linalg.inv(reference_cell).T
    distances: list[np.ndarray] = []
    offset = 0.0
    ticks = [0.0]
    for segment in segments:
        cartesian = np.dot(segment, reciprocal)
        steps = np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
        local = np.concatenate(([0.0], np.cumsum(steps))) + offset
        distances.append(local)
        offset = float(local[-1])
        ticks.append(offset)
    labels = np.asarray(band.labels or tuple("" for _ in vertices), dtype=str)
    return segments, np.concatenate(distances), np.asarray(ticks), labels


def _reference_and_mapping(config: Config, phonon) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from phonopy.interface.vasp import read_vasp

    settings = config.phonon.unfolding
    if settings is None:
        raise ValueError("配置未启用 phonon.unfolding")
    ideal = read_vasp(str(settings.reference_supercell))
    actual_cell = np.asarray(phonon.supercell.cell, dtype=float)
    ideal_cell = np.asarray(ideal.cell, dtype=float)
    cell_error = float(np.max(np.abs(actual_cell - ideal_cell)))
    if cell_error > settings.mapping_tolerance_angstrom:
        raise ValueError(
            f"unfolding 参考超胞晶格与力常数超胞不一致，最大差值 {cell_error:.6g} Å"
        )
    actual_positions = np.asarray(phonon.supercell.scaled_positions, dtype=float)
    ideal_positions = np.asarray(ideal.scaled_positions, dtype=float)
    if len(actual_positions) != len(ideal_positions):
        raise ValueError(
            "unfolding 参考超胞原子数与力常数超胞不一致: "
            f"{len(ideal_positions)} != {len(actual_positions)}"
        )
    matrix = np.asarray(settings.supercell_matrix, dtype=int)
    determinant = int(round(np.linalg.det(matrix)))
    if determinant <= 0 or len(ideal_positions) % determinant:
        raise ValueError(
            "unfolding supercell_matrix 与参考超胞原子数不相容: "
            f"n_atoms={len(ideal_positions)}, det={determinant}"
        )

    delta = actual_positions[:, None, :] - ideal_positions[None, :, :]
    delta -= np.rint(delta)
    distances = np.linalg.norm(np.dot(delta, ideal_cell), axis=2)
    mapping = np.argmin(distances, axis=1)
    nearest = distances[np.arange(len(actual_positions)), mapping]
    if len(np.unique(mapping)) != len(mapping):
        duplicates = len(mapping) - len(np.unique(mapping))
        raise ValueError(f"unfolding 原子最近邻映射不是双射，存在 {duplicates} 个重复目标")
    maximum = float(np.max(nearest))
    if maximum > settings.mapping_tolerance_angstrom:
        raise ValueError(
            "unfolding 实际/理想原子映射距离超过容差: "
            f"{maximum:.6g} > {settings.mapping_tolerance_angstrom:.6g} Å"
        )
    reference_cell = np.dot(np.linalg.inv(matrix), ideal_cell)
    report = {
        "n_atoms_supercell": int(len(actual_positions)),
        "n_atoms_reference_cell": int(len(ideal_positions) // determinant),
        "supercell_matrix_determinant": determinant,
        "maximum_cell_difference_angstrom": cell_error,
        "maximum_mapping_distance_angstrom": maximum,
        "mean_mapping_distance_angstrom": float(np.mean(nearest)),
        "mapping_tolerance_angstrom": settings.mapping_tolerance_angstrom,
        "mapping_is_bijective": True,
    }
    # This ordering is locked by regression against the archived SnS3Se13
    # weights. It is the actual-site -> ideal-site permutation expected by the
    # Phonopy 4.4.0 implementation for this one-to-one, vacancy-free case.
    return ideal_positions, mapping, {"reference_cell": reference_cell, **report}


def _full_force_constants(phonon) -> None:
    from phonopy.harmonic.force_constants import compact_fc_to_full_fc

    force_constants = np.asarray(phonon.force_constants, dtype=float)
    if force_constants.shape[0] != force_constants.shape[1]:
        phonon.force_constants = compact_fc_to_full_fc(phonon.primitive, force_constants)


def _cache_key(config: Config, resultdir: Path) -> tuple[str, dict[str, str]]:
    settings = config.phonon.unfolding
    if settings is None:
        raise ValueError("配置未启用 phonon.unfolding")
    phonopy_path = resultdir / "phonopy_params.yaml"
    if not phonopy_path.is_file():
        raise FileNotFoundError(f"缺少含力常数的 phonopy_params.yaml: {phonopy_path}")
    hashes = {
        "phonopy_params_sha256": sha256_file(phonopy_path),
        "reference_supercell_sha256": sha256_file(settings.reference_supercell),
    }
    payload = {
        **hashes,
        "supercell_matrix": settings.supercell_matrix,
        "mapping_tolerance_angstrom": settings.mapping_tolerance_angstrom,
        "band_path": config.phonon.band.path,
        "band_labels": config.phonon.band.labels,
        "points_per_segment": config.phonon.band.points_per_segment,
    }
    return sha256_json(payload), hashes


def _cached_summary(resultdir: Path, cache_key: str) -> dict[str, Any] | None:
    summary_path = resultdir / SUMMARY_FILE
    data_path = resultdir / DATA_FILE
    if not summary_path.is_file() or not data_path.is_file():
        return None
    try:
        summary = load_json(summary_path)
        if summary.get("cache_key") != cache_key:
            return None
        with np.load(data_path, allow_pickle=False) as data:
            if str(data["cache_key"].item()) != cache_key:
                return None
            if data["frequencies"].shape != data["weights"].shape:
                return None
    except (KeyError, OSError, ValueError):
        return None
    return summary


def spectral_map(
    frequencies: np.ndarray,
    weights: np.ndarray,
    *,
    frequency_min: float | None = None,
    frequency_max: float | None = None,
    broadening_thz: float = PLOT_BROADENING_THZ,
    frequency_step_thz: float = PLOT_FREQUENCY_STEP_THZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw unfolding weights to a linearly scaled spectral map."""
    frequencies = np.asarray(frequencies, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if frequencies.shape != weights.shape or frequencies.ndim != 2:
        raise ValueError("unfolding 频率与权重必须是相同形状的二维数组")
    if broadening_thz <= 0 or frequency_step_thz <= 0:
        raise ValueError("谱函数展宽和频率步长必须为正数")
    finite = np.isfinite(frequencies) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        raise ValueError("unfolding 数据没有有限的正谱权重")
    margin = 4.0 * broadening_thz
    low = float(np.min(frequencies[finite]) - margin) if frequency_min is None else frequency_min
    high = float(np.max(frequencies[finite]) + margin) if frequency_max is None else frequency_max
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("谱函数频率范围无效")
    n_bins = max(2, int(np.ceil((high - low) / frequency_step_thz)))
    edges = np.linspace(low, high, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    actual_step = float(edges[1] - edges[0])
    radius = max(1, int(np.ceil(4.0 * broadening_thz / actual_step)))
    offsets = np.arange(-radius, radius + 1, dtype=float) * actual_step
    kernel = np.exp(-0.5 * (offsets / broadening_thz) ** 2)
    kernel /= np.sum(kernel)
    intensity = np.zeros((frequencies.shape[0], n_bins), dtype=float)
    center = len(kernel) // 2
    for q_index in range(frequencies.shape[0]):
        valid = finite[q_index]
        histogram, _ = np.histogram(
            frequencies[q_index, valid],
            bins=edges,
            weights=weights[q_index, valid],
        )
        convolved = np.convolve(histogram, kernel, mode="full")
        intensity[q_index] = convolved[center:center + n_bins]
    return centers, intensity


def _linear_vmax(intensity: np.ndarray) -> float:
    positive = np.asarray(intensity, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if not len(positive):
        raise ValueError("谱函数没有可绘制的正强度")
    return max(float(np.quantile(positive, 0.995)), np.finfo(float).eps)


def plot_unfolding(resultdir: Path) -> Path:
    from .analysis import _matplotlib

    data_path = resultdir / DATA_FILE
    if not data_path.is_file():
        raise FileNotFoundError(f"缺少 unfolding 数值结果: {data_path}")
    with np.load(data_path, allow_pickle=False) as data:
        distances = np.asarray(data["distances"], dtype=float)
        frequencies = np.asarray(data["frequencies"], dtype=float)
        weights = np.asarray(data["weights"], dtype=float)
        tick_positions = np.asarray(data["tick_positions"], dtype=float)
        tick_labels = [str(item) for item in data["tick_labels"]]
    if frequencies.shape != weights.shape or frequencies.shape[0] != len(distances):
        raise ValueError("unfolding_data.npz 中数组形状不一致")

    frequency_grid, intensity = spectral_map(frequencies, weights)
    keep = np.concatenate(([True], np.diff(distances) > 1.0e-12))
    plt = _matplotlib()

    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    plotted = axis.pcolormesh(
        distances[keep],
        frequency_grid,
        intensity[keep].T,
        shading="nearest",
        cmap="Greys",
        vmin=0.0,
        vmax=_linear_vmax(intensity),
        rasterized=True,
    )
    labels = [
        r"$\Gamma$" if label.upper() in {"G", "GAMMA", "Γ"} else label
        for label in tick_labels
    ]
    for position in tick_positions:
        axis.axvline(position, color="#999999", lw=0.55)
    axis.axhline(0.0, color="#777777", lw=0.8)
    axis.set_xlim(float(distances[0]), float(distances[-1]))
    axis.set_xticks(tick_positions, labels)
    axis.set_ylabel("Frequency (THz)")
    axis.grid(axis="y", alpha=0.15)
    fig.colorbar(plotted, ax=axis, label="Spectral intensity (linear scale)")
    fig.tight_layout()
    output = resultdir / PLOT_FILE
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def run_unfolding(config: Config, phonon, resultdir: Path) -> dict[str, Any]:
    import phonopy
    from phonopy.unfolding.core import Unfolding

    settings = config.phonon.unfolding
    if settings is None:
        raise ValueError("配置未启用 phonon.unfolding")
    resultdir.mkdir(parents=True, exist_ok=True)
    cache_key, hashes = _cache_key(config, resultdir)
    cached = _cached_summary(resultdir, cache_key)
    if cached is not None:
        plot_unfolding(resultdir)
        cached["plot"] = {
            "type": "Gaussian-broadened spectral intensity",
            "color_scale": "linear",
            "broadening_thz": PLOT_BROADENING_THZ,
            "frequency_step_thz": PLOT_FREQUENCY_STEP_THZ,
        }
        atomic_write_json(resultdir / SUMMARY_FILE, cached)
        return cached

    ideal_positions, mapping, mapping_report = _reference_and_mapping(config, phonon)
    reference_cell = np.asarray(mapping_report.pop("reference_cell"), dtype=float)
    segments, distances, tick_positions, tick_labels = _path_arrays(config, reference_cell)
    matrix = np.asarray(settings.supercell_matrix, dtype=int)
    expected_weight_sum = float(3 * mapping_report["n_atoms_reference_cell"])
    checkpoint_path = resultdir / CHECKPOINT_FILE
    completed_segments = 0
    frequency_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    if checkpoint_path.is_file():
        try:
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                if str(checkpoint["cache_key"].item()) == cache_key:
                    completed_segments = int(checkpoint["completed_segments"].item())
                    if not 0 <= completed_segments <= len(segments):
                        raise ValueError("invalid completed segment count")
                    if completed_segments:
                        prior_frequencies = np.asarray(checkpoint["frequencies"], dtype=float)
                        prior_weights = np.asarray(checkpoint["weights"], dtype=float)
                        expected_rows = sum(len(item) for item in segments[:completed_segments])
                        if (
                            prior_frequencies.shape != prior_weights.shape
                            or prior_frequencies.shape[0] != expected_rows
                        ):
                            raise ValueError("invalid checkpoint arrays")
                        offset = 0
                        for segment in segments[:completed_segments]:
                            end = offset + len(segment)
                            frequency_parts.append(prior_frequencies[offset:end])
                            weight_parts.append(prior_weights[offset:end])
                            offset = end
        except (KeyError, OSError, ValueError):
            completed_segments = 0
            frequency_parts = []
            weight_parts = []

    _full_force_constants(phonon)
    total_qpoints = sum(len(item) for item in segments)
    finished_qpoints = sum(len(item) for item in segments[:completed_segments])
    for segment_index, qpoints in enumerate(segments[completed_segments:], start=completed_segments):
        calculation = Unfolding(
            phonon,
            supercell_matrix=matrix,
            ideal_positions=ideal_positions,
            atom_mapping=mapping.tolist(),
            qpoints=qpoints,
        )
        calculation.prepare()
        for local_index, _ in enumerate(calculation, start=1):
            print(
                f"unfolding q-point {finished_qpoints + local_index}/{total_qpoints}",
                flush=True,
            )
        frequencies = np.asarray(calculation.frequencies, dtype=float)
        weights = np.asarray(calculation.unfolding_weights, dtype=float)
        frequency_parts.append(frequencies)
        weight_parts.append(weights)
        finished_qpoints += len(qpoints)
        complete = segment_index + 1
        _atomic_savez(
            checkpoint_path,
            cache_key=np.asarray(cache_key),
            completed_segments=np.asarray(complete),
            frequencies=np.concatenate(frequency_parts, axis=0),
            weights=np.concatenate(weight_parts, axis=0),
        )

    frequencies = np.concatenate(frequency_parts, axis=0)
    weights = np.concatenate(weight_parts, axis=0)
    reference_qpoints = np.concatenate(segments, axis=0)
    supercell_qpoints = np.dot(reference_qpoints, matrix)
    supercell_qpoints -= np.rint(supercell_qpoints)
    sums = np.sum(weights, axis=1)
    maximum_sum_error = float(np.max(np.abs(sums - expected_weight_sum)))
    if maximum_sum_error > 1.0e-5:
        raise RuntimeError(
            "unfolding 谱权重和规则失败: "
            f"max error={maximum_sum_error:.6g}, expected={expected_weight_sum:.6g}"
        )
    _atomic_savez(
        resultdir / DATA_FILE,
        cache_key=np.asarray(cache_key),
        reference_qpoints=reference_qpoints,
        supercell_qpoints=supercell_qpoints,
        distances=distances,
        frequencies=frequencies,
        weights=weights,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
    )
    summary = {
        "schema_version": 1,
        "algorithm": "Allen et al. via phonopy.unfolding.core.Unfolding",
        "phonopy_version": phonopy.__version__,
        "cache_key": cache_key,
        **hashes,
        "reference_supercell": str(settings.reference_supercell),
        "supercell_matrix": matrix.tolist(),
        "n_segments": len(segments),
        "n_qpoints": int(len(reference_qpoints)),
        "n_modes": int(frequencies.shape[1]),
        "expected_weight_sum_per_qpoint": expected_weight_sum,
        "minimum_weight_sum": float(np.min(sums)),
        "maximum_weight_sum": float(np.max(sums)),
        "maximum_weight_sum_error": maximum_sum_error,
        "mapping": mapping_report,
        "plot": {
            "type": "Gaussian-broadened spectral intensity",
            "color_scale": "linear",
            "broadening_thz": PLOT_BROADENING_THZ,
            "frequency_step_thz": PLOT_FREQUENCY_STEP_THZ,
        },
    }
    atomic_write_json(resultdir / SUMMARY_FILE, summary)
    plot_unfolding(resultdir)
    try:
        checkpoint_path.unlink()
    except FileNotFoundError:
        pass
    return summary


def unfold_result(config: Config, resultdir: Path) -> dict[str, Any]:
    from phonopy import load

    phonopy_path = resultdir / "phonopy_params.yaml"
    if not phonopy_path.is_file():
        raise FileNotFoundError(f"缺少含力常数的 phonopy_params.yaml: {phonopy_path}")
    phonon = load(str(phonopy_path))
    return run_unfolding(config, phonon, resultdir)

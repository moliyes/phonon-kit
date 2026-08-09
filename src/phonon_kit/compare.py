from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, DeepMDMethod, VaspMethod
from .util import atomic_write_json, load_json

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phonon-kit-matplotlib")


def _plt():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/phonon-kit-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def compare_methods(config: Config, selected: list[str], results_root: Path) -> dict[str, Any] | None:
    dft_names = [name for name in selected if isinstance(config.methods[name], VaspMethod) and (results_root / name / "summary.json").is_file()]
    dpa_names = [name for name in selected if isinstance(config.methods[name], DeepMDMethod) and (results_root / name / "summary.json").is_file()]
    if not dft_names or not dpa_names:
        return None
    dft_name = dft_names[0]
    dft_dir = results_root / dft_name
    dft_forces = np.asarray(np.load(dft_dir / "forces.npy"), dtype=float)
    with np.load(dft_dir / "band_data.npz") as data:
        dft_distances = np.asarray(data["distances"], dtype=float)
        dft_frequencies = np.asarray(data["frequencies"], dtype=float)
    dft_summary = load_json(dft_dir / "summary.json")
    out = results_root / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    plt = _plt()
    fig_band, ax_band = plt.subplots(figsize=(8.2, 5.0))
    for branch in range(dft_frequencies.shape[1]):
        ax_band.plot(dft_distances, dft_frequencies[:, branch], color="black", linewidth=1.8, alpha=0.9, label="DFT" if branch == 0 else None)

    fig_force, axes_force = plt.subplots(1, len(dpa_names), figsize=(5.2 * len(dpa_names), 4.6), squeeze=False)
    fig_frequency, axes_frequency = plt.subplots(1, len(dpa_names), figsize=(5.2 * len(dpa_names), 4.6), squeeze=False)
    colors = ["#197a9b", "#b44d4d", "#5c8540", "#815aa1"]
    for model_index, name in enumerate(dpa_names):
        model_dir = results_root / name
        forces = np.asarray(np.load(model_dir / "forces.npy"), dtype=float)
        with np.load(model_dir / "band_data.npz") as data:
            distances = np.asarray(data["distances"], dtype=float)
            frequencies = np.asarray(data["frequencies"], dtype=float)
        if forces.shape != dft_forces.shape:
            raise ValueError(f"{name} 与 DFT 力数组 shape 不一致: {forces.shape} vs {dft_forces.shape}")
        if frequencies.shape != dft_frequencies.shape or not np.allclose(distances, dft_distances):
            raise ValueError(f"{name} 与 DFT 声子路径不一致，无法直接比较")
        force_error = forces - dft_forces
        frequency_error = frequencies - dft_frequencies
        summary = load_json(model_dir / "summary.json")
        row = {
            "method": name,
            "reference": dft_name,
            "force_mae_mev_angstrom": float(np.mean(np.abs(force_error)) * 1000.0),
            "force_rmse_mev_angstrom": float(np.sqrt(np.mean(force_error**2)) * 1000.0),
            "frequency_mae_thz": float(np.mean(np.abs(frequency_error))),
            "frequency_rmse_thz": float(np.sqrt(np.mean(frequency_error**2))),
            "min_frequency_thz": summary["min_frequency_thz"],
            "dft_min_frequency_thz": dft_summary["min_frequency_thz"],
        }
        rows.append(row)
        color = colors[model_index % len(colors)]
        for branch in range(frequencies.shape[1]):
            ax_band.plot(distances, frequencies[:, branch], color=color, linewidth=1.0, alpha=0.75, label=name if branch == 0 else None)

        axis = axes_force[0, model_index]
        axis.scatter(dft_forces.reshape(-1), forces.reshape(-1), s=8, alpha=0.45, color=color)
        low = min(float(dft_forces.min()), float(forces.min()))
        high = max(float(dft_forces.max()), float(forces.max()))
        axis.plot([low, high], [low, high], color="black", linewidth=1)
        axis.set_title(name)
        axis.set_xlabel("DFT force (eV/Å)")
        axis.set_ylabel("DPA force (eV/Å)")
        axis.grid(alpha=0.2)

        axis = axes_frequency[0, model_index]
        flat_frequency_error = frequency_error.reshape(-1)
        if float(np.ptp(flat_frequency_error)) < 1.0e-12:
            center = float(np.mean(flat_frequency_error))
            bins = np.linspace(center - 1.0e-6, center + 1.0e-6, 21)
        else:
            bins = 60
        axis.hist(flat_frequency_error, bins=bins, color=color, alpha=0.8)
        axis.axvline(0.0, color="black", linewidth=1)
        axis.set_title(name)
        axis.set_xlabel("Frequency error (THz)")
        axis.set_ylabel("Count")
        axis.grid(alpha=0.2)

    ax_band.axhline(0.0, color="#777777", linewidth=0.8)
    ax_band.set_xlabel("q-path distance")
    ax_band.set_ylabel("Frequency (THz)")
    ax_band.set_title("Phonon dispersion comparison")
    ax_band.legend(frameon=False)
    fig_band.tight_layout()
    fig_band.savefig(out / "phonon_band_compare.png", dpi=220, bbox_inches="tight")
    plt.close(fig_band)
    fig_force.tight_layout()
    fig_force.savefig(out / "force_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig_force)
    fig_frequency.tight_layout()
    fig_frequency.savefig(out / "frequency_error.png", dpi=220, bbox_inches="tight")
    plt.close(fig_frequency)

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"reference": dft_name, "models": rows}
    atomic_write_json(out / "summary.json", result)
    return result

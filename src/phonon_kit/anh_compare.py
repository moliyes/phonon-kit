from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .util import atomic_write_json


def _table(path: Path) -> np.ndarray:
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8"))


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def compare_anh_methods(methods: list[str], results_root: Path, lifetime_temperature: float) -> dict | None:
    available = [name for name in methods if (results_root / name / "summary.json").is_file()]
    if len(available) < 2:
        return None
    outdir = results_root / "comparison"
    outdir.mkdir(parents=True, exist_ok=True)
    baseline = available[0]
    kappa_tables = {name: _table(results_root / name / "thermal_conductivity.csv") for name in available}
    temperatures = kappa_tables[baseline]["temperature_k"]
    for name, table in kappa_tables.items():
        if table.shape != kappa_tables[baseline].shape or not np.allclose(table["temperature_k"], temperatures):
            raise RuntimeError(f"模型 {name} 的热导率温度网格不一致")
    with (outdir / "thermal_conductivity_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = ["temperature_k"]
        for name in available:
            columns += [f"{name}_trace_over_3_w_mk", f"{name}_difference_from_{baseline}_w_mk"]
        writer.writerow(columns)
        for index, temperature in enumerate(temperatures):
            base_value = float(kappa_tables[baseline]["kappa_trace_over_3_w_mk"][index])
            row: list[float] = [float(temperature)]
            for name in available:
                value = float(kappa_tables[name]["kappa_trace_over_3_w_mk"][index])
                row += [value, value - base_value]
            writer.writerow(row)

    lifetime_name = f"lifetimes_{lifetime_temperature:g}K.csv"
    mode_tables = {name: _table(results_root / name / lifetime_name) for name in available}
    summaries = {name: json.loads((results_root / name / "summary.json").read_text(encoding="utf-8")) for name in available}
    base_modes = mode_tables[baseline]
    for name, table in mode_tables.items():
        if table.shape != base_modes.shape or not np.allclose(table["q_index"], base_modes["q_index"]) or not np.allclose(table["band"], base_modes["band"]):
            raise RuntimeError(f"模型 {name} 的 q 点或声子支网格不一致")
    with (outdir / "mode_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = ["q_index", "band", "q_x", "q_y", "q_z", "weight"]
        for name in available:
            columns += [f"{name}_frequency_thz", f"{name}_gamma_thz", f"{name}_lifetime_ps", f"{name}_diagnostic"]
        writer.writerow(columns)
        for index in range(len(base_modes)):
            row = [
                int(base_modes["q_index"][index]), int(base_modes["band"][index]),
                float(base_modes["q_x"][index]), float(base_modes["q_y"][index]),
                float(base_modes["q_z"][index]), int(base_modes["weight"][index]),
            ]
            for name in available:
                row += [
                    float(mode_tables[name]["frequency_thz"][index]),
                    float(mode_tables[name]["gamma_thz"][index]),
                    float(mode_tables[name]["lifetime_ps"][index]),
                    int(not summaries[name]["thermodynamic_stability"]),
                ]
            writer.writerow(row)

    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for name in available:
        ax.plot(temperatures, kappa_tables[name]["kappa_trace_over_3_w_mk"], marker="o", ms=3, label=name)
    ax.set(xlabel="Temperature (K)", ylabel="Lattice thermal conductivity, trace/3 (W m$^{-1}$ K$^{-1}$)")
    if any(not summaries[name]["thermodynamic_stability"] for name in available):
        ax.set_title("WARNING: at least one model has significant imaginary modes")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(outdir / "kappa_comparison.png", dpi=200); plt.close(fig)
    for quantity, ylabel, filename, log_scale in (
        ("gamma_thz", "Gamma (THz)", "gamma_comparison.png", False),
        ("lifetime_ps", "Lifetime (ps)", "lifetime_comparison.png", True),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for name in available:
            table = mode_tables[name]
            values = table[quantity]
            mask = np.isfinite(values) & (values > 0)
            ax.scatter(table["frequency_thz"][mask], values[mask], s=8, alpha=.55, label=name)
        ax.set(xlabel="Frequency (THz)", ylabel=ylabel)
        if log_scale: ax.set_yscale("log")
        if any(not summaries[name]["thermodynamic_stability"] for name in available):
            ax.set_title("WARNING: at least one model has significant imaginary modes")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(outdir / filename, dpi=200); plt.close(fig)
    summary = {
        "baseline": baseline, "methods": available,
        "wording": "comparison/change/difference only; no external reference",
        "lifetime_temperature_k": lifetime_temperature,
    }
    atomic_write_json(outdir / "summary.json", summary)
    return summary

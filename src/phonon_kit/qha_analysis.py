from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import _matplotlib, _write_force_files
from .qha_config import QHAConfig
from .qha_state import QHARunPaths, volume_id
from .qha_structure import composition_metadata
from .util import atomic_write_json, load_json


def analyze_qha_volume(
    config: QHAConfig,
    method_name: str,
    phase_name: str,
    ratio: float,
    phonopy_yaml: Path,
    static_energy_cell_ev: float,
    relaxation: dict[str, Any],
    resultdir: Path,
) -> dict[str, Any]:
    from phonopy import load

    force_path = resultdir / "forces.npy"
    if not force_path.is_file():
        raise FileNotFoundError(f"缺少 QHA forces.npy: {force_path}")
    phonon = load(str(phonopy_yaml), produce_fc=False)
    forces = np.asarray(np.load(force_path), dtype=float)
    expected = len(phonon.supercells_with_displacements)
    if forces.shape != (expected, len(phonon.supercell), 3) or not np.all(np.isfinite(forces)):
        raise ValueError(
            f"QHA forces.npy shape/数值无效: {forces.shape}, 预期 {(expected, len(phonon.supercell), 3)}"
        )
    phonon.forces = forces
    phonon.produce_force_constants()
    resultdir.mkdir(parents=True, exist_ok=True)
    _write_force_files(phonon, resultdir)
    phonon.save(filename=str(resultdir / "phonopy_params.yaml"), settings={"force_constants": True})

    settings = config.phases[phase_name].phonon
    phonon.run_mesh(list(settings.mesh))
    frequencies = np.asarray(phonon.mesh.frequencies, dtype=float)
    weights = np.asarray(phonon.mesh.weights, dtype=float)
    denominator = float(np.sum(weights) * frequencies.shape[1])
    imaginary = float(np.dot(weights, np.count_nonzero(frequencies < 0.0, axis=1)))
    significant = float(
        np.dot(weights, np.count_nonzero(frequencies < settings.significant_imaginary_thz, axis=1))
    )
    n_unit = len(phonon.unitcell)
    n_primitive = len(phonon.primitive)
    composition = composition_metadata(config.phases[phase_name].structure)
    atoms_per_formula = int(composition["atoms_per_formula_unit"])
    z_primitive = n_primitive / atoms_per_formula
    if abs(z_primitive - round(z_primitive)) > 1.0e-8 or z_primitive <= 0:
        raise ValueError(
            f"{phase_name}: primitive cell 原子数 {n_primitive} 不能归一到 {composition['formula_unit']}"
        )
    z_primitive = float(round(z_primitive))
    primitive_scale = n_primitive / n_unit
    energy_primitive = float(static_energy_cell_ev) * primitive_scale
    summary = {
        "method": method_name,
        "phase": phase_name,
        "ratio": float(ratio),
        "formula_unit": composition["formula_unit"],
        "n_atoms_unitcell": n_unit,
        "n_atoms_primitive": n_primitive,
        "formula_units_primitive": z_primitive,
        "primitive_energy_scale_from_unitcell": primitive_scale,
        "static_energy_ev_cell": float(static_energy_cell_ev),
        "static_energy_ev_primitive": energy_primitive,
        "static_energy_ev_formula_unit": energy_primitive / z_primitive,
        "unitcell_volume_angstrom3": float(phonon.unitcell.volume),
        "primitive_volume_angstrom3": float(phonon.primitive.volume),
        "volume_angstrom3_formula_unit": float(phonon.primitive.volume) / z_primitive,
        "n_displacements": expected,
        "mesh": list(settings.mesh),
        "min_frequency_thz": float(np.min(frequencies)),
        "imaginary_mode_fraction": imaginary / denominator,
        "significant_imaginary_mode_fraction": significant / denominator,
        "integrated_mode_fraction": 1.0 - imaginary / denominator,
        "thermodynamic_stability": significant == 0,
        "imaginary_policy": "exclude_below_0_thz",
        "significant_imaginary_threshold_thz": settings.significant_imaginary_thz,
        "relaxation": relaxation,
    }
    atomic_write_json(resultdir / "volume_summary.json", summary)
    return summary


def _write_thermal_yaml_and_csv(phonon, resultdir: Path) -> None:
    properties = phonon.thermal_properties
    properties.write_yaml(filename=str(resultdir / "thermal_properties.yaml"))
    with (resultdir / "thermal_properties.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["temperature_K", "free_energy_kJ_mol_primitive", "entropy_J_mol_primitive_K", "cv_J_mol_primitive_K"])
        for values in zip(
            properties.temperatures,
            properties.free_energy,
            properties.entropy,
            properties.heat_capacity,
        ):
            writer.writerow([float(value) for value in values])


def _plot_phase_qha(
    path: Path,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    gibbs_fu: np.ndarray,
    equilibrium_volumes_fu: np.ndarray,
    bulk_moduli: np.ndarray,
    warning: bool,
) -> None:
    plt = _matplotlib()
    pressure_index = 0
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axes[0].plot(temperatures, equilibrium_volumes_fu[pressure_index], color="#176b87")
    axes[0].set_ylabel("Equilibrium volume (A$^3$/formula unit)")
    axes[1].plot(temperatures, bulk_moduli[pressure_index], color="#527d3e")
    axes[1].set_ylabel("Bulk modulus (GPa)")
    relative = gibbs_fu[pressure_index] - gibbs_fu[pressure_index, 0]
    axes[2].plot(temperatures, relative, color="#a54a45")
    axes[2].set_ylabel("G - G(0 K) (eV/formula unit)")
    for axis in axes:
        axis.set_xlabel("Temperature (K)")
        axis.grid(alpha=0.25)
    title = f"QHA at {pressures[pressure_index]:g} GPa"
    if warning:
        title += " — WARNING: imaginary modes excluded"
    fig.suptitle(title, color="#a00000" if warning else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_static_eos(path: Path, volumes_fu: np.ndarray, energies_fu: np.ndarray, warning: bool) -> None:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    shifted = energies_fu - np.min(energies_fu)
    ax.scatter(volumes_fu, shifted, color="#176b87", zorder=3, label="sampled volumes")
    if len(volumes_fu) >= 3:
        x = np.linspace(float(np.min(volumes_fu)), float(np.max(volumes_fu)), 300)
        coefficients = np.polyfit(volumes_fu, shifted, 2)
        ax.plot(x, np.polyval(coefficients, x), color="#6d6d6d", alpha=0.8, label="quadratic guide")
    ax.set_xlabel("Volume (A$^3$/formula unit)")
    ax.set_ylabel("U - Umin (eV/formula unit)")
    ax.grid(alpha=0.25)
    ax.legend()
    title = "Static energy-volume samples"
    if warning:
        title += " — imaginary-mode warning"
    ax.set_title(title, color="#a00000" if warning else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze_phase_qha(config: QHAConfig, method_name: str, phase_name: str, paths: QHARunPaths) -> dict[str, Any]:
    from phonopy import load, run_qha
    from phonopy.physical_units import get_physical_units
    from phonopy.qha.core import BulkModulus

    phase = config.phases[phase_name]
    phase_result = paths.phase_results(method_name, phase_name)
    summaries: list[dict[str, Any]] = []
    phonopys = []
    resultdirs: list[Path] = []
    for index, ratio in enumerate(phase.volume_ratios):
        vid = volume_id(index, ratio)
        resultdir = phase_result / "volumes" / vid
        summaries.append(load_json(resultdir / "volume_summary.json"))
        phonopys.append(load(str(resultdir / "phonopy_params.yaml")))
        resultdirs.append(resultdir)
    order = np.argsort([item["primitive_volume_angstrom3"] for item in summaries])
    summaries = [summaries[int(index)] for index in order]
    phonopys = [phonopys[int(index)] for index in order]
    resultdirs = [resultdirs[int(index)] for index in order]
    basis_signatures = {
        (
            int(item["n_atoms_unitcell"]),
            int(item["n_atoms_primitive"]),
            float(item["formula_units_primitive"]),
            float(item["primitive_energy_scale_from_unitcell"]),
        )
        for item in summaries
    }
    if len(basis_signatures) != 1:
        details = sorted(basis_signatures)
        raise RuntimeError(
            f"{method_name}/{phase_name}: 各体积点的 primitive-cell 归一化不一致: {details}。"
            "这通常由逐点自动识别对称性造成；请使用修正版创建新运行。"
        )
    volumes = np.asarray([item["primitive_volume_angstrom3"] for item in summaries], dtype=float)
    energies = np.asarray([item["static_energy_ev_primitive"] for item in summaries], dtype=float)
    if len(np.unique(volumes)) < 5:
        raise RuntimeError(f"{method_name}/{phase_name}: QHA 至少需要 5 个唯一 primitive-cell 体积")
    input_temperatures = np.asarray(config.qha.temperature.values(include_extra=True), dtype=float)
    result = run_qha(
        phonopys,
        input_temperatures,
        energies,
        mesh=list(phase.phonon.mesh),
        eos=config.qha.eos,
    )
    temperatures = np.asarray(result.temperatures, dtype=float)
    requested = np.asarray(config.qha.temperature.values(), dtype=float)
    if len(temperatures) != len(requested) or not np.allclose(temperatures, requested):
        raise RuntimeError(
            f"{method_name}/{phase_name}: QHA 温度点不完整 ({len(temperatures)}/{len(requested)})，可能有 EOS 拟合失败"
        )
    for phonon, resultdir in zip(phonopys, resultdirs):
        _write_thermal_yaml_and_csv(phonon, resultdir)

    pressures = np.asarray(config.qha.pressure.values(), dtype=float)
    z_primitive = float(summaries[0]["formula_units_primitive"])
    n_pressure = len(pressures)
    n_temperature = len(temperatures)
    gibbs_primitive = np.full((n_pressure, n_temperature), np.nan)
    equilibrium_volumes = np.full_like(gibbs_primitive, np.nan)
    bulk_moduli = np.full_like(gibbs_primitive, np.nan)
    valid = np.zeros_like(gibbs_primitive, dtype=bool)
    near_edge = np.zeros_like(gibbs_primitive, dtype=bool)
    min_volume = float(np.min(result.volumes))
    max_volume = float(np.max(result.volumes))
    edge_width = float(np.median(np.diff(np.sort(result.volumes))))
    conversion = float(get_physical_units().EVAngstromToGPa)
    for pressure_index, pressure in enumerate(pressures):
        try:
            fitted = BulkModulus(
                result.volumes,
                result.helmholtz_volume,
                pressure=float(pressure),
                eos=config.qha.eos,
            )
        except (RuntimeError, TypeError, ValueError):
            # A failed pressure-row fit is represented as invalid data instead
            # of destroying otherwise useful QHA results at other pressures.
            continue
        energy = np.asarray(fitted.energy, dtype=float)
        volume = np.asarray(fitted.equilibrium_volume, dtype=float)
        bulk = np.asarray(fitted.bulk_modulus, dtype=float) * conversion
        gibbs_primitive[pressure_index] = energy
        equilibrium_volumes[pressure_index] = volume
        bulk_moduli[pressure_index] = bulk
        finite = np.isfinite(energy) & np.isfinite(volume) & np.isfinite(bulk)
        valid[pressure_index] = finite & (volume >= min_volume) & (volume <= max_volume) & (bulk > 0)
        near_edge[pressure_index] = valid[pressure_index] & (
            (volume <= min_volume + edge_width) | (volume >= max_volume - edge_width)
        )
    gibbs_fu = gibbs_primitive / z_primitive
    volumes_fu = equilibrium_volumes / z_primitive
    warning = any(not item["thermodynamic_stability"] for item in summaries)

    np.savez(
        phase_result / "qha_grid.npz",
        temperatures_k=temperatures,
        pressures_gpa=pressures,
        gibbs_ev_primitive=gibbs_primitive,
        gibbs_ev_formula_unit=gibbs_fu,
        equilibrium_volume_angstrom3_primitive=equilibrium_volumes,
        equilibrium_volume_angstrom3_formula_unit=volumes_fu,
        bulk_modulus_gpa=bulk_moduli,
        valid=valid,
        near_volume_edge=near_edge,
    )
    with (phase_result / "qha_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "pressure_GPa",
            "temperature_K",
            "gibbs_eV_formula_unit",
            "equilibrium_volume_A3_formula_unit",
            "bulk_modulus_GPa",
            "valid",
            "near_volume_edge",
            "imaginary_modes_excluded",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p_index, pressure in enumerate(pressures):
            for t_index, temperature in enumerate(temperatures):
                writer.writerow({
                    "pressure_GPa": float(pressure),
                    "temperature_K": float(temperature),
                    "gibbs_eV_formula_unit": float(gibbs_fu[p_index, t_index]),
                    "equilibrium_volume_A3_formula_unit": float(volumes_fu[p_index, t_index]),
                    "bulk_modulus_GPa": float(bulk_moduli[p_index, t_index]),
                    "valid": bool(valid[p_index, t_index]),
                    "near_volume_edge": bool(near_edge[p_index, t_index]),
                    "imaginary_modes_excluded": warning,
                })

    with (phase_result / "volume_points.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "ratio",
            "target_volume_A3_cell",
            "actual_volume_A3_cell",
            "volume_A3_primitive",
            "volume_A3_formula_unit",
            "static_energy_eV_cell",
            "static_energy_eV_primitive",
            "static_energy_eV_formula_unit",
            "fmax_eV_A",
            "max_deviatoric_stress_GPa",
            "min_frequency_THz",
            "imaginary_mode_fraction",
            "significant_imaginary_mode_fraction",
            "thermodynamic_stability",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            relax = item["relaxation"]
            writer.writerow({
                "ratio": item["ratio"],
                "target_volume_A3_cell": relax["target_volume_angstrom3"],
                "actual_volume_A3_cell": relax["volume_angstrom3"],
                "volume_A3_primitive": item["primitive_volume_angstrom3"],
                "volume_A3_formula_unit": item["volume_angstrom3_formula_unit"],
                "static_energy_eV_cell": item["static_energy_ev_cell"],
                "static_energy_eV_primitive": item["static_energy_ev_primitive"],
                "static_energy_eV_formula_unit": item["static_energy_ev_formula_unit"],
                "fmax_eV_A": relax["fmax_ev_angstrom"],
                "max_deviatoric_stress_GPa": relax["max_deviatoric_stress_gpa"],
                "min_frequency_THz": item["min_frequency_thz"],
                "imaginary_mode_fraction": item["imaginary_mode_fraction"],
                "significant_imaginary_mode_fraction": item["significant_imaginary_mode_fraction"],
                "thermodynamic_stability": item["thermodynamic_stability"],
            })
    with (phase_result / "e-v.dat").open("w", encoding="utf-8") as handle:
        handle.write("# primitive-cell volume (A^3)  static energy (eV)\n")
        for volume, energy in zip(volumes, energies):
            handle.write(f"{volume:20.10f} {energy:24.12f}\n")

    _plot_static_eos(
        phase_result / "static_eos.png",
        np.asarray([item["volume_angstrom3_formula_unit"] for item in summaries]),
        np.asarray([item["static_energy_ev_formula_unit"] for item in summaries]),
        warning,
    )
    _plot_phase_qha(
        phase_result / "qha_thermodynamics.png",
        temperatures,
        pressures,
        gibbs_fu,
        volumes_fu,
        bulk_moduli,
        warning,
    )
    summary = {
        "method": method_name,
        "phase": phase_name,
        "formula_unit": summaries[0]["formula_unit"],
        "formula_units_primitive": z_primitive,
        "eos": config.qha.eos,
        "n_volume_points": len(summaries),
        "volume_range_angstrom3_primitive": [min_volume, max_volume],
        "temperature_range_k": [float(temperatures[0]), float(temperatures[-1]), config.qha.temperature.step],
        "pressure_range_gpa": [float(pressures[0]), float(pressures[-1]), config.qha.pressure.step],
        "thermodynamic_stability": not warning,
        "imaginary_modes_excluded": warning,
        "invalid_grid_points": int(np.count_nonzero(~valid)),
        "near_volume_edge_grid_points": int(np.count_nonzero(near_edge)),
        "note": (
            "Significant imaginary modes were excluded from QHA thermodynamics; results are diagnostic."
            if warning
            else "No significant imaginary mode was found on sampled meshes."
        ),
    }
    atomic_write_json(phase_result / "summary.json", summary)
    return summary


def _phase_colors(config: QHAConfig) -> list[Any]:
    plt = _matplotlib()
    palette = list(plt.get_cmap("tab10").colors)
    colors: list[Any] = []
    for index, phase in enumerate(config.phases.values()):
        colors.append(phase.color or palette[index % len(palette)])
    return colors


def analyze_phase_diagram(config: QHAConfig, method_name: str, paths: QHARunPaths) -> dict[str, Any]:
    phase_names = list(config.phases)
    arrays = []
    summaries = []
    for phase in phase_names:
        resultdir = paths.phase_results(method_name, phase)
        arrays.append(np.load(resultdir / "qha_grid.npz"))
        summaries.append(load_json(resultdir / "summary.json"))
    temperatures = np.asarray(arrays[0]["temperatures_k"], dtype=float)
    pressures = np.asarray(arrays[0]["pressures_gpa"], dtype=float)
    gibbs = np.asarray([item["gibbs_ev_formula_unit"] for item in arrays], dtype=float)
    valid_by_phase = np.asarray([item["valid"] for item in arrays], dtype=bool)
    valid = np.all(valid_by_phase, axis=0)
    stable = np.argmin(gibbs, axis=0).astype(int)
    stable[~valid] = -1
    sorted_g = np.sort(gibbs, axis=0)
    gap = sorted_g[1] - sorted_g[0]
    warning = any(item["imaginary_modes_excluded"] for item in summaries)

    outdir = paths.method_results(method_name) / "phase_diagram"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        outdir / "phase_map.npz",
        temperatures_k=temperatures,
        pressures_gpa=pressures,
        stable_phase_index=stable,
        phase_names=np.asarray(phase_names, dtype=str),
        valid=valid,
        gibbs_ev_formula_unit=gibbs,
    )
    with (outdir / "phase_map.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["pressure_GPa", "temperature_K", "stable_phase", "gibbs_eV_formula_unit", "gap_to_second_eV_formula_unit", "valid", "imaginary_modes_excluded"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p_index, pressure in enumerate(pressures):
            for t_index, temperature in enumerate(temperatures):
                index = int(stable[p_index, t_index])
                writer.writerow({
                    "pressure_GPa": float(pressure),
                    "temperature_K": float(temperature),
                    "stable_phase": "" if index < 0 else phase_names[index],
                    "gibbs_eV_formula_unit": "" if index < 0 else float(gibbs[index, p_index, t_index]),
                    "gap_to_second_eV_formula_unit": "" if index < 0 else float(gap[p_index, t_index]),
                    "valid": bool(valid[p_index, t_index]),
                    "imaginary_modes_excluded": warning,
                })

    boundaries: list[dict[str, Any]] = []
    for t_index, temperature in enumerate(temperatures):
        for p_index in range(len(pressures) - 1):
            left = int(stable[p_index, t_index])
            right = int(stable[p_index + 1, t_index])
            if left < 0 or right < 0 or left == right:
                continue
            p0, p1 = pressures[p_index], pressures[p_index + 1]
            d0 = gibbs[left, p_index, t_index] - gibbs[right, p_index, t_index]
            d1 = gibbs[left, p_index + 1, t_index] - gibbs[right, p_index + 1, t_index]
            crossing = float(p0 - d0 * (p1 - p0) / (d1 - d0)) if d1 != d0 else float((p0 + p1) / 2)
            boundaries.append({
                "temperature_K": float(temperature),
                "pressure_GPa": crossing,
                "phase_low_pressure": phase_names[left],
                "phase_high_pressure": phase_names[right],
                "interpolation": "linear_delta_g",
            })
    with (outdir / "phase_boundaries.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["temperature_K", "pressure_GPa", "phase_low_pressure", "phase_high_pressure", "interpolation"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(boundaries)

    _plot_phase_map(config, outdir / "phase_diagram.png", temperatures, pressures, stable, warning)
    _plot_gibbs_slices(outdir, config, temperatures, pressures, gibbs, warning)
    summary = {
        "method": method_name,
        "phases": phase_names,
        "formula_unit": summaries[0]["formula_unit"],
        "valid_grid_points": int(np.count_nonzero(valid)),
        "invalid_grid_points": int(np.count_nonzero(~valid)),
        "boundary_points": len(boundaries),
        "thermodynamic_stability": not warning,
        "imaginary_modes_excluded": warning,
    }
    atomic_write_json(outdir / "summary.json", summary)
    for item in arrays:
        item.close()
    return summary


def _plot_phase_map(
    config: QHAConfig,
    path: Path,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    stable: np.ndarray,
    warning: bool,
) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    plt = _matplotlib()
    colors = _phase_colors(config)
    cmap = ListedColormap(colors).with_extremes(bad="#d5d5d5")
    masked = np.ma.masked_where(stable < 0, stable)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), len(colors))
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    mesh = ax.pcolormesh(temperatures, pressures, masked, cmap=cmap, norm=norm, shading="nearest")
    colorbar = fig.colorbar(mesh, ax=ax, ticks=np.arange(len(config.phases)))
    colorbar.ax.set_yticklabels([phase.label for phase in config.phases.values()])
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (GPa)")
    title = "QHA P-T phase diagram"
    if warning:
        title += " — WARNING: imaginary modes excluded"
    ax.set_title(title, color="#a00000" if warning else "black")
    ax.text(0.01, 0.01, "Gray: insufficient volume coverage / invalid EOS", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _nearest_indices(values: np.ndarray, targets: list[float]) -> list[int]:
    return sorted(set(int(np.argmin(np.abs(values - target))) for target in targets))


def _plot_gibbs_slices(
    outdir: Path,
    config: QHAConfig,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    gibbs: np.ndarray,
    warning: bool,
) -> None:
    plt = _matplotlib()
    colors = _phase_colors(config)
    temperature_indices = _nearest_indices(temperatures, [temperatures[0], 300, 600, temperatures[-1]])
    fig, axes = plt.subplots(1, len(temperature_indices), figsize=(4.2 * len(temperature_indices), 3.8), squeeze=False)
    for axis, t_index in zip(axes[0], temperature_indices):
        reference = np.nanmin(gibbs[:, :, t_index], axis=0)
        for phase_index, phase in enumerate(config.phases.values()):
            axis.plot(pressures, gibbs[phase_index, :, t_index] - reference, label=phase.label, color=colors[phase_index])
        axis.set_title(f"T={temperatures[t_index]:g} K")
        axis.set_xlabel("Pressure (GPa)")
        axis.set_ylabel("G - min(G) (eV/f.u.)")
        axis.grid(alpha=0.25)
    axes[0, -1].legend()
    if warning:
        fig.suptitle("WARNING: imaginary modes excluded", color="#a00000")
    fig.tight_layout()
    fig.savefig(outdir / "gibbs_vs_pressure.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    pressure_indices = _nearest_indices(pressures, [pressures[0], 5, 10, pressures[-1]])
    fig, axes = plt.subplots(1, len(pressure_indices), figsize=(4.2 * len(pressure_indices), 3.8), squeeze=False)
    for axis, p_index in zip(axes[0], pressure_indices):
        reference = np.nanmin(gibbs[:, p_index, :], axis=0)
        for phase_index, phase in enumerate(config.phases.values()):
            axis.plot(temperatures, gibbs[phase_index, p_index, :] - reference, label=phase.label, color=colors[phase_index])
        axis.set_title(f"P={pressures[p_index]:g} GPa")
        axis.set_xlabel("Temperature (K)")
        axis.set_ylabel("G - min(G) (eV/f.u.)")
        axis.grid(alpha=0.25)
    axes[0, -1].legend()
    if warning:
        fig.suptitle("WARNING: imaginary modes excluded", color="#a00000")
    fig.tight_layout()
    fig.savefig(outdir / "gibbs_vs_temperature.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def compare_qha_methods(config: QHAConfig, methods: list[str], paths: QHARunPaths) -> dict[str, Any] | None:
    available = [
        method for method in methods
        if (paths.method_results(method) / "phase_diagram" / "phase_map.npz").is_file()
    ]
    if len(available) < 2:
        return None
    reference = next((name for name in available if getattr(config.methods[name], "type", "") == "vasp"), available[0])
    reference_data = np.load(paths.method_results(reference) / "phase_diagram" / "phase_map.npz")
    ref_stable = np.asarray(reference_data["stable_phase_index"], dtype=int)
    temperatures = np.asarray(reference_data["temperatures_k"], dtype=float)
    pressures = np.asarray(reference_data["pressures_gpa"], dtype=float)
    outdir = paths.results / "comparison"
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    maps: list[tuple[str, np.ndarray]] = []
    for method in available:
        if method == reference:
            continue
        data = np.load(paths.method_results(method) / "phase_diagram" / "phase_map.npz")
        stable = np.asarray(data["stable_phase_index"], dtype=int)
        valid = (stable >= 0) & (ref_stable >= 0)
        agreement = valid & (stable == ref_stable)
        fraction = float(np.count_nonzero(agreement) / np.count_nonzero(valid)) if np.any(valid) else float("nan")
        rows.append({"reference_method": reference, "method": method, "overlap_grid_points": int(np.count_nonzero(valid)), "agreement_fraction": fraction})
        maps.append((method, np.where(valid, agreement.astype(float), np.nan)))
        data.close()
    with (outdir / "method_agreement.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["reference_method", "method", "overlap_grid_points", "agreement_fraction"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plt = _matplotlib()
    fig, axes = plt.subplots(1, len(maps), figsize=(5.4 * len(maps), 4.5), squeeze=False)
    for axis, (method, values) in zip(axes[0], maps):
        image = axis.pcolormesh(temperatures, pressures, values, vmin=0, vmax=1, cmap="RdYlGn", shading="nearest")
        axis.set_title(f"{method} vs {reference}")
        axis.set_xlabel("Temperature (K)")
        axis.set_ylabel("Pressure (GPa)")
        fig.colorbar(image, ax=axis, ticks=[0, 1], label="phase agreement")
    fig.tight_layout()
    fig.savefig(outdir / "method_agreement.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    reference_data.close()
    summary = {"reference_method": reference, "comparisons": rows}
    atomic_write_json(outdir / "summary.json", summary)
    return summary

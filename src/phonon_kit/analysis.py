from __future__ import annotations

import csv
import math
import os
from types import ModuleType
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .structure import phonopy_to_ase
from .util import atomic_write_json, load_json

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phonon-kit-matplotlib")


def _matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/phonon-kit-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_plot(plot_object: Any, path: Path) -> None:
    plt = _matplotlib()
    if isinstance(plot_object, ModuleType):
        figure = plot_object.gcf()
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
    elif hasattr(plot_object, "savefig"):
        plot_object.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(plot_object)
    else:
        plt.savefig(path, dpi=220, bbox_inches="tight")
        plt.close()


def _write_force_files(phonon, resultdir: Path) -> None:
    try:
        from phonopy.file_IO import write_FORCE_SETS

        write_FORCE_SETS(phonon.dataset, filename=str(resultdir / "FORCE_SETS"))
    except Exception as exc:
        raise RuntimeError(f"写入 FORCE_SETS 失败: {exc}") from exc
    try:
        from phonopy.file_IO import write_force_constants_to_hdf5

        write_force_constants_to_hdf5(
            phonon.force_constants,
            filename=str(resultdir / "force_constants.hdf5"),
        )
    except Exception as exc:
        raise RuntimeError(f"写入 force_constants.hdf5 失败: {exc}") from exc


def _extract_segment_arrays(values: Any) -> list[np.ndarray]:
    if isinstance(values, np.ndarray):
        if values.ndim >= 2:
            return [np.asarray(values)]
    return [np.asarray(item) for item in values]


def _band_path(config: Config, phonon, phonopy_yaml: Path, resultdir: Path) -> dict[str, Any]:
    source = phonopy_yaml.parent / "band_path.json"
    if source.is_file():
        data = load_json(source)
    else:
        from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath

        bands, labels, connections = get_band_qpoints_by_seekpath(
            phonon.primitive,
            npoints=config.phonon.band.points_per_segment,
            is_const_interval=True,
        )
        data = {
            "bands": [np.asarray(band, dtype=float).tolist() for band in bands],
            "labels": list(labels),
            "path_connections": [bool(value) for value in connections],
            "points_per_segment": config.phonon.band.points_per_segment,
        }
    atomic_write_json(resultdir / "band_path.json", data)
    return data


def _write_thermal_csv(path: Path, temperatures: np.ndarray, free_energy: np.ndarray, entropy: np.ndarray, heat_capacity: np.ndarray, zpe: float, n_atoms: int, ev_to_kj_mol: float) -> None:
    fields = [
        "temperature_K",
        "free_energy_kJ_mol_cell",
        "free_energy_eV_atom",
        "entropy_J_mol_cell_K",
        "entropy_eV_atom_K",
        "heat_capacity_J_mol_cell_K",
        "heat_capacity_eV_atom_K",
        "zero_point_energy_kJ_mol_cell",
        "zero_point_energy_eV_atom",
    ]
    zpe_ev_atom = zpe / ev_to_kj_mol / n_atoms
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, temperature in enumerate(temperatures):
            writer.writerow({
                "temperature_K": float(temperature),
                "free_energy_kJ_mol_cell": float(free_energy[index]),
                "free_energy_eV_atom": float(free_energy[index] / ev_to_kj_mol / n_atoms),
                "entropy_J_mol_cell_K": float(entropy[index]),
                "entropy_eV_atom_K": float(entropy[index] / (ev_to_kj_mol * 1000.0) / n_atoms),
                "heat_capacity_J_mol_cell_K": float(heat_capacity[index]),
                "heat_capacity_eV_atom_K": float(heat_capacity[index] / (ev_to_kj_mol * 1000.0) / n_atoms),
                "zero_point_energy_kJ_mol_cell": zpe,
                "zero_point_energy_eV_atom": zpe_ev_atom,
            })


def _plot_thermal(path: Path, temperatures: np.ndarray, free_energy: np.ndarray, entropy: np.ndarray, heat_capacity: np.ndarray, n_atoms: int, ev_to_kj_mol: float, warning: bool) -> None:
    plt = _matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axes[0].plot(temperatures, free_energy / ev_to_kj_mol / n_atoms, color="#176b87")
    axes[0].set_ylabel("Fvib (eV/atom)")
    axes[1].plot(temperatures, entropy / (ev_to_kj_mol * 1000.0) / n_atoms, color="#a54a45")
    axes[1].set_ylabel("Entropy (eV/atom/K)")
    axes[2].plot(temperatures, heat_capacity / (ev_to_kj_mol * 1000.0) / n_atoms, color="#527d3e")
    axes[2].set_ylabel("Cv (eV/atom/K)")
    for axis in axes:
        axis.set_xlabel("Temperature (K)")
        axis.grid(alpha=0.25)
    title = "Harmonic vibrational thermodynamics"
    if warning:
        title += " — WARNING: significant imaginary modes excluded"
    fig.suptitle(title, color="#a00000" if warning else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze_method(config: Config, method_name: str, phonopy_yaml: Path, resultdir: Path) -> dict[str, Any]:
    from phonopy import load
    from phonopy.physical_units import get_physical_units

    force_path = resultdir / "forces.npy"
    if not force_path.is_file():
        raise FileNotFoundError(f"缺少 forces.npy: {force_path}")
    forces = np.asarray(np.load(force_path), dtype=float)
    phonon = load(str(phonopy_yaml), produce_fc=False)
    expected = len(phonon.supercells_with_displacements)
    if forces.shape != (expected, len(phonon.supercell), 3):
        raise ValueError(f"forces.npy shape 错误: {forces.shape}, 预期 {(expected, len(phonon.supercell), 3)}")
    if not np.all(np.isfinite(forces)):
        raise ValueError("forces.npy 含 NaN 或无穷值")
    resultdir.mkdir(parents=True, exist_ok=True)
    phonon.forces = forces
    phonon.produce_force_constants()
    _write_force_files(phonon, resultdir)
    phonon.save(filename=str(resultdir / "phonopy_params.yaml"), settings={"force_constants": True})

    band_path = _band_path(config, phonon, phonopy_yaml, resultdir)
    bands = [np.asarray(item, dtype=float) for item in band_path["bands"]]
    phonon.run_band_structure(
        bands,
        path_connections=band_path["path_connections"],
        labels=band_path["labels"],
    )
    band = phonon.band_structure
    band.write_yaml(filename=str(resultdir / "band.yaml"))
    distance_segments = _extract_segment_arrays(band.distances)
    frequency_segments = _extract_segment_arrays(band.frequencies)
    distances = np.concatenate([item.reshape(-1) for item in distance_segments])
    frequencies = np.concatenate(frequency_segments, axis=0)
    np.savez(
        resultdir / "band_data.npz",
        distances=distances,
        frequencies=frequencies,
        labels=np.asarray(band_path["labels"], dtype=str),
    )
    _save_plot(phonon.plot_band_structure(), resultdir / "phonon_band.png")

    mesh = list(config.phonon.mesh)
    phonon.run_mesh(mesh)
    mesh_frequencies = np.asarray(phonon.mesh.frequencies, dtype=float)
    weights = np.asarray(phonon.mesh.weights, dtype=float)
    np.save(resultdir / "mesh_frequencies.npy", mesh_frequencies)
    min_frequency = float(np.min(mesh_frequencies))
    denominator = float(np.sum(weights) * mesh_frequencies.shape[1])
    imaginary_count = float(np.dot(weights, np.count_nonzero(mesh_frequencies < 0.0, axis=1)))
    significant_count = float(
        np.dot(
            weights,
            np.count_nonzero(mesh_frequencies < config.phonon.thermal.significant_imaginary_thz, axis=1),
        )
    )
    integrated_count = float(np.dot(weights, np.count_nonzero(mesh_frequencies >= 0.0, axis=1)))
    imaginary_fraction = imaginary_count / denominator
    significant_fraction = significant_count / denominator
    integrated_fraction = integrated_count / denominator
    stable = significant_count == 0

    phonon.run_total_dos()
    phonon.total_dos.write(filename=str(resultdir / "total_dos.dat"))
    _save_plot(phonon.plot_band_structure_and_dos(), resultdir / "phonon_band_dos.png")

    thermal = config.phonon.thermal
    phonon.run_thermal_properties(
        t_min=thermal.temperature_min_k,
        t_max=thermal.temperature_max_k,
        t_step=thermal.temperature_step_k,
        cutoff_frequency=0.0,
    )
    properties = phonon.thermal_properties
    properties.write_yaml(filename=str(resultdir / "thermal_properties.yaml"))
    temperatures = np.asarray(properties.temperatures, dtype=float)
    free_energy = np.asarray(properties.free_energy, dtype=float)
    entropy = np.asarray(properties.entropy, dtype=float)
    heat_capacity = np.asarray(properties.heat_capacity, dtype=float)
    zpe = float(properties.zero_point_energy)
    ev_to_kj_mol = float(get_physical_units().EvTokJmol)
    n_atoms = len(phonon.unitcell)
    _write_thermal_csv(
        resultdir / "thermal_properties.csv",
        temperatures,
        free_energy,
        entropy,
        heat_capacity,
        zpe,
        n_atoms,
        ev_to_kj_mol,
    )
    _plot_thermal(
        resultdir / "thermal_properties.png",
        temperatures,
        free_energy,
        entropy,
        heat_capacity,
        n_atoms,
        ev_to_kj_mol,
        not stable,
    )

    atoms = phonopy_to_ase(phonon.unitcell)
    unique_symbols, symbol_counts = np.unique(atoms.get_chemical_symbols(), return_counts=True)
    summary = {
        "method": method_name,
        "formula": atoms.get_chemical_formula(mode="hill"),
        "composition": {str(symbol): int(count) for symbol, count in zip(unique_symbols, symbol_counts)},
        "volume_angstrom3_cell": float(atoms.get_volume()),
        "n_atoms_unitcell": n_atoms,
        "n_atoms_supercell": len(phonon.supercell),
        "n_displacements": expected,
        "supercell": list(config.phonon.supercell),
        "displacement_angstrom": config.phonon.displacement_angstrom,
        "mesh": mesh,
        "min_frequency_thz": min_frequency,
        "imaginary_mode_fraction": imaginary_fraction,
        "significant_imaginary_mode_fraction": significant_fraction,
        "integrated_mode_fraction": integrated_fraction,
        "thermodynamic_stability": stable,
        "imaginary_policy": "exclude_below_0_thz",
        "significant_imaginary_threshold_thz": thermal.significant_imaginary_thz,
        "zero_point_energy_kj_mol_cell": zpe,
        "zero_point_energy_ev_atom": zpe / ev_to_kj_mol / n_atoms,
        "temperature_grid_k": [float(temperatures[0]), float(temperatures[-1]), thermal.temperature_step_k],
        "note": (
            "Significant imaginary modes exist. Thermal curves exclude all negative-frequency modes and are diagnostic only."
            if not stable
            else "No significant imaginary mode below the configured threshold was found."
        ),
    }
    atomic_write_json(resultdir / "summary.json", summary)
    return summary


def result_is_analyzed(resultdir: Path) -> bool:
    required = [
        "forces.npy",
        "phonopy_params.yaml",
        "band.yaml",
        "total_dos.dat",
        "thermal_properties.yaml",
        "thermal_properties.csv",
        "summary.json",
        "phonon_band.png",
        "phonon_band_dos.png",
        "thermal_properties.png",
    ]
    return all((resultdir / name).is_file() for name in required)

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .anh_config import AnhConfig
from .anh_structure import load_anh_displacements
from .errors import IncompleteResultsError
from .util import atomic_write_json


KAPPA_COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def lifetime_ps(gamma_thz: np.ndarray | float) -> np.ndarray:
    gamma = np.asarray(gamma_thz, dtype=float)
    result = np.full(gamma.shape, np.nan, dtype=float)
    positive = gamma > 0
    result[positive] = 1.0 / (4.0 * np.pi * gamma[positive])
    return result


def _load_force_set(directory: Path, cells: list[Any], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    forces, energies, missing = [], [], []
    for index, cell in enumerate(cells, start=1):
        path = directory / f"disp-{index:05d}.npz"
        if not path.is_file():
            missing.append(path.name); continue
        try:
            with np.load(path) as data:
                force = np.asarray(data["forces"], dtype=float)
                energy = float(data["energy"])
        except Exception as exc:
            raise IncompleteResultsError(f"损坏的 {prefix} checkpoint: {path}: {exc}") from exc
        if force.shape != (len(cell), 3) or not np.all(np.isfinite(force)) or not np.isfinite(energy):
            raise IncompleteResultsError(f"无效的 {prefix} checkpoint: {path}")
        forces.append(force); energies.append(energy)
    if missing:
        raise IncompleteResultsError(f"缺少 {len(missing)} 个 {prefix} checkpoint: {', '.join(missing[:5])}")
    return np.asarray(forces), np.asarray(energies)


def _residual(path: Path, n_atoms: int) -> np.ndarray:
    if not path.is_file():
        raise IncompleteResultsError(f"缺少背景力 checkpoint: {path}")
    with np.load(path) as data:
        force = np.asarray(data["forces"], dtype=float)
    if force.shape != (n_atoms, 3) or not np.all(np.isfinite(force)):
        raise IncompleteResultsError(f"背景力 checkpoint 无效: {path}")
    return force


def produce_force_constants(config: AnhConfig, yaml_path: Path, workdir: Path, resultdir: Path) -> dict[str, Any]:
    from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5

    ph3 = load_anh_displacements(config, yaml_path)
    checkpoints = workdir / "checkpoints"
    fc3_forces, fc3_energies = _load_force_set(checkpoints / "fc3", ph3.supercells_with_displacements, "fc3")
    separate_fc2 = ph3.phonon_supercell_matrix is not None
    fc2_forces = fc2_energies = None
    if separate_fc2:
        fc2_forces, fc2_energies = _load_force_set(
            checkpoints / "fc2", ph3.phonon_supercells_with_displacements, "fc2"
        )
    if config.anharmonic.subtract_residual_forces:
        fc3_forces = fc3_forces - _residual(checkpoints / "residual" / "fc3.npz", len(ph3.supercell))[None, :, :]
        if separate_fc2 and fc2_forces is not None:
            fc2_forces = fc2_forces - _residual(
                checkpoints / "residual" / "fc2.npz", len(ph3.phonon_supercell)
            )[None, :, :]

    resultdir.mkdir(parents=True, exist_ok=True)
    np.save(resultdir / "forces_fc3.npy", fc3_forces)
    np.save(resultdir / "energies_fc3.npy", fc3_energies)
    ph3.forces = fc3_forces
    if separate_fc2:
        assert fc2_forces is not None and fc2_energies is not None
        np.save(resultdir / "forces_fc2.npy", fc2_forces)
        np.save(resultdir / "energies_fc2.npy", fc2_energies)
        ph3.phonon_forces = fc2_forces

    ph3.produce_fc3(fc_calculator="traditional", is_compact_fc=True)
    ph3.symmetrize_fc3()
    if separate_fc2:
        ph3.produce_fc2(fc_calculator="traditional", is_compact_fc=True)
    ph3.symmetrize_fc2()
    if ph3.fc2 is None or ph3.fc3 is None:
        raise RuntimeError("Phono3py 未能生成 fc2/fc3")
    write_fc2_to_hdf5(
        ph3.fc2, filename=str(resultdir / "fc2.hdf5"), p2s_map=ph3.phonon_primitive.p2s_map,
        physical_unit="eV/angstrom^2",
    )
    write_fc3_to_hdf5(
        ph3.fc3, filename=str(resultdir / "fc3.hdf5"), p2s_map=ph3.primitive.p2s_map,
    )
    ph3.save(resultdir / "phono3py_params.yaml", settings={"force_sets": True, "force_constants": False})
    summary = {
        "fc3_shape": list(ph3.fc3.shape), "fc2_shape": list(ph3.fc2.shape),
        "fc3_displacements": len(fc3_forces),
        "fc2_displacements": 0 if fc2_forces is None else len(fc2_forces),
        "fc2_uses_separate_displacements": separate_fc2,
        "subtract_residual_forces": config.anharmonic.subtract_residual_forces,
        "nac_enabled": config.structure.born_file is not None,
    }
    atomic_write_json(resultdir / "force_constants_summary.json", summary)
    return summary


def _load_with_force_constants(config: AnhConfig, yaml_path: Path, resultdir: Path):
    from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5

    ph3 = load_anh_displacements(config, yaml_path)
    ph3.fc2 = read_fc2_from_hdf5(resultdir / "fc2.hdf5", p2s_map=ph3.phonon_primitive.p2s_map)
    fc3_data = read_fc3_from_hdf5(resultdir / "fc3.hdf5", p2s_map=ph3.primitive.p2s_map)
    ph3.fc3 = fc3_data["fc3"] if isinstance(fc3_data, dict) else fc3_data
    return ph3


def _valid_gamma_grid_points(scattering_dir: Path, mesh: tuple[int, int, int], temperatures: list[float], grid_points: np.ndarray) -> set[int]:
    from phono3py.file_IO import read_gamma_from_hdf5

    valid: set[int] = set()
    with _pushd(scattering_dir):
        for gp in grid_points:
            try:
                data, _ = read_gamma_from_hdf5(np.asarray(mesh, dtype="int64"), grid_point=int(gp))
                gamma = None if not data else np.asarray(data.get("gamma"))
                if gamma is not None and gamma.ndim == 2 and gamma.shape[0] == len(temperatures) and np.all(np.isfinite(gamma)):
                    valid.add(int(gp))
            except (OSError, KeyError, ValueError):
                pass
    return valid


def _write_kappa_csv(path: Path, temperatures: np.ndarray, kappa: np.ndarray, warning: bool) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["temperature_k", *(f"kappa_{x}_w_mk" for x in KAPPA_COMPONENTS), "kappa_trace_over_3_w_mk", "diagnostic_due_to_imaginary_modes"])
        for temperature, tensor in zip(temperatures, kappa, strict=True):
            writer.writerow([float(temperature), *map(float, tensor), float(np.mean(tensor[:3])), int(warning)])


def _write_lifetime_csv(path: Path, temperature: float, tc: Any, warning: bool) -> dict[str, Any]:
    temperatures = np.asarray(tc.temperatures, dtype=float)
    index = int(np.where(np.isclose(temperatures, temperature))[0][0])
    gamma = np.asarray(tc.gamma, dtype=float)[0, index]
    frequencies = np.asarray(tc.frequencies, dtype=float)
    qpoints = np.asarray(tc.qpoints, dtype=float)
    weights = np.asarray(tc.grid_weights, dtype=int)
    lifetimes = lifetime_ps(gamma)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["temperature_k", "q_index", "q_x", "q_y", "q_z", "weight", "band", "frequency_thz", "gamma_thz", "linewidth_thz", "lifetime_ps", "diagnostic_due_to_imaginary_modes"])
        for iq, (qpoint, weight) in enumerate(zip(qpoints, weights, strict=True)):
            for band in range(frequencies.shape[1]):
                writer.writerow([
                    temperature, iq, *map(float, qpoint), int(weight), band + 1,
                    float(frequencies[iq, band]), float(gamma[iq, band]), float(2 * gamma[iq, band]),
                    float(lifetimes[iq, band]) if np.isfinite(lifetimes[iq, band]) else "NaN", int(warning),
                ])
    return {
        "gamma_min_positive_thz": float(np.min(gamma[gamma > 0])) if np.any(gamma > 0) else None,
        "gamma_nonpositive_modes": int(np.count_nonzero(gamma <= 0)),
        "lifetime_finite_modes": int(np.count_nonzero(np.isfinite(lifetimes))),
    }


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_method(resultdir: Path) -> None:
    plt = _matplotlib()
    summary = json.loads((resultdir / "summary.json").read_text(encoding="utf-8"))
    warning = not summary["thermodynamic_stability"]
    data = np.atleast_1d(np.genfromtxt(resultdir / "thermal_conductivity.csv", delimiter=",", names=True, dtype=float, encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for key, label in (("kappa_xx_w_mk", "xx"), ("kappa_yy_w_mk", "yy"), ("kappa_zz_w_mk", "zz"), ("kappa_trace_over_3_w_mk", "trace/3")):
        ax.plot(data["temperature_k"], data[key], marker="o", ms=3, label=label)
    ax.set(xlabel="Temperature (K)", ylabel="Lattice thermal conductivity (W m$^{-1}$ K$^{-1}$)")
    ax.legend(); ax.grid(alpha=.25)
    if warning: ax.set_title("WARNING: significant imaginary modes; diagnostic result")
    fig.tight_layout(); fig.savefig(resultdir / "kappa.png", dpi=200); plt.close(fig)

    lifetime_path = resultdir / f"lifetimes_{summary['lifetime_temperature_k']:g}K.csv"
    modes = np.atleast_1d(np.genfromtxt(lifetime_path, delimiter=",", names=True, dtype=float, encoding="utf-8"))
    finite_gamma = np.isfinite(modes["gamma_thz"]) & (modes["gamma_thz"] > 0)
    fig, ax = plt.subplots(figsize=(7.2, 4.8)); ax.scatter(modes["frequency_thz"][finite_gamma], modes["gamma_thz"][finite_gamma], s=9, alpha=.65)
    ax.set(xlabel="Frequency (THz)", ylabel="Gamma (THz)"); ax.grid(alpha=.25)
    if warning: ax.set_title("WARNING: significant imaginary modes; diagnostic result")
    fig.tight_layout(); fig.savefig(resultdir / "gamma.png", dpi=200); plt.close(fig)
    finite_tau = np.isfinite(modes["lifetime_ps"]) & (modes["lifetime_ps"] > 0)
    fig, ax = plt.subplots(figsize=(7.2, 4.8)); ax.scatter(modes["frequency_thz"][finite_tau], modes["lifetime_ps"][finite_tau], s=9, alpha=.65)
    ax.set(xlabel="Frequency (THz)", ylabel="Lifetime (ps)"); ax.set_yscale("log"); ax.grid(alpha=.25)
    if warning: ax.set_title("WARNING: significant imaginary modes; diagnostic result")
    fig.tight_layout(); fig.savefig(resultdir / "lifetime.png", dpi=200); plt.close(fig)


def analyze_anh_method(config: AnhConfig, method_name: str, yaml_path: Path, workdir: Path, resultdir: Path) -> dict[str, Any]:
    from phonopy.phonon.grid import get_ir_grid_points

    ph3 = _load_with_force_constants(config, yaml_path, resultdir)
    ph3.mesh_numbers = config.anharmonic.mesh
    ph3.init_phph_interaction()
    all_frequencies = np.asarray(ph3.get_phonon_data()[0], dtype=float)
    min_frequency = float(np.min(all_frequencies))
    significant_count = int(np.count_nonzero(all_frequencies < config.anharmonic.significant_imaginary_thz))
    stability = significant_count == 0
    if not stability and not config.anharmonic.continue_on_imaginary:
        raise RuntimeError(
            f"{method_name} 存在明显虚频: min={min_frequency:.6f} THz；continue_on_imaginary=false"
        )

    # ``get_ir_grid_points`` returns regular-grid (GRG) indices, whereas the
    # conductivity API and gamma HDF5 filenames use BZ-grid (BZG) indices.
    # They coincide for many low indices, which made the mismatch easy to
    # miss, but diverge whenever the BZ grid contains duplicated boundary
    # points (e.g. Si on an 11x11x11 mesh).
    ir_grid_points, grid_weights, _ = get_ir_grid_points(ph3.grid)
    grid_points = np.asarray(ph3.grid.grg2bzg[ir_grid_points], dtype="int64")
    scattering_dir = workdir / "scattering"
    temperatures = config.anharmonic.temperatures
    valid = _valid_gamma_grid_points(scattering_dir, config.anharmonic.mesh, temperatures, grid_points)
    missing = [int(gp) for gp in grid_points if int(gp) not in valid]
    atomic_write_json(scattering_dir / "manifest.json", {
        "mesh": list(config.anharmonic.mesh), "grid_points": list(map(int, grid_points)),
        "grid_weights": list(map(int, grid_weights)), "temperatures_k": temperatures,
        "completed_before_run": len(valid), "missing_before_run": len(missing),
    })
    if missing:
        with _pushd(scattering_dir):
            ph3.run_thermal_conductivity(
                is_LBTE=False, temperatures=temperatures, grid_points=missing,
                write_gamma=True, is_isotope=False, boundary_mfp=None, log_level=1,
            )
    valid = _valid_gamma_grid_points(scattering_dir, config.anharmonic.mesh, temperatures, grid_points)
    if len(valid) != len(grid_points):
        raise IncompleteResultsError(f"三声子散射缺少 {len(grid_points) - len(valid)} 个 q 点结果")
    with _pushd(scattering_dir):
        ph3.run_thermal_conductivity(
            is_LBTE=False, temperatures=temperatures, read_gamma=True,
            write_kappa=True, is_isotope=False, boundary_mfp=None, log_level=1,
        )
    tc = ph3.thermal_conductivity
    if tc is None or tc.kappa is None or tc.gamma is None:
        raise RuntimeError("Phono3py RTA 没有返回 kappa/gamma")
    kappa = np.asarray(tc.kappa, dtype=float)[0]
    tc_temperatures = np.asarray(tc.temperatures, dtype=float)
    _write_kappa_csv(resultdir / "thermal_conductivity.csv", tc_temperatures, kappa, not stability)
    lifetime_name = f"lifetimes_{config.anharmonic.lifetime_temperature_k:g}K.csv"
    lifetime_summary = _write_lifetime_csv(
        resultdir / lifetime_name, config.anharmonic.lifetime_temperature_k, tc, not stability
    )
    full_kappa = [path for path in scattering_dir.glob("kappa-m*.hdf5") if not re.search(r"-g\d+", path.stem)]
    if full_kappa:
        shutil.copy2(max(full_kappa, key=lambda path: path.stat().st_mtime), resultdir / "kappa.hdf5")
    summary = {
        "method": method_name, "mesh": list(config.anharmonic.mesh),
        "temperatures_k": list(map(float, tc_temperatures)),
        "lifetime_temperature_k": config.anharmonic.lifetime_temperature_k,
        "kappa_tensor_w_mk": kappa.tolist(),
        "kappa_trace_over_3_w_mk": np.mean(kappa[:, :3], axis=1).tolist(),
        "min_frequency_thz": min_frequency,
        "significant_imaginary_threshold_thz": config.anharmonic.significant_imaginary_thz,
        "significant_imaginary_modes": significant_count,
        "significant_imaginary_mode_fraction": significant_count / float(all_frequencies.size),
        "thermodynamic_stability": stability,
        "continue_on_imaginary": config.anharmonic.continue_on_imaginary,
        "diagnostic_due_to_imaginary_modes": not stability,
        "nac_enabled": config.structure.born_file is not None,
        "subtract_residual_forces": config.anharmonic.subtract_residual_forces,
        "rta_scattering": "intrinsic_three_phonon_only",
        "integration": "linear_tetrahedron",
        "irreducible_grid_points": len(grid_points),
        **lifetime_summary,
    }
    atomic_write_json(resultdir / "summary.json", summary)
    plot_method(resultdir)
    return summary


def result_is_complete(resultdir: Path, lifetime_temperature: float) -> bool:
    required = [
        "fc2.hdf5", "fc3.hdf5", "phono3py_params.yaml", "kappa.hdf5", "thermal_conductivity.csv",
        f"lifetimes_{lifetime_temperature:g}K.csv", "kappa.png", "gamma.png", "lifetime.png", "summary.json",
    ]
    return all((resultdir / name).is_file() for name in required)


def force_constants_are_valid(resultdir: Path) -> bool:
    from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5

    if not (resultdir / "phono3py_params.yaml").is_file():
        return False
    try:
        fc2 = np.asarray(read_fc2_from_hdf5(resultdir / "fc2.hdf5"), dtype=float)
        raw_fc3 = read_fc3_from_hdf5(resultdir / "fc3.hdf5")
        fc3 = np.asarray(raw_fc3["fc3"] if isinstance(raw_fc3, dict) else raw_fc3, dtype=float)
    except (OSError, KeyError, ValueError):
        return False
    return fc2.ndim == 4 and fc3.ndim == 6 and np.all(np.isfinite(fc2)) and np.all(np.isfinite(fc3))

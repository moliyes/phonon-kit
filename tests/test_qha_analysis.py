from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from phonon_kit.qha_analysis import analyze_phase_diagram, analyze_phase_qha, analyze_qha_volume
from phonon_kit.qha_config import load_qha_config
from phonon_kit.qha_initializer import init_qha_case
from phonon_kit.qha_state import choose_qha_run, volume_id
from phonon_kit.qha_structure import generate_qha_displacements, scale_structure
from phonon_kit.structure import read_structure
from phonon_kit.util import atomic_write_json


def test_qha_volume_normalizes_energy_to_formula_unit(tmp_path: Path):
    case = init_qha_case(tmp_path / "case", ["alpha", "beta"])
    config = load_qha_config(case / "qha.yaml")
    phase = config.phases["alpha"]
    phonon_dir = tmp_path / "phonon"
    manifest = generate_qha_displacements(phase.structure, phase.phonon, phonon_dir)
    resultdir = tmp_path / "result"
    resultdir.mkdir()
    np.save(
        resultdir / "forces.npy",
        np.zeros((manifest["n_displacements"], manifest["n_atoms_supercell"], 3)),
    )
    relaxation = {
        "target_volume_angstrom3": manifest["unitcell_volume_angstrom3"],
        "volume_angstrom3": manifest["unitcell_volume_angstrom3"],
        "fmax_ev_angstrom": 0.0,
        "max_deviatoric_stress_gpa": 0.0,
    }
    summary = analyze_qha_volume(
        config,
        "dpa4",
        "alpha",
        1.0,
        phonon_dir / "phonopy_disp.yaml",
        -12.0,
        relaxation,
        resultdir,
    )
    assert summary["formula_units_primitive"] == 2
    assert summary["static_energy_ev_formula_unit"] == -6.0
    assert (resultdir / "phonopy_params.yaml").is_file()


def test_multiphase_map_and_boundary_interpolation(tmp_path: Path):
    case = init_qha_case(tmp_path / "case", ["alpha", "beta"])
    config = load_qha_config(case / "qha.yaml")
    paths, _, _ = choose_qha_run(config, ["dpa4"])
    pressures = np.array([0.0, 1.0, 2.0])
    temperatures = np.array([0.0, 100.0])
    alpha = np.zeros((3, 2))
    beta = 1.0 - pressures[:, None] + np.zeros((3, 2))
    for phase, values in (("alpha", alpha), ("beta", beta)):
        resultdir = paths.phase_results("dpa4", phase)
        resultdir.mkdir(parents=True)
        np.savez(
            resultdir / "qha_grid.npz",
            temperatures_k=temperatures,
            pressures_gpa=pressures,
            gibbs_ev_formula_unit=values,
            valid=np.ones_like(values, dtype=bool),
        )
        atomic_write_json(
            resultdir / "summary.json",
            {"formula_unit": "SiO2", "imaginary_modes_excluded": False},
        )
    summary = analyze_phase_diagram(config, "dpa4", paths)
    assert summary["boundary_points"] == 2
    boundary_path = paths.method_results("dpa4") / "phase_diagram" / "phase_boundaries.csv"
    with boundary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(np.isclose(float(row["pressure_GPa"]), 1.0) for row in rows)
    assert (paths.method_results("dpa4") / "phase_diagram" / "phase_diagram.png").is_file()


def test_phase_qha_real_phonopy_api(tmp_path: Path):
    case = init_qha_case(tmp_path / "case", ["alpha", "beta"])
    config_path = case / "qha.yaml"
    data = yaml.safe_load(config_path.read_text())
    data["volume_grid"]["ratios"] = [0.9, 0.95, 1.0, 1.05, 1.1]
    for phase in data["phases"].values():
        phase["phonon"]["supercell"] = [1, 1, 1]
    data["phonon"]["mesh"] = [2, 2, 2]
    data["qha"]["temperature"] = {"min_k": 0, "max_k": 20, "step_k": 10}
    data["qha"]["pressure"] = {"min_gpa": 0, "max_gpa": 1, "step_gpa": 1}
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    config = load_qha_config(config_path)
    paths, _, _ = choose_qha_run(config, ["dpa4"])
    phase = config.phases["alpha"]
    reference_volume = read_structure(phase.structure).get_volume()
    for index, ratio in enumerate(phase.volume_ratios):
        vid = volume_id(index, ratio)
        workdir = paths.volume_work("dpa4", "alpha", vid)
        structure = workdir / "POSCAR-relaxed"
        scaling = scale_structure(phase.structure, structure, ratio)
        phonon_dir = workdir / "phonon"
        manifest = generate_qha_displacements(structure, phase.phonon, phonon_dir)
        resultdir = paths.phase_results("dpa4", "alpha") / "volumes" / vid
        resultdir.mkdir(parents=True)
        np.save(resultdir / "forces.npy", np.zeros((manifest["n_displacements"], manifest["n_atoms_supercell"], 3)))
        actual_volume = scaling["actual_volume_angstrom3"]
        analyze_qha_volume(
            config,
            "dpa4",
            "alpha",
            ratio,
            phonon_dir / "phonopy_disp.yaml",
            -12.0 + 0.01 * (actual_volume - reference_volume) ** 2,
            {
                "target_volume_angstrom3": scaling["target_volume_angstrom3"],
                "volume_angstrom3": actual_volume,
                "fmax_ev_angstrom": 0.0,
                "max_deviatoric_stress_gpa": 0.0,
            },
            resultdir,
        )
    summary = analyze_phase_qha(config, "dpa4", "alpha", paths)
    assert summary["n_volume_points"] == 5
    grid = np.load(paths.phase_results("dpa4", "alpha") / "qha_grid.npz")
    assert grid["gibbs_ev_formula_unit"].shape == (2, 3)
    assert np.any(grid["valid"])

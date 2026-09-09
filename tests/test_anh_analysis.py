from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write

from phonon_kit.anh_analysis import (
    _load_with_force_constants,
    analyze_anh_method,
    force_constants_are_valid,
    lifetime_ps,
    produce_force_constants,
    result_is_complete,
)
from phonon_kit.anh_compare import compare_anh_methods
from phonon_kit.anh_config import load_anh_config
from phonon_kit.anh_structure import generate_anh_displacements, load_anh_displacements
from phonon_kit.anh_worker import run as run_anh_worker
from phonon_kit.deepmd_worker import _atomic_npz
from phonon_kit.structure import phonopy_to_ase


def _config(
    root: Path,
    *,
    subtract: bool = False,
    separate_fc2: bool = False,
    fc2_distance: float | None = None,
    fc2_is_diagonal: bool = False,
) -> Path:
    write(root / "POSCAR", bulk("Al", "fcc", a=4.05), format="vasp", direct=True, vasp5=True)
    (root / "model.pth").write_bytes(b"dummy")
    path = root / "anh.yaml"
    path.write_text(f"""schema_version: 1
project:
  name: al
  runs_dir: runs
structure:
  file: POSCAR
anharmonic:
  supercell: [2, 2, 2]
  fc2_supercell: {'[2, 1, 1]' if separate_fc2 else 'null'}
  displacement_angstrom: 0.03
  fc2_displacement_angstrom: {fc2_distance if fc2_distance is not None else 'null'}
  fc2_is_diagonal: {str(fc2_is_diagonal).lower()}
  primitive: auto
  symmetry_tolerance: 1.0e-5
  subtract_residual_forces: {str(subtract).lower()}
  mesh: [3, 3, 3]
  temperature_min_k: 300
  temperature_max_k: 300
  temperature_step_k: 100
  lifetime_temperature_k: 300
  significant_imaginary_thz: -0.1
  continue_on_imaginary: true
methods:
  emt:
    type: deepmd
    model: model.pth
    device: cpu
""", encoding="utf-8")
    return path


def _write_emt_checkpoints(config, yaml_path: Path, workdir: Path) -> None:
    ph3 = load_anh_displacements(config, yaml_path)
    root = workdir / "checkpoints"
    calc = EMT()
    for index, cell in enumerate(ph3.supercells_with_displacements, start=1):
        atoms = phonopy_to_ase(cell); atoms.calc = calc
        _atomic_npz(root / "fc3" / f"disp-{index:05d}.npz", forces=atoms.get_forces(), energy=np.asarray(atoms.get_potential_energy()))
    if ph3.phonon_supercell_matrix is not None:
        for index, cell in enumerate(ph3.phonon_supercells_with_displacements, start=1):
            atoms = phonopy_to_ase(cell); atoms.calc = calc
            _atomic_npz(root / "fc2" / f"disp-{index:05d}.npz", forces=atoms.get_forces(), energy=np.asarray(atoms.get_potential_energy()))
    if config.anharmonic.subtract_residual_forces:
        atoms = phonopy_to_ase(ph3.supercell); atoms.calc = calc
        _atomic_npz(root / "residual" / "fc3.npz", forces=atoms.get_forces(), energy=np.asarray(atoms.get_potential_energy()))
        if ph3.phonon_supercell_matrix is not None:
            atoms = phonopy_to_ase(ph3.phonon_supercell); atoms.calc = calc
            _atomic_npz(root / "residual" / "fc2.npz", forces=atoms.get_forces(), energy=np.asarray(atoms.get_potential_energy()))


def test_lifetime_formula_and_nonpositive_nan() -> None:
    result = lifetime_ps(np.array([1.0, 0.0, -1.0]))
    np.testing.assert_allclose(result[0], 1 / (4 * np.pi))
    assert np.isnan(result[1:]).all()


def test_small_analytic_fc3_rta_and_resume(tmp_path: Path) -> None:
    config = load_anh_config(_config(tmp_path))
    disp = tmp_path / "disp"
    generate_anh_displacements(config, config.structure.file, disp)
    yaml_path = disp / "phono3py_disp.yaml"
    work = tmp_path / "work"
    results = tmp_path / "results" / "emt"
    _write_emt_checkpoints(config, yaml_path, work)
    fc_summary = produce_force_constants(config, yaml_path, work, results)
    assert fc_summary["fc3_shape"] and fc_summary["fc2_shape"]
    assert force_constants_are_valid(results)
    summary = analyze_anh_method(config, "emt", yaml_path, work, results)
    assert summary["rta_scattering"] == "intrinsic_three_phonon_only"
    assert result_is_complete(results, 300)
    ph3 = _load_with_force_constants(config, yaml_path, results)
    ph3.mesh_numbers = config.anharmonic.mesh
    ph3.init_phph_interaction()
    from phonopy.phonon.grid import get_ir_grid_points

    ir_grg, _, _ = get_ir_grid_points(ph3.grid)
    expected_bzg = np.asarray(ph3.grid.grg2bzg[ir_grg], dtype=int).tolist()
    manifest = json.loads((work / "scattering" / "manifest.json").read_text())
    assert manifest["grid_points"] == expected_bzg
    gamma_files = list((work / "scattering").glob("kappa-m*-g*.hdf5"))
    mtimes = {path.name: path.stat().st_mtime_ns for path in gamma_files}
    analyze_anh_method(config, "emt", yaml_path, work, results)
    assert mtimes == {path.name: path.stat().st_mtime_ns for path in gamma_files}

    second = tmp_path / "results" / "emt-copy"
    shutil.copytree(results, second)
    comparison = compare_anh_methods(["emt", "emt-copy"], tmp_path / "results", 300)
    assert comparison is not None and comparison["baseline"] == "emt"
    assert (tmp_path / "results" / "comparison" / "kappa_comparison.png").is_file()


def test_optional_residual_force_subtraction_is_recorded(tmp_path: Path) -> None:
    config = load_anh_config(_config(tmp_path, subtract=True))
    disp = tmp_path / "disp"; generate_anh_displacements(config, config.structure.file, disp)
    work = tmp_path / "work"; results = tmp_path / "results"
    _write_emt_checkpoints(config, disp / "phono3py_disp.yaml", work)
    summary = produce_force_constants(config, disp / "phono3py_disp.yaml", work, results)
    assert summary["subtract_residual_forces"] is True
    assert np.load(results / "forces_fc3.npy").ndim == 3


def test_separate_fc2_displacements_produce_fc2(tmp_path: Path) -> None:
    config = load_anh_config(
        _config(
            tmp_path,
            separate_fc2=True,
            fc2_distance=0.01,
            fc2_is_diagonal=True,
        )
    )
    disp = tmp_path / "disp"; generate_anh_displacements(config, config.structure.file, disp)
    ph3 = load_anh_displacements(config, disp / "phono3py_disp.yaml")
    assert all(
        np.linalg.norm(item["displacement"]) == pytest.approx(0.03)
        for item in ph3.dataset["first_atoms"]
    )
    assert all(
        np.linalg.norm(item["displacement"]) == pytest.approx(0.01)
        for item in ph3.phonon_dataset["first_atoms"]
    )
    manifest = json.loads((disp / "manifest.json").read_text())
    assert manifest["fc3_displacement_angstrom"] == 0.03
    assert manifest["fc2_displacement_angstrom"] == 0.01
    assert manifest["fc2_is_diagonal"] is True
    work = tmp_path / "work"; results = tmp_path / "results"
    _write_emt_checkpoints(config, disp / "phono3py_disp.yaml", work)
    summary = produce_force_constants(config, disp / "phono3py_disp.yaml", work, results)
    assert summary["fc2_uses_separate_displacements"] is True
    assert summary["fc2_displacements"] > 0
    assert (results / "forces_fc2.npy").is_file()


def test_anh_worker_skips_existing_checkpoints(tmp_path: Path, monkeypatch) -> None:
    config = load_anh_config(_config(tmp_path))
    disp = tmp_path / "disp"; generate_anh_displacements(config, config.structure.file, disp)
    monkeypatch.setattr("phonon_kit.anh_worker.make_calculator", lambda model, head: EMT())
    payload = {
        "phono3py_yaml": str(disp / "phono3py_disp.yaml"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "model": "unused", "head": None, "subtract_residual_forces": False,
    }
    first = run_anh_worker(payload)
    files = sorted((tmp_path / "checkpoints" / "fc3").glob("disp-*.npz"))
    mtimes = {path.name: path.stat().st_mtime_ns for path in files}
    second = run_anh_worker(payload)
    assert first == second
    assert mtimes == {path.name: path.stat().st_mtime_ns for path in files}

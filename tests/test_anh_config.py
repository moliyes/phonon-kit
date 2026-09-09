from __future__ import annotations

from pathlib import Path

import pytest

from phonon_kit.anh_config import load_anh_config
from phonon_kit.anh_initializer import init_anh_case
from phonon_kit.anh_snapshot import create_anh_snapshot
from phonon_kit.anh_state import AnhRunPaths, choose_anh_run
from phonon_kit.anh_structure import build_phono3py, displacement_plan
from phonon_kit.errors import ConfigError


POSCAR = """Al
1.0
4.05 0 0
0 4.05 0
0 0 4.05
Al
1
Direct
0 0 0
"""


def write_anh_config(root: Path, *, extra_anh: str = "", extra_structure: str = "", model: str = "model.pth") -> Path:
    (root / "POSCAR").write_text(POSCAR, encoding="utf-8")
    (root / model).write_bytes(b"model")
    path = root / "anh.yaml"
    path.write_text(f"""schema_version: 1
project:
  name: test
  runs_dir: runs
structure:
  file: POSCAR
{extra_structure}anharmonic:
  supercell: [1, 1, 1]
  mesh: [3, 3, 3]
  temperature_min_k: 300
  temperature_max_k: 300
  temperature_step_k: 100
  lifetime_temperature_k: 300
{extra_anh}methods:
  dp:
    type: deepmd
    model: {model}
    device: cpu
""", encoding="utf-8")
    return path


def test_anh_init_and_defaults(tmp_path: Path) -> None:
    target = init_anh_case(tmp_path / "case")
    config = load_anh_config(target / "anh.yaml")
    assert config.anharmonic.supercell == (2, 2, 2)
    assert config.anharmonic.fc2_supercell is None
    assert config.anharmonic.fc2_displacement_angstrom is None
    assert config.anharmonic.resolved_fc2_displacement_angstrom == 0.03
    assert config.anharmonic.fc2_is_diagonal is False
    assert config.anharmonic.subtract_residual_forces is False


def test_anh_strict_and_lifetime_grid(tmp_path: Path) -> None:
    path = write_anh_config(tmp_path, extra_anh="  mystery: 1\n")
    with pytest.raises(ConfigError, match="未知字段"):
        load_anh_config(path)
    path.write_text(path.read_text().replace("  mystery: 1\n", "  lifetime_temperature_k: 250\n"), encoding="utf-8")
    with pytest.raises(ConfigError, match="必须落在温度网格"):
        load_anh_config(path)


def test_anh_pt2_rejects_head(tmp_path: Path) -> None:
    path = write_anh_config(tmp_path, model="model.pt2")
    path.write_text(path.read_text().replace("    device: cpu", "    device: cpu\n    head: Omat24"), encoding="utf-8")
    with pytest.raises(ConfigError, match="不能再设置 head"):
        load_anh_config(path)


def test_anh_plan_counts_separate_fc2_and_residual(tmp_path: Path) -> None:
    path = write_anh_config(
        tmp_path,
        extra_anh=(
            "  fc2_supercell: [2, 1, 1]\n"
            "  fc2_displacement_angstrom: 0.01\n"
            "  fc2_is_diagonal: true\n"
            "  subtract_residual_forces: true\n"
        ),
    )
    config = load_anh_config(path)
    plan = displacement_plan(config)
    assert config.anharmonic.displacement_angstrom == 0.03
    assert config.anharmonic.fc2_displacement_angstrom == 0.01
    assert config.anharmonic.fc2_is_diagonal is True
    assert plan["fc2_uses_separate_displacements"] is True
    assert plan["fc3_displacements"] > 0
    assert plan["fc2_displacements"] > 0
    assert plan["residual_evaluations_per_model"] == 2
    assert plan["fc3_displacement_angstrom"] == 0.03
    assert plan["fc2_displacement_angstrom"] == 0.01
    assert plan["fc2_is_diagonal"] is True
    assert plan["total_force_evaluations"] == plan["fc3_displacements"] + plan["fc2_displacements"] + 2


def test_fc2_distance_requires_separate_supercell(tmp_path: Path) -> None:
    path = write_anh_config(tmp_path, extra_anh="  fc2_displacement_angstrom: 0.01\n")
    with pytest.raises(ConfigError, match="必须同时设置独立的 fc2_supercell"):
        load_anh_config(path)


def test_fc2_diagonal_requires_separate_supercell(tmp_path: Path) -> None:
    path = write_anh_config(tmp_path, extra_anh="  fc2_is_diagonal: true\n")
    with pytest.raises(ConfigError, match="必须同时设置独立的 fc2_supercell"):
        load_anh_config(path)


def test_anh_snapshot_is_self_contained_with_born(tmp_path: Path) -> None:
    born = tmp_path / "BORN"
    born.write_text("default\n10 0 0 0 10 0 0 0 10\n1 0 0 0 1 0 0 0 1\n", encoding="utf-8")
    path = write_anh_config(tmp_path, extra_structure="  born_file: BORN\n")
    config = load_anh_config(path)
    paths, _, _ = choose_anh_run(config)
    snap = create_anh_snapshot(config, paths.root)
    assert snap.structure.file.is_relative_to(paths.root)
    assert snap.structure.born_file is not None and snap.structure.born_file.is_relative_to(paths.root)
    assert snap.methods["dp"].model.is_relative_to(paths.root)
    paths2 = AnhRunPaths(paths.root)
    assert paths2.state_file.is_file()
    assert displacement_plan(snap)["nac_enabled"] is True
    ph3 = build_phono3py(snap, snap.structure.file)
    assert ph3.nac_params is not None
    assert ph3.nac_params["factor"] == pytest.approx(14.39965172592227)

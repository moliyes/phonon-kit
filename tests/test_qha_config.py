from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from phonon_kit.errors import ConfigError
from phonon_kit.initializer import POSCAR
from phonon_kit.qha_config import load_qha_config
from phonon_kit.qha_validation import qha_plan, validate_qha_compositions


def write_qha_config(root: Path, *, second_poscar: str | None = None) -> Path:
    (root / "inputs" / "a").mkdir(parents=True)
    (root / "inputs" / "b").mkdir(parents=True)
    (root / "inputs" / "a" / "POSCAR").write_text(POSCAR, encoding="utf-8")
    (root / "inputs" / "b" / "POSCAR").write_text(second_poscar or POSCAR, encoding="utf-8")
    (root / "model.pth").write_bytes(b"fake")
    data = {
        "schema_version": 1,
        "project": {"name": "test-qha", "runs_dir": "runs"},
        "phases": {
            "a": {"structure": "inputs/a/POSCAR", "phonon": {"supercell": [1, 1, 1]}},
            "b": {
                "structure": "inputs/b/POSCAR",
                "volume_ratios": [0.9, 0.95, 1.0, 1.05, 1.1],
                "phonon": {"mesh": [8, 8, 8]},
            },
        },
        "volume_grid": {"ratios": [0.88, 0.94, 1.0, 1.06, 1.12]},
        "volume_relaxation": {},
        "phonon": {"supercell": [2, 2, 2], "mesh": [6, 6, 6]},
        "qha": {
            "temperature": {"min_k": 0, "max_k": 20, "step_k": 10},
            "pressure": {"min_gpa": 0, "max_gpa": 1, "step_gpa": 0.5},
        },
        "methods": {"dp": {"type": "deepmd", "model": "model.pth", "device": "cpu"}},
    }
    path = root / "qha.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_qha_config_inheritance_and_plan(tmp_path: Path):
    config = load_qha_config(write_qha_config(tmp_path))
    assert config.phases["a"].volume_ratios == (0.88, 0.94, 1.0, 1.06, 1.12)
    assert config.phases["a"].phonon.supercell == (1, 1, 1)
    assert config.phases["a"].phonon.mesh == (6, 6, 6)
    assert config.phases["b"].phonon.supercell == (2, 2, 2)
    assert config.phases["b"].phonon.mesh == (8, 8, 8)
    plan = qha_plan(config)
    assert plan["temperature_points_requested"] == 3
    assert plan["temperature_points_computed"] == 4
    assert plan["pressure_points"] == 3
    assert plan["methods"]["dp"]["volume_relaxations"] == 10


def test_qha_unknown_field_rejected(tmp_path: Path):
    path = write_qha_config(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["qha"]["mystery"] = 1
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="未知字段"):
        load_qha_config(path)


def test_qha_requires_same_reduced_composition(tmp_path: Path):
    other = POSCAR.replace("O Si\n4 2", "O Si\n3 3")
    config = load_qha_config(write_qha_config(tmp_path, second_poscar=other))
    with pytest.raises(ConfigError, match="相同最简组成"):
        validate_qha_compositions(config)


def test_qha_strict_boolean(tmp_path: Path):
    path = write_qha_config(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["methods"]["dp"]["enabled"] = "false"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="true 或 false"):
        load_qha_config(path)

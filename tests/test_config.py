from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from phonon_kit.config import ConfigError, DeepMDMethod, load_config

from conftest import write_config


def test_paths_are_relative_to_yaml(config_path: Path):
    config = load_config(config_path)
    method = config.methods["dp"]
    assert config.structure.file == config_path.parent / "POSCAR"
    assert isinstance(method, DeepMDMethod)
    assert method.model == config_path.parent / "model.pth"
    assert config.phonon.supercell == (1, 1, 1)


def test_unknown_field_is_rejected(config_path: Path):
    text = config_path.read_text(encoding="utf-8").replace("schema_version: 1", "schema_version: 1\nunknown: true")
    config_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="未知字段"):
        load_config(config_path)


def test_pt2_head_is_rejected(tmp_path: Path):
    path = write_config(tmp_path, model="model.pt2")
    text = path.read_text(encoding="utf-8").replace("    device: cpu", "    device: cpu\n    head: branch")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="固化 head"):
        load_config(path)


def test_fingerprint_changes_when_structure_changes(config_path: Path):
    config = load_config(config_path)
    before = config.fingerprint()
    poscar = config_path.parent / "POSCAR"
    poscar.write_text(poscar.read_text(encoding="utf-8").replace("5.43", "5.44", 1), encoding="utf-8")
    after = load_config(config_path).fingerprint()
    assert before != after


def test_resolved_snapshot_is_valid_config(config_path: Path):
    config = load_config(config_path)
    snapshot = config_path.parent / "resolved.yaml"
    snapshot.write_text(yaml.safe_dump(config.resolved_dict(), sort_keys=False), encoding="utf-8")
    restored = load_config(snapshot)
    assert restored.fingerprint() == config.fingerprint()

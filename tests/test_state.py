from __future__ import annotations

from pathlib import Path

import pytest

from phonon_kit.config import load_config
from phonon_kit.errors import RunStateError
from phonon_kit.state import StateStore, choose_run


def test_run_versioning_and_resume(config_path: Path):
    config = load_config(config_path)
    first, state, created = choose_run(config, ["dp"])
    assert created
    assert first.root.name == "test-001"
    same, _, created = choose_run(config, ["dp"])
    assert not created
    assert same.root == first.root

    StateStore(first).update(status="completed")
    poscar = config.structure.file
    poscar.write_text(poscar.read_text(encoding="utf-8").replace("5.43", "5.45", 1), encoding="utf-8")
    changed = load_config(config_path)
    second, _, created = choose_run(changed, ["dp"])
    assert created
    assert second.root.name == "test-002"


def test_changed_config_does_not_mix_with_incomplete_run(config_path: Path):
    config = load_config(config_path)
    choose_run(config, ["dp"])
    config.structure.file.write_text(config.structure.file.read_text().replace("5.43", "5.46", 1))
    with pytest.raises(RunStateError, match="尚未完成"):
        choose_run(load_config(config_path), ["dp"])


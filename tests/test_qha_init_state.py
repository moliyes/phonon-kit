from __future__ import annotations

from pathlib import Path

from phonon_kit.qha_config import load_qha_config
from phonon_kit.qha_initializer import init_qha_case
from phonon_kit.qha_state import choose_qha_run, matching_qha_run


def test_qha_init_and_versioned_state(tmp_path: Path):
    case = init_qha_case(tmp_path / "case", ["alpha", "beta"])
    assert (case / "qha.yaml").is_file()
    assert (case / "inputs" / "vasp" / "volume_relax" / "INCAR").is_file()
    config = load_qha_config(case / "qha.yaml")
    paths, state, created = choose_qha_run(config, ["dpa4"])
    assert created
    assert state["workflow"] == "qha"
    assert len(state["methods"]["dpa4"]["phases"]["alpha"]["volumes"]) == 7
    assert matching_qha_run(config)[0] == paths


def test_qha_completed_config_creates_next_run(tmp_path: Path):
    case = init_qha_case(tmp_path / "case", ["alpha", "beta"])
    config = load_qha_config(case / "qha.yaml")
    first, state, _ = choose_qha_run(config, ["dpa4"])
    state["status"] = "completed"
    from phonon_kit.qha_state import QHAStateStore

    QHAStateStore(first).save(state)
    second, _, created = choose_qha_run(config, ["dpa4"], force_new=True)
    assert created
    assert second.root.name.endswith("-002")

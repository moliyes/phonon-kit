from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from phonon_kit.errors import ConfigError
from phonon_kit.qha_config import QHAVaspMethod, load_qha_config
from phonon_kit.qha_initializer import init_qha_case
from phonon_kit.qha_state import choose_qha_run
from phonon_kit.qha_validation import validate_qha_vasp_templates
from phonon_kit.qha_vasp import parse_e0_energy, prepare_relaxation_tasks


def enable_fake_vasp(case: Path):
    for stage in ("volume_relax", "static", "phonon"):
        (case / "inputs" / "vasp" / stage / "POTCAR").write_bytes(b"same-potcar")
    shutil.copy2(case / "inputs" / "dispatcher" / "machine.json.example", case / "inputs" / "dispatcher" / "machine.json")
    shutil.copy2(case / "inputs" / "dispatcher" / "resources.json.example", case / "inputs" / "dispatcher" / "resources.json")
    config_path = case / "qha.yaml"
    data = yaml.safe_load(config_path.read_text())
    data["methods"]["dft"]["enabled"] = True
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_qha_config(config_path)


def test_qha_vasp_templates_and_relaxation_jobs(tmp_path: Path, monkeypatch):
    case = init_qha_case(tmp_path / "case", ["alpha", "beta"])
    config = enable_fake_vasp(case)
    method = config.methods["dft"]
    assert isinstance(method, QHAVaspMethod)
    monkeypatch.setattr(
        "phonon_kit.qha_validation.validate_dispatcher_files",
        lambda method: {"machine_class": "Fake", "resources_class": "Fake"},
    )
    report = validate_qha_vasp_templates(config, method)
    assert report["potcar_sha256"]
    paths, _, _ = choose_qha_run(config, ["dft"])
    tasks = prepare_relaxation_tasks(config, method, paths)
    assert len(tasks) == 14
    assert all((task / "POSCAR").is_file() for task in tasks)


def test_qha_vasp_static_template_must_be_static(tmp_path: Path, monkeypatch):
    case = init_qha_case(tmp_path / "case", ["alpha", "beta"])
    config = enable_fake_vasp(case)
    static_incar = case / "inputs" / "vasp" / "static" / "INCAR"
    static_incar.write_text(static_incar.read_text().replace("NSW = 0", "NSW = 2"), encoding="utf-8")
    config = load_qha_config(case / "qha.yaml")
    monkeypatch.setattr("phonon_kit.qha_validation.validate_dispatcher_files", lambda method: {})
    with pytest.raises(ConfigError, match="NSW 必须为 0"):
        validate_qha_vasp_templates(config, config.methods["dft"])


def test_parse_vasp_e0_energy(tmp_path: Path):
    xml = tmp_path / "vasprun.xml"
    xml.write_text(
        "<modeling><calculation><energy><i name='e_0_energy'>-10.5</i></energy></calculation>"
        "<calculation><energy><i name='e_0_energy'>-10.75</i></energy></calculation></modeling>",
        encoding="utf-8",
    )
    assert parse_e0_energy(xml) == -10.75

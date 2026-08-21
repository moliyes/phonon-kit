from __future__ import annotations

import stat
from pathlib import Path

import yaml

from phonon_kit.cli import _qha_run_target, _single_run_target, main
from phonon_kit.config import DeepMDMethod, VaspMethod, load_config
from phonon_kit.qha_config import QHAVaspMethod, load_qha_config
from phonon_kit.qha_initializer import init_qha_case
from phonon_kit.qha_state import QHAStateStore, choose_qha_run
from phonon_kit.snapshot import create_config_snapshot, create_qha_config_snapshot
from phonon_kit.state import StateStore, choose_run

from conftest import POSCAR
from test_qha_vasp import enable_fake_vasp


def _single_vasp_config(root: Path) -> Path:
    (root / "POSCAR").write_text(POSCAR, encoding="utf-8")
    template = root / "vasp"
    template.mkdir()
    (template / "INCAR").write_text("IBRION=-1\nNSW=0\n", encoding="utf-8")
    (template / "KPOINTS").write_text("KPOINTS\n", encoding="utf-8")
    (template / "POTCAR").write_text("POTCAR\n", encoding="utf-8")
    (root / "machine.json").write_text('{"machine": "original"}\n', encoding="utf-8")
    (root / "resources.json").write_text("{}\n", encoding="utf-8")
    data = {
        "schema_version": 1,
        "project": {"name": "snapshot", "runs_dir": "runs"},
        "structure": {"file": "POSCAR"},
        "relaxation": {"enabled": False},
        "phonon": {"supercell": [1, 1, 1]},
        "methods": {
            "dft": {
                "type": "vasp",
                "template_dir": "vasp",
                "executor": {
                    "type": "dpdispatcher",
                    "machine": "machine.json",
                    "resources": "resources.json",
                    "command": "vasp_std",
                },
            }
        },
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_single_run_snapshot_is_independent_from_case_inputs(tmp_path: Path):
    source_path = _single_vasp_config(tmp_path)
    source = load_config(source_path)
    paths, _, _ = choose_run(source, ["dft"])
    snapshot = create_config_snapshot(source, paths.root)

    source.structure.file.write_text(source.structure.file.read_text().replace("5.43", "6.00"), encoding="utf-8")
    method = source.methods["dft"]
    assert isinstance(method, VaspMethod)
    (method.template_dir / "INCAR").write_text("IBRION=-1\nNSW=0\nEDIFF=1E-4\n", encoding="utf-8")
    method.executor.machine.write_text('{"machine": "changed"}\n', encoding="utf-8")

    restored = load_config(paths.root / "config.resolved.yaml")
    restored_method = restored.methods["dft"]
    assert isinstance(restored_method, VaspMethod)
    assert "5.43" in restored.structure.file.read_text(encoding="utf-8")
    assert "EDIFF" not in (restored_method.template_dir / "INCAR").read_text(encoding="utf-8")
    assert "original" in restored_method.executor.machine.read_text(encoding="utf-8")
    assert stat.S_IMODE(restored_method.executor.machine.stat().st_mode) == 0o600
    assert snapshot.path == paths.root / "config.resolved.yaml"
    assert snapshot.project.runs_dir == paths.root.parent

    config_path, explicit = _single_run_target(paths.root)
    assert config_path == paths.root / "config.resolved.yaml"
    assert explicit is not None and explicit.root == paths.root
    config_path, explicit = _single_run_target(paths.root / "config.resolved.yaml")
    assert explicit is not None and config_path.name == "config.resolved.yaml"
    StateStore(paths).update(status="completed")
    assert main(["resume", str(paths.root)]) == 0


def test_deepmd_model_is_copied_into_run(tmp_path: Path, config_path: Path):
    source = load_config(config_path)
    paths, _, _ = choose_run(source, ["dp"])
    snapshot = create_config_snapshot(source, paths.root)
    method = snapshot.methods["dp"]
    assert isinstance(method, DeepMDMethod)
    assert method.model.parent == paths.root / "inputs" / "models"
    assert method.model.read_bytes() == b"dummy-model"


def test_force_new_runs_keep_independent_input_snapshots(config_path: Path):
    first_config = load_config(config_path)
    first_paths, _, _ = choose_run(first_config, ["dp"])
    first = create_config_snapshot(first_config, first_paths.root)

    first_config.structure.file.write_text(
        first_config.structure.file.read_text(encoding="utf-8").replace("5.43", "5.80", 1),
        encoding="utf-8",
    )
    second_config = load_config(config_path)
    second_paths, _, _ = choose_run(second_config, ["dp"], force_new=True)
    second = create_config_snapshot(second_config, second_paths.root)

    assert first_paths.root.name == "test-001"
    assert second_paths.root.name == "test-002"
    assert "5.43" in first.structure.file.read_text(encoding="utf-8")
    assert "5.80" in second.structure.file.read_text(encoding="utf-8")


def test_qha_run_snapshot_copies_phases_templates_and_dispatcher(tmp_path: Path):
    case = init_qha_case(tmp_path / "case", ["alpha", "beta"])
    config = enable_fake_vasp(case)
    data = yaml.safe_load((case / "qha.yaml").read_text(encoding="utf-8"))
    data["methods"]["dpa4"]["enabled"] = False
    (case / "qha.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    config = load_qha_config(case / "qha.yaml")
    paths, _, _ = choose_qha_run(config, ["dft"])
    snapshot = create_qha_config_snapshot(config, paths.root)

    method = snapshot.methods["dft"]
    assert isinstance(method, QHAVaspMethod)
    assert all(phase.structure.is_relative_to(paths.root / "inputs") for phase in snapshot.phases.values())
    assert method.templates.phonon.is_relative_to(paths.root / "inputs")
    assert method.executor.machine.is_relative_to(paths.root / "inputs")
    assert stat.S_IMODE(method.executor.machine.stat().st_mode) == 0o600

    config_path, explicit = _qha_run_target(paths.root)
    assert config_path == paths.root / "config.resolved.yaml"
    assert explicit is not None and explicit.root == paths.root
    QHAStateStore(paths).update(status="completed")
    assert main(["qha", "resume", str(paths.root)]) == 0

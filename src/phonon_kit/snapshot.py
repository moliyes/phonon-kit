from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .config import Config, DeepMDMethod, VaspMethod, load_config
from .qha_config import QHAConfig, QHAVaspMethod, QHA_VASP_STAGES, load_qha_config
from .util import atomic_write_text


SNAPSHOT_CONFIG = "config.resolved.yaml"


def _copy_file(source: Path, destination: Path, *, private: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if private:
        destination.chmod(0o600)


def _copy_directory(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination)


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _write_snapshot(root: Path, temporary_inputs: Path, data: dict[str, Any]) -> None:
    inputs = root / "inputs"
    if inputs.exists():
        raise FileExistsError(f"运行输入快照已存在但配置快照缺失: {inputs}")
    os.replace(temporary_inputs, inputs)
    atomic_write_text(
        root / SNAPSHOT_CONFIG,
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
    )


def create_config_snapshot(config: Config, root: Path) -> Config:
    snapshot_path = root / SNAPSHOT_CONFIG
    if snapshot_path.is_file():
        return load_config(snapshot_path)

    with tempfile.TemporaryDirectory(prefix=".inputs-", dir=root) as temporary:
        temporary_inputs = Path(temporary)
        final_inputs = root / "inputs"
        data = config.resolved_dict()
        data["project"]["runs_dir"] = ".."

        structure = temporary_inputs / "structure" / "POSCAR"
        _copy_file(config.structure.file, structure)
        data["structure"]["file"] = _relative(final_inputs / "structure" / "POSCAR", root)

        if config.phonon.unfolding is not None:
            reference = temporary_inputs / "unfolding" / "SPOSCAR-Ref.vasp"
            _copy_file(config.phonon.unfolding.reference_supercell, reference)
            data["phonon"]["unfolding"]["reference_supercell"] = _relative(
                final_inputs / "unfolding" / reference.name, root
            )

        methods: dict[str, Any] = {}
        for name, method in config.enabled_methods.items():
            item = data["methods"][name]
            if isinstance(method, DeepMDMethod):
                model = temporary_inputs / "models" / f"{name}{method.model.suffix.lower()}"
                _copy_file(method.model, model)
                item["model"] = _relative(final_inputs / "models" / model.name, root)
            elif isinstance(method, VaspMethod):
                method_root = temporary_inputs / "methods" / name
                template = method_root / "vasp"
                _copy_directory(method.template_dir, template)
                machine = method_root / "dispatcher" / "machine.json"
                resources = method_root / "dispatcher" / "resources.json"
                _copy_file(method.executor.machine, machine, private=True)
                _copy_file(method.executor.resources, resources, private=True)
                item["template_dir"] = _relative(final_inputs / "methods" / name / "vasp", root)
                item["executor"]["machine"] = _relative(
                    final_inputs / "methods" / name / "dispatcher" / "machine.json", root
                )
                item["executor"]["resources"] = _relative(
                    final_inputs / "methods" / name / "dispatcher" / "resources.json", root
                )
            methods[name] = item
        data["methods"] = methods
        _write_snapshot(root, temporary_inputs, data)
    return load_config(snapshot_path)


def create_qha_config_snapshot(config: QHAConfig, root: Path) -> QHAConfig:
    snapshot_path = root / SNAPSHOT_CONFIG
    if snapshot_path.is_file():
        return load_qha_config(snapshot_path)

    with tempfile.TemporaryDirectory(prefix=".inputs-", dir=root) as temporary:
        temporary_inputs = Path(temporary)
        final_inputs = root / "inputs"
        data = config.resolved_dict()
        data["project"]["runs_dir"] = ".."

        for name, phase in config.phases.items():
            structure = temporary_inputs / "phases" / name / "POSCAR"
            _copy_file(phase.structure, structure)
            data["phases"][name]["structure"] = _relative(
                final_inputs / "phases" / name / "POSCAR", root
            )

        methods: dict[str, Any] = {}
        for name, method in config.enabled_methods.items():
            item = data["methods"][name]
            if isinstance(method, DeepMDMethod):
                model = temporary_inputs / "models" / f"{name}{method.model.suffix.lower()}"
                _copy_file(method.model, model)
                item["model"] = _relative(final_inputs / "models" / model.name, root)
            elif isinstance(method, QHAVaspMethod):
                method_root = temporary_inputs / "methods" / name
                default_templates: dict[str, str] = {}
                for stage in QHA_VASP_STAGES:
                    target = method_root / "templates" / "default" / stage
                    _copy_directory(method.templates.get(stage), target)
                    default_templates[stage] = _relative(
                        final_inputs / "methods" / name / "templates" / "default" / stage, root
                    )
                item["templates"] = default_templates
                phase_templates: dict[str, dict[str, str]] = {}
                for phase_name, templates in method.phase_templates.items():
                    phase_templates[phase_name] = {}
                    for stage in QHA_VASP_STAGES:
                        target = method_root / "templates" / "phases" / phase_name / stage
                        _copy_directory(templates.get(stage), target)
                        phase_templates[phase_name][stage] = _relative(
                            final_inputs / "methods" / name / "templates" / "phases" / phase_name / stage,
                            root,
                        )
                item["phase_templates"] = phase_templates
                machine = method_root / "dispatcher" / "machine.json"
                resources = method_root / "dispatcher" / "resources.json"
                _copy_file(method.executor.machine, machine, private=True)
                _copy_file(method.executor.resources, resources, private=True)
                item["executor"]["machine"] = _relative(
                    final_inputs / "methods" / name / "dispatcher" / "machine.json", root
                )
                item["executor"]["resources"] = _relative(
                    final_inputs / "methods" / name / "dispatcher" / "resources.json", root
                )
            methods[name] = item
        data["methods"] = methods
        _write_snapshot(root, temporary_inputs, data)
    return load_qha_config(snapshot_path)

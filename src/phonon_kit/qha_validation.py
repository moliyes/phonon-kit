from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .config import DeepMDMethod
from .dispatcher import validate_dispatcher_files
from .errors import ConfigError
from .providers.deepmd import validate_method
from .providers.vasp import parse_incar
from .qha_config import QHAConfig, QHAVaspMethod, QHA_VASP_STAGES
from .qha_structure import composition_metadata, estimate_displacements
from .util import sha256_file


def validate_qha_compositions(config: QHAConfig) -> dict[str, Any]:
    metadata = {name: composition_metadata(phase.structure) for name, phase in config.phases.items()}
    keys = {item["reduced_key"] for item in metadata.values()}
    if len(keys) != 1:
        details = ", ".join(f"{name}={item['formula_unit']}" for name, item in metadata.items())
        raise ConfigError(f"QHA 只支持相同最简组成的多晶型；当前为 {details}")
    formula = next(iter(metadata.values()))["formula_unit"]
    return {"formula_unit": formula, "phases": metadata}


def validate_qha_vasp_templates(config: QHAConfig, method: QHAVaspMethod) -> dict[str, Any]:
    warnings: list[str] = []
    potcar_hashes: set[str] = set()
    templates: dict[str, Any] = {}
    for phase in config.phases:
        templates[phase] = {}
        for stage in QHA_VASP_STAGES:
            directory = method.template_for(phase, stage)
            incar = parse_incar(directory / "INCAR")
            potcar_hashes.add(sha256_file(directory / "POTCAR"))
            if stage == "volume_relax":
                if incar.get("ISIF", "").split()[0] != "4":
                    raise ConfigError(f"{method.name}/{phase}/volume_relax: INCAR 必须设置 ISIF = 4")
                try:
                    nsw = int(float(incar.get("NSW", "0").split()[0]))
                except ValueError as exc:
                    raise ConfigError(f"{method.name}/{phase}/volume_relax: 无法解析 NSW") from exc
                if nsw <= 0:
                    raise ConfigError(f"{method.name}/{phase}/volume_relax: NSW 必须大于 0")
                if incar.get("IBRION", "-1").split()[0] not in {"1", "2", "3"}:
                    raise ConfigError(f"{method.name}/{phase}/volume_relax: IBRION 必须为 1、2 或 3")
            else:
                if incar.get("IBRION", "-1").split()[0] != "-1":
                    raise ConfigError(f"{method.name}/{phase}/{stage}: IBRION 必须为 -1")
                try:
                    nsw = int(float(incar.get("NSW", "0").split()[0]))
                except ValueError as exc:
                    raise ConfigError(f"{method.name}/{phase}/{stage}: 无法解析 NSW") from exc
                if nsw != 0:
                    raise ConfigError(f"{method.name}/{phase}/{stage}: NSW 必须为 0")
            if incar.get("LREAL", ".FALSE.").upper() not in {".FALSE.", "F", "FALSE"}:
                warnings.append(f"{method.name}/{phase}/{stage}: 建议 LREAL = .FALSE.")
            if "EDIFF" not in incar:
                warnings.append(f"{method.name}/{phase}/{stage}: 建议显式设置严格 EDIFF")
            templates[phase][stage] = str(directory)
    if len(potcar_hashes) != 1:
        raise ConfigError(f"{method.name}: 各相和三个阶段的 POTCAR 内容必须完全一致")
    dispatcher = validate_dispatcher_files(_dispatcher_compatible_method(method))
    return {"templates": templates, "potcar_sha256": next(iter(potcar_hashes)), "warnings": warnings, "dispatcher": dispatcher}


def _dispatcher_compatible_method(method: QHAVaspMethod):
    from .config import VaspMethod

    return VaspMethod(method.name, "vasp", method.templates.phonon, method.executor, method.enabled)


def qha_plan(config: QHAConfig) -> dict[str, Any]:
    composition = validate_qha_compositions(config)
    phases: dict[str, Any] = {}
    for name, phase in config.phases.items():
        estimate = estimate_displacements(phase.structure, phase.phonon)
        reference_volume = estimate["primitive_volume_angstrom3"]
        phases[name] = {
            "formula_unit": composition["formula_unit"],
            "reference_volume_angstrom3": reference_volume,
            "volume_ratios": list(phase.volume_ratios),
            "target_volumes_angstrom3": [reference_volume * ratio for ratio in phase.volume_ratios],
            **estimate,
            "note": "体积和位移数基于一次性规范化后的 primitive cell；固定体积弛豫后位移数仍可能改变。",
        }
    methods: dict[str, Any] = {}
    for name, method in config.enabled_methods.items():
        n_volumes = sum(len(phase.volume_ratios) for phase in config.phases.values())
        n_displacements = sum(
            len(config.phases[phase].volume_ratios) * int(item["n_displacements"])
            for phase, item in phases.items()
        )
        if isinstance(method, DeepMDMethod):
            methods[name] = {
                "type": "deepmd",
                "volume_relaxations": n_volumes,
                "static_energy_evaluations": n_volumes,
                "finite_displacement_force_evaluations": n_displacements,
            }
        else:
            methods[name] = {
                "type": "vasp",
                "volume_relaxation_jobs": n_volumes,
                "static_jobs": n_volumes,
                "phonon_force_jobs_estimate": n_displacements,
                "total_vasp_jobs_estimate": 2 * n_volumes + n_displacements,
            }
    return {
        "workflow": "qha",
        "formula_unit": composition["formula_unit"],
        "phases": phases,
        "methods": methods,
        "temperature_points_requested": len(config.qha.temperature.values()),
        "temperature_points_computed": len(config.qha.temperature.values(include_extra=True)),
        "pressure_points": len(config.qha.pressure.values()),
    }


def validate_qha_runtime(
    config: QHAConfig,
    selected: list[str] | None = None,
    *,
    real_inference: bool = True,
) -> dict[str, Any]:
    names = selected or list(config.enabled_methods)
    invalid = [name for name in names if name not in config.enabled_methods]
    if invalid:
        raise ConfigError(f"方法未定义或未启用: {', '.join(invalid)}")
    report = qha_plan(config)
    method_report: dict[str, Any] = {}
    for name in names:
        method = config.methods[name]
        if isinstance(method, DeepMDMethod):
            phase_reports: dict[str, Any] = {}
            if real_inference:
                with tempfile.TemporaryDirectory(prefix="phonon-kit-qha-validate-") as tmp:
                    root = Path(tmp)
                    for phase_name, phase in config.phases.items():
                        phase_reports[phase_name] = validate_method(
                            method,
                            phase.structure,
                            root / name / phase_name,
                            root / "logs",
                        )
            method_report[name] = {"type": "deepmd", "inference": phase_reports}
        else:
            method_report[name] = {"type": "vasp", **validate_qha_vasp_templates(config, method)}
    report["validation"] = method_report
    return report

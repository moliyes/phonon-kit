from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from .config import DeepMDMethod, DispatcherConfig, SUPPORTED_MODEL_SUFFIXES, builtin_model_path
from .errors import ConfigError
from .util import safe_name, sha256_file, sha256_json
from .vasp_input import missing_vasp_template_inputs


QHA_VASP_STAGES = ("volume_relax", "static", "phonon")


@dataclass(frozen=True)
class QHAProjectConfig:
    name: str
    runs_dir: Path


@dataclass(frozen=True)
class QHAPhononConfig:
    supercell: tuple[int, int, int] = (2, 2, 2)
    displacement_angstrom: float = 0.01
    symmetry_tolerance: float = 1.0e-5
    mesh: tuple[int, int, int] = (30, 30, 30)
    significant_imaginary_thz: float = -0.1


@dataclass(frozen=True)
class QHAPhaseConfig:
    name: str
    structure: Path
    volume_ratios: tuple[float, ...]
    phonon: QHAPhononConfig
    label: str
    color: str | None = None


@dataclass(frozen=True)
class VolumeRelaxationConfig:
    fmax_ev_angstrom: float = 0.01
    max_deviatoric_stress_gpa: float = 0.1
    max_steps: int = 1000
    trajectory_interval: int = 10
    volume_tolerance_relative: float = 1.0e-5


@dataclass(frozen=True)
class GridConfig:
    minimum: float
    maximum: float
    step: float

    def values(self, *, include_extra: bool = False) -> tuple[float, ...]:
        stop = self.maximum + (self.step if include_extra else 0.0)
        count = int(round((stop - self.minimum) / self.step))
        return tuple(self.minimum + index * self.step for index in range(count + 1))


@dataclass(frozen=True)
class QHASettings:
    eos: Literal["vinet", "birch_murnaghan", "murnaghan"]
    temperature: GridConfig
    pressure: GridConfig
    imaginary_policy: Literal["exclude"] = "exclude"


@dataclass(frozen=True)
class QHAVaspTemplates:
    volume_relax: Path
    static: Path
    phonon: Path

    def get(self, stage: str) -> Path:
        if stage not in QHA_VASP_STAGES:
            raise KeyError(stage)
        return getattr(self, stage)


@dataclass(frozen=True)
class QHAVaspMethod:
    name: str
    type: Literal["vasp"]
    templates: QHAVaspTemplates
    phase_templates: dict[str, QHAVaspTemplates]
    executor: DispatcherConfig
    enabled: bool = True

    def template_for(self, phase: str, stage: str) -> Path:
        return self.phase_templates.get(phase, self.templates).get(stage)


QHAMethodConfig = DeepMDMethod | QHAVaspMethod


@dataclass(frozen=True)
class QHAConfig:
    path: Path
    schema_version: int
    project: QHAProjectConfig
    phases: dict[str, QHAPhaseConfig]
    volume_relaxation: VolumeRelaxationConfig
    phonon: QHAPhononConfig
    qha: QHASettings
    methods: dict[str, QHAMethodConfig]

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    @property
    def enabled_methods(self) -> dict[str, QHAMethodConfig]:
        return {name: method for name, method in self.methods.items() if method.enabled}

    def resolved_dict(self) -> dict[str, Any]:
        phases: dict[str, Any] = {}
        for name, phase in self.phases.items():
            phases[name] = {
                "structure": str(phase.structure),
                "label": phase.label,
                "color": phase.color,
                "volume_ratios": list(phase.volume_ratios),
                "phonon": _convert(asdict(phase.phonon)),
            }
        methods: dict[str, Any] = {}
        for name, method in self.methods.items():
            if isinstance(method, DeepMDMethod):
                methods[name] = {
                    "type": "deepmd",
                    "model": str(method.model),
                    "device": method.device,
                    "head": method.head,
                    "enabled": method.enabled,
                }
            else:
                methods[name] = {
                    "type": "vasp",
                    "templates": _templates_dict(method.templates),
                    "phase_templates": {
                        phase: _templates_dict(templates)
                        for phase, templates in method.phase_templates.items()
                    },
                    "executor": {
                        "type": "dpdispatcher",
                        "machine": str(method.executor.machine),
                        "resources": str(method.executor.resources),
                        "command": method.executor.command,
                        "clean_remote_after_success": method.executor.clean_remote_after_success,
                    },
                    "enabled": method.enabled,
                }
        return {
            "schema_version": self.schema_version,
            "project": {"name": self.project.name, "runs_dir": str(self.project.runs_dir)},
            "phases": phases,
            "volume_grid": {"ratios": list(next(iter(self.phases.values())).volume_ratios)},
            "volume_relaxation": _convert(asdict(self.volume_relaxation)),
            "phonon": _convert(asdict(self.phonon)),
            "qha": {
                "eos": self.qha.eos,
                "temperature": {
                    "min_k": self.qha.temperature.minimum,
                    "max_k": self.qha.temperature.maximum,
                    "step_k": self.qha.temperature.step,
                },
                "pressure": {
                    "min_gpa": self.qha.pressure.minimum,
                    "max_gpa": self.qha.pressure.maximum,
                    "step_gpa": self.qha.pressure.step,
                },
                "imaginary_policy": self.qha.imaginary_policy,
            },
            "methods": methods,
        }

    def fingerprint(self) -> str:
        payload = self.resolved_dict()
        files: dict[str, str] = {}
        for phase, settings in self.phases.items():
            files[f"phase:{phase}:structure"] = sha256_file(settings.structure)
        for name, method in self.enabled_methods.items():
            if isinstance(method, DeepMDMethod):
                files[f"method:{name}:model"] = sha256_file(method.model)
            else:
                for phase in self.phases:
                    for stage in QHA_VASP_STAGES:
                        directory = method.template_for(phase, stage)
                        for path in sorted(item for item in directory.iterdir() if item.is_file()):
                            files[f"method:{name}:{phase}:{stage}:{path.name}"] = sha256_file(path)
                files[f"method:{name}:machine"] = sha256_file(method.executor.machine)
                files[f"method:{name}:resources"] = sha256_file(method.executor.resources)
        payload["input_sha256"] = files
        return sha256_json(payload)


def _convert(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_convert(item) for item in value]
    if isinstance(value, dict):
        return {key: _convert(item) for key, item in value.items()}
    return value


def _templates_dict(templates: QHAVaspTemplates) -> dict[str, str]:
    return {stage: str(templates.get(stage)) for stage in QHA_VASP_STAGES}


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} 必须是 YAML mapping")
    return value


def _strict(data: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"{where} 含未知字段: {', '.join(unknown)}")


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} 必须为 true 或 false")
    return value


def _triple(value: Any, where: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigError(f"{where} 必须恰好包含三个正整数")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where} 必须恰好包含三个正整数") from exc
    if any(item <= 0 for item in result):
        raise ConfigError(f"{where} 必须恰好包含三个正整数")
    return result  # type: ignore[return-value]


def _ratios(value: Any, where: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{where} 必须是体积比例列表")
    try:
        ratios = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where} 含非数值体积比例") from exc
    if len(ratios) < 5 or any(item <= 0 for item in ratios):
        raise ConfigError(f"{where} 至少需要 5 个正数")
    if tuple(sorted(set(ratios))) != ratios:
        raise ConfigError(f"{where} 必须严格递增且不能重复")
    return ratios


def _phonon(raw: dict[str, Any], defaults: QHAPhononConfig, where: str) -> QHAPhononConfig:
    _strict(
        raw,
        {"supercell", "displacement_angstrom", "symmetry_tolerance", "mesh", "significant_imaginary_thz"},
        where,
    )
    result = QHAPhononConfig(
        supercell=_triple(raw.get("supercell", defaults.supercell), f"{where}.supercell"),
        displacement_angstrom=float(raw.get("displacement_angstrom", defaults.displacement_angstrom)),
        symmetry_tolerance=float(raw.get("symmetry_tolerance", defaults.symmetry_tolerance)),
        mesh=_triple(raw.get("mesh", defaults.mesh), f"{where}.mesh"),
        significant_imaginary_thz=float(raw.get("significant_imaginary_thz", defaults.significant_imaginary_thz)),
    )
    if result.displacement_angstrom <= 0 or result.symmetry_tolerance <= 0:
        raise ConfigError(f"{where} 的 displacement_angstrom 和 symmetry_tolerance 必须为正")
    if result.significant_imaginary_thz > 0:
        raise ConfigError(f"{where}.significant_imaginary_thz 必须为 0 或负数")
    return result


def _grid(raw: dict[str, Any], prefix: str, where: str) -> GridConfig:
    allowed = {f"min_{prefix}", f"max_{prefix}", f"step_{prefix}"}
    _strict(raw, allowed, where)
    defaults = (0.0, 1000.0, 10.0) if prefix == "k" else (0.0, 20.0, 0.25)
    minimum = float(raw.get(f"min_{prefix}", defaults[0]))
    maximum = float(raw.get(f"max_{prefix}", defaults[1]))
    step = float(raw.get(f"step_{prefix}", defaults[2]))
    if minimum < 0 or maximum < minimum or step <= 0:
        raise ConfigError(f"{where} 范围无效")
    quotient = (maximum - minimum) / step
    if abs(quotient - round(quotient)) > 1.0e-8:
        raise ConfigError(f"{where} 的范围必须能被步长整除")
    return GridConfig(minimum, maximum, step)


def _parse_templates(base: Path, raw: Any, where: str) -> QHAVaspTemplates:
    data = _mapping(raw, where)
    _strict(data, set(QHA_VASP_STAGES), where)
    missing = [stage for stage in QHA_VASP_STAGES if stage not in data]
    if missing:
        raise ConfigError(f"{where} 缺少: {', '.join(missing)}")
    return QHAVaspTemplates(*(_resolve(base, data[stage]) for stage in QHA_VASP_STAGES))


def load_qha_config(path: str | Path, *, require_inputs: bool = True) -> QHAConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"QHA 配置文件不存在: {config_path}")
    root = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "QHA 配置根节点")
    _strict(
        root,
        {"schema_version", "project", "phases", "volume_grid", "volume_relaxation", "phonon", "qha", "methods"},
        "QHA 配置根节点",
    )
    if root.get("schema_version") != 1:
        raise ConfigError("QHA schema_version 必须为 1")
    base = config_path.parent

    project_raw = _mapping(root.get("project"), "project")
    _strict(project_raw, {"name", "runs_dir"}, "project")
    try:
        project_name = safe_name(str(project_raw["name"]), what="project.name")
    except (KeyError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    project = QHAProjectConfig(project_name, _resolve(base, project_raw.get("runs_dir", "runs")))

    volume_raw = _mapping(root.get("volume_grid"), "volume_grid")
    _strict(volume_raw, {"ratios"}, "volume_grid")
    default_ratios = _ratios(volume_raw.get("ratios"), "volume_grid.ratios")

    global_phonon_raw = _mapping(root.get("phonon", {}), "phonon")
    global_phonon = _phonon(global_phonon_raw, QHAPhononConfig(), "phonon")

    phases_raw = _mapping(root.get("phases"), "phases")
    if len(phases_raw) < 2:
        raise ConfigError("QHA 至少需要定义两个晶相")
    phases: dict[str, QHAPhaseConfig] = {}
    for phase_name, value in phases_raw.items():
        try:
            name = safe_name(str(phase_name), what="phase name")
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        item = _mapping(value, f"phases.{name}")
        _strict(item, {"structure", "label", "color", "volume_ratios", "phonon"}, f"phases.{name}")
        if "structure" not in item:
            raise ConfigError(f"phases.{name}.structure 为必填项")
        phase_phonon = _phonon(
            _mapping(item.get("phonon", {}), f"phases.{name}.phonon"),
            global_phonon,
            f"phases.{name}.phonon",
        )
        phases[name] = QHAPhaseConfig(
            name=name,
            structure=_resolve(base, item["structure"]),
            volume_ratios=_ratios(item.get("volume_ratios", default_ratios), f"phases.{name}.volume_ratios"),
            phonon=phase_phonon,
            label=str(item.get("label", name)),
            color=None if item.get("color") is None else str(item["color"]),
        )

    relax_raw = _mapping(root.get("volume_relaxation", {}), "volume_relaxation")
    _strict(
        relax_raw,
        {"fmax_ev_angstrom", "max_deviatoric_stress_gpa", "max_steps", "trajectory_interval", "volume_tolerance_relative"},
        "volume_relaxation",
    )
    relaxation = VolumeRelaxationConfig(
        fmax_ev_angstrom=float(relax_raw.get("fmax_ev_angstrom", 0.01)),
        max_deviatoric_stress_gpa=float(relax_raw.get("max_deviatoric_stress_gpa", 0.1)),
        max_steps=int(relax_raw.get("max_steps", 1000)),
        trajectory_interval=int(relax_raw.get("trajectory_interval", 10)),
        volume_tolerance_relative=float(relax_raw.get("volume_tolerance_relative", 1.0e-5)),
    )
    if any(
        value <= 0
        for value in (
            relaxation.fmax_ev_angstrom,
            relaxation.max_deviatoric_stress_gpa,
            relaxation.max_steps,
            relaxation.trajectory_interval,
            relaxation.volume_tolerance_relative,
        )
    ):
        raise ConfigError("volume_relaxation 的所有数值必须为正")

    qha_raw = _mapping(root.get("qha", {}), "qha")
    _strict(qha_raw, {"eos", "temperature", "pressure", "imaginary_policy"}, "qha")
    eos = str(qha_raw.get("eos", "vinet"))
    if eos not in {"vinet", "birch_murnaghan", "murnaghan"}:
        raise ConfigError("qha.eos 必须为 vinet、birch_murnaghan 或 murnaghan")
    imaginary_policy = str(qha_raw.get("imaginary_policy", "exclude"))
    if imaginary_policy != "exclude":
        raise ConfigError("qha.imaginary_policy 当前仅支持 exclude")
    qha = QHASettings(
        eos=eos,  # type: ignore[arg-type]
        temperature=_grid(_mapping(qha_raw.get("temperature", {}), "qha.temperature"), "k", "qha.temperature"),
        pressure=_grid(_mapping(qha_raw.get("pressure", {}), "qha.pressure"), "gpa", "qha.pressure"),
        imaginary_policy="exclude",
    )

    methods_raw = _mapping(root.get("methods"), "methods")
    if not methods_raw:
        raise ConfigError("methods 至少需要定义一个方法")
    methods: dict[str, QHAMethodConfig] = {}
    for method_name, value in methods_raw.items():
        try:
            name = safe_name(str(method_name), what="method name")
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        item = _mapping(value, f"methods.{name}")
        kind = item.get("type")
        if kind == "deepmd":
            _strict(item, {"type", "model", "device", "head", "enabled"}, f"methods.{name}")
            if "model" not in item:
                raise ConfigError(f"methods.{name}.model 为必填项")
            model_spec = str(item["model"])
            model = builtin_model_path() if model_spec == "builtin:dpa4" else _resolve(base, model_spec)
            head = None if item.get("head") is None else str(item["head"])
            enabled = _boolean(item.get("enabled", True), f"methods.{name}.enabled")
            method = DeepMDMethod(name, "deepmd", model, model_spec, str(item.get("device", "cuda:0")), head, enabled)
            if model.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
                raise ConfigError(f"methods.{name}.model 不支持扩展名 {model.suffix}")
            if model.suffix.lower() == ".pt2" and head:
                raise ConfigError(f"methods.{name}: .pt2 已固化 head，不能再设置 head")
            methods[name] = method
        elif kind == "vasp":
            _strict(item, {"type", "templates", "phase_templates", "executor", "enabled"}, f"methods.{name}")
            if "templates" not in item or "executor" not in item:
                raise ConfigError(f"methods.{name} 需要 templates 和 executor")
            templates = _parse_templates(base, item["templates"], f"methods.{name}.templates")
            overrides_raw = _mapping(item.get("phase_templates", {}), f"methods.{name}.phase_templates")
            unknown_phases = sorted(set(overrides_raw) - set(phases))
            if unknown_phases:
                raise ConfigError(f"methods.{name}.phase_templates 含未知晶相: {', '.join(unknown_phases)}")
            phase_templates = {
                phase: _parse_templates(base, value, f"methods.{name}.phase_templates.{phase}")
                for phase, value in overrides_raw.items()
            }
            executor_raw = _mapping(item["executor"], f"methods.{name}.executor")
            _strict(executor_raw, {"type", "machine", "resources", "command", "clean_remote_after_success"}, f"methods.{name}.executor")
            if executor_raw.get("type") != "dpdispatcher":
                raise ConfigError(f"methods.{name}.executor.type 当前仅支持 dpdispatcher")
            missing = [key for key in ("machine", "resources", "command") if key not in executor_raw]
            if missing:
                raise ConfigError(f"methods.{name}.executor 缺少: {', '.join(missing)}")
            executor = DispatcherConfig(
                "dpdispatcher",
                _resolve(base, executor_raw["machine"]),
                _resolve(base, executor_raw["resources"]),
                str(executor_raw["command"]),
                _boolean(executor_raw.get("clean_remote_after_success", False), f"methods.{name}.executor.clean_remote_after_success"),
            )
            methods[name] = QHAVaspMethod(
                name,
                "vasp",
                templates,
                phase_templates,
                executor,
                _boolean(item.get("enabled", True), f"methods.{name}.enabled"),
            )
        else:
            raise ConfigError(f"methods.{name}.type 必须为 deepmd 或 vasp")
    if not any(method.enabled for method in methods.values()):
        raise ConfigError("至少需要启用一个方法")

    config = QHAConfig(config_path, 1, project, phases, relaxation, global_phonon, qha, methods)
    if require_inputs:
        validate_qha_input_paths(config)
    return config


def validate_qha_input_paths(config: QHAConfig) -> None:
    for name, phase in config.phases.items():
        if not phase.structure.is_file():
            raise ConfigError(f"晶相结构不存在 ({name}): {phase.structure}")
    for name, method in config.enabled_methods.items():
        if isinstance(method, DeepMDMethod):
            if not method.model.is_file():
                raise ConfigError(f"DeepMD 模型不存在 ({name}): {method.model}")
            continue
        if not method.executor.machine.is_file():
            raise ConfigError(f"DPDispatcher machine 配置不存在 ({name}): {method.executor.machine}")
        if not method.executor.resources.is_file():
            raise ConfigError(f"DPDispatcher resources 配置不存在 ({name}): {method.executor.resources}")
        if not method.executor.command.strip():
            raise ConfigError(f"methods.{name}.executor.command 不能为空")
        for phase in config.phases:
            for stage in QHA_VASP_STAGES:
                directory = method.template_for(phase, stage)
                if not directory.is_dir():
                    raise ConfigError(f"VASP 模板目录不存在 ({name}/{phase}/{stage}): {directory}")
                missing = missing_vasp_template_inputs(directory)
                if missing:
                    raise ConfigError(f"VASP 模板缺少 ({name}/{phase}/{stage}): {', '.join(missing)}")

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from .config import (
    DeepMDMethod,
    SUPPORTED_MODEL_SUFFIXES,
    builtin_model_path,
)
from .errors import ConfigError
from .util import safe_name, sha256_file, sha256_json


@dataclass(frozen=True)
class AnhProjectConfig:
    name: str
    runs_dir: Path


@dataclass(frozen=True)
class AnhStructureConfig:
    file: Path
    born_file: Path | None = None


@dataclass(frozen=True)
class AnharmonicConfig:
    supercell: tuple[int, int, int] = (2, 2, 2)
    fc2_supercell: tuple[int, int, int] | None = None
    displacement_angstrom: float = 0.03
    fc2_displacement_angstrom: float | None = None
    fc2_is_diagonal: bool = False
    primitive: Literal["auto"] = "auto"
    symmetry_tolerance: float = 1.0e-5
    subtract_residual_forces: bool = False
    mesh: tuple[int, int, int] = (11, 11, 11)
    temperature_min_k: float = 100.0
    temperature_max_k: float = 1000.0
    temperature_step_k: float = 100.0
    lifetime_temperature_k: float = 300.0
    significant_imaginary_thz: float = -0.1
    continue_on_imaginary: bool = True

    @property
    def temperatures(self) -> list[float]:
        count = int(round((self.temperature_max_k - self.temperature_min_k) / self.temperature_step_k))
        return [self.temperature_min_k + index * self.temperature_step_k for index in range(count + 1)]

    @property
    def resolved_fc2_displacement_angstrom(self) -> float:
        """Return the fc2 distance, falling back to the fc3 distance."""
        if self.fc2_displacement_angstrom is None:
            return self.displacement_angstrom
        return self.fc2_displacement_angstrom


@dataclass(frozen=True)
class AnhConfig:
    path: Path
    schema_version: int
    project: AnhProjectConfig
    structure: AnhStructureConfig
    anharmonic: AnharmonicConfig
    methods: dict[str, DeepMDMethod]

    @property
    def enabled_methods(self) -> dict[str, DeepMDMethod]:
        return {name: method for name, method in self.methods.items() if method.enabled}

    def resolved_dict(self) -> dict[str, Any]:
        methods = {
            name: {
                "type": "deepmd",
                "model": str(method.model),
                "device": method.device,
                "head": method.head,
                "enabled": method.enabled,
            }
            for name, method in self.methods.items()
        }
        return {
            "schema_version": self.schema_version,
            "project": {"name": self.project.name, "runs_dir": str(self.project.runs_dir)},
            "structure": {
                "file": str(self.structure.file),
                "born_file": None if self.structure.born_file is None else str(self.structure.born_file),
            },
            "anharmonic": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in asdict(self.anharmonic).items()
            },
            "methods": methods,
        }

    def fingerprint(self) -> str:
        payload = self.resolved_dict()
        files = {"structure": sha256_file(self.structure.file)}
        if self.structure.born_file is not None:
            files["born"] = sha256_file(self.structure.born_file)
        for name, method in self.enabled_methods.items():
            files[f"method:{name}:model"] = sha256_file(method.model)
        payload["input_sha256"] = files
        return sha256_json(payload)


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


def load_anh_config(path: str | Path, *, require_inputs: bool = True) -> AnhConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"三阶配置文件不存在: {config_path}")
    root = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "配置根节点")
    _strict(root, {"schema_version", "project", "structure", "anharmonic", "methods"}, "配置根节点")
    if root.get("schema_version") != 1:
        raise ConfigError("schema_version 必须为 1")
    base = config_path.parent

    project_raw = _mapping(root.get("project"), "project")
    _strict(project_raw, {"name", "runs_dir"}, "project")
    try:
        name = safe_name(str(project_raw["name"]), what="project.name")
    except (KeyError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    project = AnhProjectConfig(name, _resolve(base, project_raw.get("runs_dir", "runs")))

    structure_raw = _mapping(root.get("structure"), "structure")
    _strict(structure_raw, {"file", "born_file"}, "structure")
    if "file" not in structure_raw:
        raise ConfigError("structure.file 为必填项")
    born_value = structure_raw.get("born_file")
    structure = AnhStructureConfig(
        _resolve(base, structure_raw["file"]),
        None if born_value in (None, "") else _resolve(base, born_value),
    )

    anh_raw = _mapping(root.get("anharmonic", {}), "anharmonic")
    allowed = {
        "supercell", "fc2_supercell", "displacement_angstrom",
        "fc2_displacement_angstrom", "fc2_is_diagonal", "primitive",
        "symmetry_tolerance", "subtract_residual_forces", "mesh",
        "temperature_min_k", "temperature_max_k", "temperature_step_k",
        "lifetime_temperature_k", "significant_imaginary_thz", "continue_on_imaginary",
    }
    _strict(anh_raw, allowed, "anharmonic")
    fc2_value = anh_raw.get("fc2_supercell")
    fc2_displacement_value = anh_raw.get("fc2_displacement_angstrom")
    anh = AnharmonicConfig(
        supercell=_triple(anh_raw.get("supercell", [2, 2, 2]), "anharmonic.supercell"),
        fc2_supercell=None if fc2_value is None else _triple(fc2_value, "anharmonic.fc2_supercell"),
        displacement_angstrom=float(anh_raw.get("displacement_angstrom", 0.03)),
        fc2_displacement_angstrom=(
            None if fc2_displacement_value is None else float(fc2_displacement_value)
        ),
        fc2_is_diagonal=bool(anh_raw.get("fc2_is_diagonal", False)),
        primitive=str(anh_raw.get("primitive", "auto")),  # type: ignore[arg-type]
        symmetry_tolerance=float(anh_raw.get("symmetry_tolerance", 1.0e-5)),
        subtract_residual_forces=bool(anh_raw.get("subtract_residual_forces", False)),
        mesh=_triple(anh_raw.get("mesh", [11, 11, 11]), "anharmonic.mesh"),
        temperature_min_k=float(anh_raw.get("temperature_min_k", 100)),
        temperature_max_k=float(anh_raw.get("temperature_max_k", 1000)),
        temperature_step_k=float(anh_raw.get("temperature_step_k", 100)),
        lifetime_temperature_k=float(anh_raw.get("lifetime_temperature_k", 300)),
        significant_imaginary_thz=float(anh_raw.get("significant_imaginary_thz", -0.1)),
        continue_on_imaginary=bool(anh_raw.get("continue_on_imaginary", True)),
    )
    if anh.primitive != "auto":
        raise ConfigError("anharmonic.primitive 当前仅支持 auto")
    if (
        anh.displacement_angstrom <= 0
        or anh.resolved_fc2_displacement_angstrom <= 0
        or anh.symmetry_tolerance <= 0
    ):
        raise ConfigError("位移距离和对称性容差必须为正数")
    if (
        anh.fc2_displacement_angstrom is not None or anh.fc2_is_diagonal
    ) and anh.fc2_supercell is None:
        raise ConfigError("设置 fc2 独立位移参数时必须同时设置独立的 fc2_supercell")
    if anh.temperature_min_k <= 0 or anh.temperature_max_k < anh.temperature_min_k or anh.temperature_step_k <= 0:
        raise ConfigError("三阶热导率温度范围无效，且最低温度必须大于 0 K")
    span = (anh.temperature_max_k - anh.temperature_min_k) / anh.temperature_step_k
    if abs(span - round(span)) > 1e-8:
        raise ConfigError("温度最大值必须落在 min + n*step 网格上")
    if not any(abs(t - anh.lifetime_temperature_k) < 1e-8 for t in anh.temperatures):
        raise ConfigError("lifetime_temperature_k 必须落在温度网格上")
    if anh.significant_imaginary_thz > 0:
        raise ConfigError("significant_imaginary_thz 应为 0 或负数")

    methods_raw = _mapping(root.get("methods"), "methods")
    if not methods_raw:
        raise ConfigError("methods 至少需要定义一个 DeepMD 方法")
    methods: dict[str, DeepMDMethod] = {}
    for raw_name, raw_method in methods_raw.items():
        try:
            method_name = safe_name(str(raw_name), what="method name")
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        item = _mapping(raw_method, f"methods.{method_name}")
        _strict(item, {"type", "model", "device", "head", "enabled"}, f"methods.{method_name}")
        if item.get("type") != "deepmd":
            raise ConfigError(f"methods.{method_name}.type 三阶首版仅支持 deepmd")
        if "model" not in item:
            raise ConfigError(f"methods.{method_name}.model 为必填项")
        model_spec = str(item["model"])
        model = builtin_model_path() if model_spec == "builtin:dpa4" else _resolve(base, model_spec)
        head_value = item.get("head")
        method = DeepMDMethod(
            method_name, "deepmd", model, model_spec, str(item.get("device", "cuda:0")),
            None if head_value is None else str(head_value), bool(item.get("enabled", True)),
        )
        if model.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
            raise ConfigError(f"methods.{method_name}.model 不支持扩展名 {model.suffix}")
        if model.suffix.lower() == ".pt2" and method.head:
            raise ConfigError(f"methods.{method_name}: .pt2 已固化 head，不能再设置 head")
        methods[method_name] = method
    if not any(method.enabled for method in methods.values()):
        raise ConfigError("至少需要启用一个 DeepMD 方法")

    config = AnhConfig(config_path, 1, project, structure, anh, methods)
    if require_inputs:
        validate_anh_input_paths(config)
    return config


def validate_anh_input_paths(config: AnhConfig) -> None:
    if not config.structure.file.is_file():
        raise ConfigError(f"结构文件不存在: {config.structure.file}")
    if config.structure.born_file is not None and not config.structure.born_file.is_file():
        raise ConfigError(f"BORN 文件不存在: {config.structure.born_file}")
    for name, method in config.enabled_methods.items():
        if not method.model.is_file():
            raise ConfigError(f"DeepMD 模型不存在 ({name}): {method.model}")

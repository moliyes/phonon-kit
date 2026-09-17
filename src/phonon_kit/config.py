from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from .errors import ConfigError
from .util import safe_name, sha256_file, sha256_json
from .vasp_input import missing_vasp_template_inputs


BUILTIN_MODEL_NAME = "dpa4_omat_neo704.pt2"
SUPPORTED_MODEL_SUFFIXES = {".pt2", ".pt", ".pth", ".pb"}


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    runs_dir: Path


@dataclass(frozen=True)
class StructureConfig:
    file: Path


@dataclass(frozen=True)
class RelaxationConfig:
    enabled: bool = False
    method: str = "dpa4"
    variable_cell: bool = True
    pressure_gpa: float = 0.0
    fmax_ev_angstrom: float = 0.01
    max_steps: int = 1000
    trajectory_interval: int = 10


@dataclass(frozen=True)
class BandConfig:
    path: Literal["auto"] | tuple[tuple[float, float, float], ...] = "auto"
    points_per_segment: int = 101
    labels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ThermalConfig:
    temperature_min_k: float = 0.0
    temperature_max_k: float = 1000.0
    temperature_step_k: float = 10.0
    imaginary_policy: Literal["exclude"] = "exclude"
    significant_imaginary_thz: float = -0.1


@dataclass(frozen=True)
class UnfoldingConfig:
    reference_supercell: Path
    supercell_matrix: tuple[tuple[int, int, int], ...]
    mapping_tolerance_angstrom: float = 0.2


@dataclass(frozen=True)
class PhononConfig:
    supercell: tuple[int, int, int] = (2, 2, 2)
    displacement_angstrom: float = 0.01
    displacement_plusminus: Literal["auto"] | bool = "auto"
    displacement_diagonal: bool = True
    primitive: Literal["auto", "P"] = "auto"
    symmetry_tolerance: float = 1.0e-5
    band: BandConfig = BandConfig()
    mesh: tuple[int, int, int] = (30, 30, 30)
    thermal: ThermalConfig = ThermalConfig()
    unfolding: UnfoldingConfig | None = None


@dataclass(frozen=True)
class DeepMDMethod:
    name: str
    type: Literal["deepmd"]
    model: Path
    model_spec: str
    device: str = "cuda:0"
    head: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class DispatcherConfig:
    type: Literal["dpdispatcher"]
    machine: Path
    resources: Path
    command: str
    clean_remote_after_success: bool = False


@dataclass(frozen=True)
class VaspMethod:
    name: str
    type: Literal["vasp"]
    template_dir: Path
    executor: DispatcherConfig
    enabled: bool = True


MethodConfig = DeepMDMethod | VaspMethod


@dataclass(frozen=True)
class Config:
    path: Path
    schema_version: int
    project: ProjectConfig
    structure: StructureConfig
    relaxation: RelaxationConfig
    phonon: PhononConfig
    methods: dict[str, MethodConfig]

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    @property
    def enabled_methods(self) -> dict[str, MethodConfig]:
        return {name: method for name, method in self.methods.items() if method.enabled}

    def resolved_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

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
                    "template_dir": str(method.template_dir),
                    "enabled": method.enabled,
                    "executor": {
                        "type": "dpdispatcher",
                        "machine": str(method.executor.machine),
                        "resources": str(method.executor.resources),
                        "command": method.executor.command,
                        "clean_remote_after_success": method.executor.clean_remote_after_success,
                    },
                }
        data = {
            "schema_version": self.schema_version,
            "project": {"name": self.project.name, "runs_dir": str(self.project.runs_dir)},
            "structure": {"file": str(self.structure.file)},
            "relaxation": asdict(self.relaxation),
            "phonon": asdict(self.phonon),
            "methods": methods,
        }
        return convert(data)

    def fingerprint(self) -> str:
        payload = self.resolved_dict()
        files: dict[str, str] = {"structure": sha256_file(self.structure.file)}
        if self.phonon.unfolding is not None:
            files["unfolding:reference_supercell"] = sha256_file(
                self.phonon.unfolding.reference_supercell
            )
        for name, method in self.enabled_methods.items():
            if isinstance(method, DeepMDMethod):
                files[f"method:{name}:model"] = sha256_file(method.model)
            else:
                for filename in ("INCAR", "KPOINTS", "POTCAR"):
                    path = method.template_dir / filename
                    if path.exists():
                        files[f"method:{name}:template:{filename}"] = sha256_file(path)
                for label, path in (("machine", method.executor.machine), ("resources", method.executor.resources)):
                    if path.exists():
                        files[f"method:{name}:{label}"] = sha256_file(path)
        payload["input_sha256"] = files
        return sha256_json(payload)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def builtin_model_path() -> Path:
    source_model = project_root() / "models" / BUILTIN_MODEL_NAME
    if source_model.is_file():
        return source_model
    packaged_model = Path(__file__).resolve().parent / "data" / "models" / BUILTIN_MODEL_NAME
    return packaged_model


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


def _integer_matrix3(value: Any, where: str) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigError(f"{where} 必须是 3x3 整数矩阵")
    rows: list[tuple[int, int, int]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ConfigError(f"{where} 必须是 3x3 整数矩阵")
        parsed: list[int] = []
        for item in row:
            if isinstance(item, bool):
                raise ConfigError(f"{where} 必须是 3x3 整数矩阵")
            try:
                integer = int(item)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{where} 必须是 3x3 整数矩阵") from exc
            if isinstance(item, float) and not item.is_integer():
                raise ConfigError(f"{where} 必须是 3x3 整数矩阵")
            parsed.append(integer)
        rows.append(tuple(parsed))  # type: ignore[arg-type]
    determinant = round(
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if determinant <= 0:
        raise ConfigError(f"{where} 的行列式必须为正")
    return tuple(rows)


def _band_path(value: Any) -> Literal["auto"] | tuple[tuple[float, float, float], ...]:
    if value == "auto":
        return "auto"
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ConfigError("phonon.band.path 必须为 auto，或至少两个三维 q 点")
    points: list[tuple[float, float, float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            raise ConfigError(f"phonon.band.path[{index}] 必须恰好包含三个数")
        try:
            parsed = tuple(float(component) for component in point)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"phonon.band.path[{index}] 必须恰好包含三个数") from exc
        if not all(math.isfinite(component) for component in parsed):
            raise ConfigError(f"phonon.band.path[{index}] 含非有限数值")
        points.append(parsed)  # type: ignore[arg-type]
    return tuple(points)


def load_config(path: str | Path, *, require_inputs: bool = True) -> Config:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"配置文件不存在: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "配置根节点")
    _strict(root, {"schema_version", "project", "structure", "relaxation", "phonon", "methods"}, "配置根节点")
    if root.get("schema_version") != 1:
        raise ConfigError("schema_version 必须为 1")
    base = config_path.parent

    project_raw = _mapping(root.get("project"), "project")
    _strict(project_raw, {"name", "runs_dir"}, "project")
    try:
        project_name = safe_name(str(project_raw["name"]), what="project.name")
    except (KeyError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    project = ProjectConfig(project_name, _resolve(base, project_raw.get("runs_dir", "runs")))

    structure_raw = _mapping(root.get("structure"), "structure")
    _strict(structure_raw, {"file"}, "structure")
    if "file" not in structure_raw:
        raise ConfigError("structure.file 为必填项")
    structure = StructureConfig(_resolve(base, structure_raw["file"]))

    relax_raw = _mapping(root.get("relaxation", {}), "relaxation")
    _strict(relax_raw, {"enabled", "method", "variable_cell", "pressure_gpa", "fmax_ev_angstrom", "max_steps", "trajectory_interval"}, "relaxation")
    relaxation = RelaxationConfig(
        enabled=bool(relax_raw.get("enabled", False)),
        method=str(relax_raw.get("method", "dpa4")),
        variable_cell=bool(relax_raw.get("variable_cell", True)),
        pressure_gpa=float(relax_raw.get("pressure_gpa", 0.0)),
        fmax_ev_angstrom=float(relax_raw.get("fmax_ev_angstrom", 0.01)),
        max_steps=int(relax_raw.get("max_steps", 1000)),
        trajectory_interval=int(relax_raw.get("trajectory_interval", 10)),
    )
    if relaxation.fmax_ev_angstrom <= 0 or relaxation.max_steps <= 0 or relaxation.trajectory_interval <= 0:
        raise ConfigError("relaxation 的 fmax、max_steps 和 trajectory_interval 必须为正数")

    phonon_raw = _mapping(root.get("phonon", {}), "phonon")
    _strict(
        phonon_raw,
        {
            "supercell", "displacement_angstrom", "displacement_plusminus",
            "displacement_diagonal", "primitive", "symmetry_tolerance", "band",
            "mesh", "thermal", "unfolding",
        },
        "phonon",
    )
    band_raw = _mapping(phonon_raw.get("band", {}), "phonon.band")
    _strict(band_raw, {"path", "points_per_segment", "labels"}, "phonon.band")
    path_value = _band_path(band_raw.get("path", "auto"))
    labels_value = band_raw.get("labels")
    labels: tuple[str, ...] | None = None
    if labels_value is not None:
        if not isinstance(labels_value, (list, tuple)):
            raise ConfigError("phonon.band.labels 必须是字符串列表")
        labels = tuple(str(item) for item in labels_value)
    band = BandConfig(
        path=path_value,
        points_per_segment=int(band_raw.get("points_per_segment", 101)),
        labels=labels,
    )
    if band.points_per_segment < 2:
        raise ConfigError("phonon.band.points_per_segment 必须 >= 2")
    if band.path == "auto" and band.labels is not None:
        raise ConfigError("phonon.band.path 为 auto 时不能手动设置 labels")
    if band.path != "auto" and band.labels is not None and len(band.labels) != len(band.path):
        raise ConfigError("显式 phonon.band.labels 数量必须与 path 的 q 点数量一致")
    thermal_raw = _mapping(phonon_raw.get("thermal", {}), "phonon.thermal")
    _strict(thermal_raw, {"temperature_min_k", "temperature_max_k", "temperature_step_k", "imaginary_policy", "significant_imaginary_thz"}, "phonon.thermal")
    thermal = ThermalConfig(
        temperature_min_k=float(thermal_raw.get("temperature_min_k", 0.0)),
        temperature_max_k=float(thermal_raw.get("temperature_max_k", 1000.0)),
        temperature_step_k=float(thermal_raw.get("temperature_step_k", 10.0)),
        imaginary_policy=str(thermal_raw.get("imaginary_policy", "exclude")),  # type: ignore[arg-type]
        significant_imaginary_thz=float(thermal_raw.get("significant_imaginary_thz", -0.1)),
    )
    if thermal.imaginary_policy != "exclude":
        raise ConfigError("phonon.thermal.imaginary_policy 当前仅支持 exclude")
    if thermal.temperature_min_k < 0 or thermal.temperature_max_k < thermal.temperature_min_k or thermal.temperature_step_k <= 0:
        raise ConfigError("thermal 温度范围无效")
    if thermal.significant_imaginary_thz > 0:
        raise ConfigError("significant_imaginary_thz 应为 0 或负数")
    unfolding: UnfoldingConfig | None = None
    if "unfolding" in phonon_raw and phonon_raw["unfolding"] is not None:
        unfolding_raw = _mapping(phonon_raw["unfolding"], "phonon.unfolding")
        _strict(
            unfolding_raw,
            {"reference_supercell", "supercell_matrix", "mapping_tolerance_angstrom"},
            "phonon.unfolding",
        )
        for required in ("reference_supercell", "supercell_matrix"):
            if required not in unfolding_raw:
                raise ConfigError(f"phonon.unfolding.{required} 为必填项")
        tolerance = float(unfolding_raw.get("mapping_tolerance_angstrom", 0.2))
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ConfigError("phonon.unfolding.mapping_tolerance_angstrom 必须为正数")
        unfolding = UnfoldingConfig(
            reference_supercell=_resolve(base, unfolding_raw["reference_supercell"]),
            supercell_matrix=_integer_matrix3(
                unfolding_raw["supercell_matrix"], "phonon.unfolding.supercell_matrix"
            ),
            mapping_tolerance_angstrom=tolerance,
        )
        if band.path == "auto":
            raise ConfigError("启用 phonon.unfolding 时必须显式设置 phonon.band.path")
    plusminus_value = phonon_raw.get("displacement_plusminus", "auto")
    if plusminus_value != "auto" and not isinstance(plusminus_value, bool):
        raise ConfigError("phonon.displacement_plusminus 必须为 auto、true 或 false")
    diagonal_value = phonon_raw.get("displacement_diagonal", True)
    if not isinstance(diagonal_value, bool):
        raise ConfigError("phonon.displacement_diagonal 必须为 true 或 false")
    primitive_value = str(phonon_raw.get("primitive", "auto"))
    if primitive_value not in {"auto", "P"}:
        raise ConfigError("phonon.primitive 必须为 auto 或 P")
    phonon = PhononConfig(
        supercell=_triple(phonon_raw.get("supercell", [2, 2, 2]), "phonon.supercell"),
        displacement_angstrom=float(phonon_raw.get("displacement_angstrom", 0.01)),
        displacement_plusminus=plusminus_value,  # type: ignore[arg-type]
        displacement_diagonal=diagonal_value,
        primitive=primitive_value,  # type: ignore[arg-type]
        symmetry_tolerance=float(phonon_raw.get("symmetry_tolerance", 1.0e-5)),
        band=band,
        mesh=_triple(phonon_raw.get("mesh", [30, 30, 30]), "phonon.mesh"),
        thermal=thermal,
        unfolding=unfolding,
    )
    if phonon.displacement_angstrom <= 0 or phonon.symmetry_tolerance <= 0:
        raise ConfigError("displacement_angstrom 和 symmetry_tolerance 必须为正")

    methods_raw = _mapping(root.get("methods"), "methods")
    if not methods_raw:
        raise ConfigError("methods 至少需要定义一个方法")
    methods: dict[str, MethodConfig] = {}
    for name, value in methods_raw.items():
        try:
            safe_name(str(name), what="method name")
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
            head = item.get("head")
            method = DeepMDMethod(str(name), "deepmd", model, model_spec, str(item.get("device", "cuda:0")), None if head is None else str(head), bool(item.get("enabled", True)))
            if model.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
                raise ConfigError(f"methods.{name}.model 不支持扩展名 {model.suffix}")
            if model.suffix.lower() == ".pt2" and method.head:
                raise ConfigError(f"methods.{name}: .pt2 已固化 head，不能再设置 head")
            methods[str(name)] = method
        elif kind == "vasp":
            _strict(item, {"type", "template_dir", "executor", "enabled"}, f"methods.{name}")
            if "template_dir" not in item or "executor" not in item:
                raise ConfigError(f"methods.{name} 需要 template_dir 和 executor")
            executor_raw = _mapping(item["executor"], f"methods.{name}.executor")
            _strict(executor_raw, {"type", "machine", "resources", "command", "clean_remote_after_success"}, f"methods.{name}.executor")
            if executor_raw.get("type") != "dpdispatcher":
                raise ConfigError(f"methods.{name}.executor.type 当前仅支持 dpdispatcher")
            for required in ("machine", "resources", "command"):
                if required not in executor_raw:
                    raise ConfigError(f"methods.{name}.executor.{required} 为必填项")
            executor = DispatcherConfig(
                "dpdispatcher",
                _resolve(base, executor_raw["machine"]),
                _resolve(base, executor_raw["resources"]),
                str(executor_raw["command"]),
                bool(executor_raw.get("clean_remote_after_success", False)),
            )
            methods[str(name)] = VaspMethod(str(name), "vasp", _resolve(base, item["template_dir"]), executor, bool(item.get("enabled", True)))
        else:
            raise ConfigError(f"methods.{name}.type 必须为 deepmd 或 vasp")

    enabled = {name: method for name, method in methods.items() if method.enabled}
    if not enabled:
        raise ConfigError("至少需要启用一个方法")
    if sum(isinstance(method, VaspMethod) for method in enabled.values()) > 1:
        raise ConfigError("初版最多允许一个启用的 VASP 方法")
    if relaxation.enabled:
        method = methods.get(relaxation.method)
        if not isinstance(method, DeepMDMethod) or not method.enabled:
            raise ConfigError("relaxation.method 必须引用一个启用的 DeepMD 方法")

    config = Config(config_path, 1, project, structure, relaxation, phonon, methods)
    if require_inputs:
        validate_input_paths(config)
    return config


def validate_input_paths(config: Config) -> None:
    if not config.structure.file.is_file():
        raise ConfigError(f"结构文件不存在: {config.structure.file}")
    if config.phonon.unfolding is not None:
        reference = config.phonon.unfolding.reference_supercell
        if not reference.is_file():
            raise ConfigError(f"unfolding 参考超胞不存在: {reference}")
    for name, method in config.enabled_methods.items():
        if isinstance(method, DeepMDMethod):
            if not method.model.is_file():
                raise ConfigError(f"DeepMD 模型不存在 ({name}): {method.model}")
        else:
            if not method.template_dir.is_dir():
                raise ConfigError(f"VASP 模板目录不存在 ({name}): {method.template_dir}")
            missing = missing_vasp_template_inputs(method.template_dir)
            if missing:
                raise ConfigError(f"VASP 模板缺少 ({name}): {', '.join(missing)}")
            for label, path in (("machine", method.executor.machine), ("resources", method.executor.resources)):
                if not path.is_file():
                    raise ConfigError(f"DPDispatcher {label} 配置不存在 ({name}): {path}")
            if not method.executor.command.strip():
                raise ConfigError(f"methods.{name}.executor.command 不能为空")

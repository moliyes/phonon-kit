from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

from .config import Config, DeepMDMethod, VaspMethod
from .dispatcher import validate_dispatcher_files
from .providers.deepmd import validate_method
from .providers.vasp import validate_vasp_template
from .structure import read_structure


def validate_runtime(config: Config, methods: list[str] | None = None, *, real_inference: bool = True, logdir: Path | None = None) -> dict[str, Any]:
    selected = methods or list(config.enabled_methods)
    for name in selected:
        if name not in config.enabled_methods:
            raise ValueError(f"方法未定义或未启用: {name}")
    versions: dict[str, str] = {}
    for distribution in ("phonopy", "ase", "numpy", "seekpath", "spglib", "matplotlib", "dpdispatcher"):
        versions[distribution] = importlib.metadata.version(distribution)
    atoms = read_structure(config.structure.file)
    report: dict[str, Any] = {
        "config": str(config.path),
        "config_hash": config.fingerprint(),
        "versions": versions,
        "structure": {
            "formula": atoms.get_chemical_formula(mode="hill"),
            "n_atoms": len(atoms),
            "volume_angstrom3": float(atoms.get_volume()),
        },
        "methods": {},
        "warnings": [],
    }
    if logdir is None:
        base = config.base_dir / ".phonon-kit" / "validation"
        workdir = base / "work"
        logdir = base / "logs"
    else:
        workdir = logdir.parent / "work" / "validation"
    for name in selected:
        method = config.methods[name]
        if isinstance(method, DeepMDMethod):
            info = {
                "type": "deepmd",
                "model": str(method.model),
                "device": method.device,
                "head": method.head,
            }
            if real_inference:
                info["inference"] = validate_method(method, config.structure.file, workdir, logdir)
            report["methods"][name] = info
        elif isinstance(method, VaspMethod):
            warnings = validate_vasp_template(method)
            from ase.io.vasp import get_atomtypes

            potcar_symbols = get_atomtypes(str(method.template_dir / "POTCAR"))
            structure_symbols: list[str] = []
            for symbol in atoms.get_chemical_symbols():
                if symbol not in structure_symbols:
                    structure_symbols.append(symbol)
            if potcar_symbols and potcar_symbols != structure_symbols:
                raise ValueError(
                    f"{name}: POTCAR 元素顺序 {potcar_symbols} 与结构顺序 {structure_symbols} 不一致"
                )
            if not potcar_symbols:
                warnings.append("无法从 POTCAR 的 TITEL 记录识别元素顺序，请人工确认")
            report["warnings"].extend(f"{name}: {warning}" for warning in warnings)
            report["methods"][name] = {
                "type": "vasp",
                "template_dir": str(method.template_dir),
                "potcar_symbols": potcar_symbols,
                "dispatcher": validate_dispatcher_files(method),
            }
    return report

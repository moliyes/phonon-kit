from __future__ import annotations

import math
from pathlib import Path


def parse_incar(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.split("!", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.split(";"):
            if "=" in part:
                key, value = part.split("=", 1)
                values[key.strip().upper()] = value.strip()
    return values


def has_vasp_kpoint_source(template_dir: Path) -> bool:
    """Return whether VASP can obtain a k-point mesh from the template."""
    if (template_dir / "KPOINTS").is_file():
        return True
    incar_path = template_dir / "INCAR"
    if not incar_path.is_file():
        return False
    raw_value = parse_incar(incar_path).get("KSPACING")
    if raw_value is None:
        return False
    try:
        value = float(raw_value.split()[0])
    except (IndexError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def missing_vasp_template_inputs(template_dir: Path) -> list[str]:
    missing = [
        filename
        for filename in ("INCAR", "POTCAR")
        if not (template_dir / filename).is_file()
    ]
    if not has_vasp_kpoint_source(template_dir):
        missing.append("KPOINTS（或 INCAR 中的正数 KSPACING）")
    return missing

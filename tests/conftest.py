from __future__ import annotations

from pathlib import Path

import pytest


POSCAR = """Si
1.0
5.43 0 0
0 5.43 0
0 0 5.43
Si
1
Direct
0 0 0
"""


def write_config(root: Path, *, model: str = "model.pth", extra: str = "") -> Path:
    (root / "POSCAR").write_text(POSCAR, encoding="utf-8")
    if model != "builtin:dpa4":
        (root / model).write_bytes(b"dummy-model")
    text = f"""schema_version: 1
project:
  name: test
  runs_dir: runs
structure:
  file: POSCAR
relaxation:
  enabled: false
phonon:
  supercell: [1, 1, 1]
  displacement_angstrom: 0.01
  primitive: auto
  symmetry_tolerance: 1.0e-5
  band:
    path: auto
    points_per_segment: 11
  mesh: [5, 5, 5]
  thermal:
    temperature_min_k: 0
    temperature_max_k: 100
    temperature_step_k: 10
    imaginary_policy: exclude
    significant_imaginary_thz: -0.1
methods:
  dp:
    type: deepmd
    model: {model}
    device: cpu
{extra}
"""
    path = root / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return write_config(tmp_path)


from __future__ import annotations

from pathlib import Path

from .config import builtin_model_path
from .errors import RunStateError
from .initializer import POSCAR


ANH_CONFIG = """schema_version: 1

project:
  name: sio2_anh
  runs_dir: runs

structure:
  file: inputs/POSCAR
  born_file: null

anharmonic:
  supercell: [2, 2, 2]
  fc2_supercell: null
  displacement_angstrom: 0.03
  primitive: auto
  symmetry_tolerance: 1.0e-5
  subtract_residual_forces: false
  mesh: [11, 11, 11]
  temperature_min_k: 100
  temperature_max_k: 1000
  temperature_step_k: 100
  lifetime_temperature_k: 300
  significant_imaginary_thz: -0.1
  continue_on_imaginary: true

methods:
  pretrained:
    type: deepmd
    model: builtin:dpa4
    device: cuda:0
"""


def init_anh_case(target: Path) -> Path:
    target = target.expanduser().resolve()
    if target.exists() and not target.is_dir():
        raise RunStateError(f"目标路径不是目录: {target}")
    if target.exists() and any(target.iterdir()):
        raise RunStateError(f"目标目录非空，拒绝覆盖: {target}")
    (target / "inputs" / "models").mkdir(parents=True, exist_ok=True)
    (target / "anh.yaml").write_text(ANH_CONFIG, encoding="utf-8")
    (target / "inputs" / "POSCAR").write_text(POSCAR, encoding="utf-8")
    (target / "inputs" / "models" / ".gitkeep").write_text("", encoding="utf-8")
    (target / ".gitignore").write_text("runs/\ninputs/models/*\n!inputs/models/.gitkeep\n", encoding="utf-8")
    (target / "README.txt").write_text(
        f"默认 DPA4: {builtin_model_path()}\n先运行: ph anh plan anh.yaml\n再运行: ph anh run anh.yaml\n",
        encoding="utf-8",
    )
    return target

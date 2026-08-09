from __future__ import annotations

from pathlib import Path

from .config import builtin_model_path
from .errors import RunStateError


CONFIG = """schema_version: 1

project:
  name: sio2
  runs_dir: runs

structure:
  file: inputs/POSCAR

relaxation:
  enabled: false
  method: dpa4
  variable_cell: true
  pressure_gpa: 0.0
  fmax_ev_angstrom: 0.01
  max_steps: 1000
  trajectory_interval: 10

phonon:
  supercell: [2, 2, 2]
  displacement_angstrom: 0.01
  primitive: auto
  symmetry_tolerance: 1.0e-5
  band:
    path: auto
    points_per_segment: 101
  mesh: [30, 30, 30]
  thermal:
    temperature_min_k: 0
    temperature_max_k: 1000
    temperature_step_k: 10
    imaginary_policy: exclude
    significant_imaginary_thz: -0.1

methods:
  dpa4:
    type: deepmd
    model: builtin:dpa4
    device: cuda:0
"""


POSCAR = """SiO2 P42/mnm example
1.0
4.17333286205 -0.00000024701 -0.00000014033
0.00000006807 4.17333325777 -0.00000007569
0.00000004364 0.00000002969 2.67532823516
O Si
4 2
Direct
0.69466857 0.80533503 0.24999999
0.30533073 0.19467294 0.24999999
0.80533118 0.30533501 0.74999995
0.19466904 0.69467291 0.75000008
0.00000449 0.49999202 0.25000001
0.49999591 0.99999216 0.74999996
"""


INCAR = """SYSTEM = phonopy displaced supercell
PREC = Accurate
IBRION = -1
NSW = 0
ENCUT = 520
EDIFF = 1E-8
ISMEAR = 0
SIGMA = 0.01
LREAL = .FALSE.
LWAVE = .FALSE.
LCHARG = .FALSE.
"""


KPOINTS = """Automatic mesh
0
Gamma
2 2 2
0 0 0
"""


MACHINE_EXAMPLE = """{
  "batch_type": "Bohrium",
  "context_type": "Bohrium",
  "local_root": "./",
  "remote_profile": {
    "email": "YOUR_EMAIL",
    "password": "YOUR_PASSWORD",
    "program_id": 0,
    "input_data": {"job_name": "phonon-kit-vasp"}
  }
}
"""


RESOURCES_EXAMPLE = """{
  "number_node": 1,
  "cpu_per_node": 16,
  "gpu_per_node": 0,
  "group_size": 1,
  "wait_time": 10
}
"""


def init_case(target: Path) -> Path:
    target = target.expanduser().resolve()
    if target.exists() and not target.is_dir():
        raise RunStateError(f"目标路径不是目录: {target}")
    if target.exists() and any(target.iterdir()):
        raise RunStateError(f"目标目录非空，拒绝覆盖: {target}")
    (target / "inputs" / "models").mkdir(parents=True, exist_ok=True)
    (target / "inputs" / "vasp").mkdir(parents=True, exist_ok=True)
    (target / "inputs" / "dispatcher").mkdir(parents=True, exist_ok=True)
    (target / "config.yaml").write_text(CONFIG, encoding="utf-8")
    (target / "inputs" / "POSCAR").write_text(POSCAR, encoding="utf-8")
    (target / "inputs" / "models" / ".gitkeep").write_text("", encoding="utf-8")
    (target / "inputs" / "vasp" / "INCAR").write_text(INCAR, encoding="utf-8")
    (target / "inputs" / "vasp" / "KPOINTS").write_text(KPOINTS, encoding="utf-8")
    (target / "inputs" / "vasp" / "README.md").write_text(
        "把与元素顺序一致的 POTCAR 放在这里；软件不会生成或分发 POTCAR。\n",
        encoding="utf-8",
    )
    (target / "inputs" / "dispatcher" / "machine.json.example").write_text(MACHINE_EXAMPLE, encoding="utf-8")
    (target / "inputs" / "dispatcher" / "resources.json.example").write_text(RESOURCES_EXAMPLE, encoding="utf-8")
    (target / ".gitignore").write_text(".phonon-kit/\nruns/\ninputs/models/*\n!inputs/models/.gitkeep\ninputs/vasp/POTCAR\ninputs/dispatcher/machine.json\n", encoding="utf-8")
    (target / "README.txt").write_text(
        f"默认 DPA4: {builtin_model_path()}\n运行: ph validate config.yaml && ph run config.yaml\n",
        encoding="utf-8",
    )
    return target

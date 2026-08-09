from __future__ import annotations

from pathlib import Path

import yaml

from .errors import RunStateError
from .initializer import KPOINTS, MACHINE_EXAMPLE, POSCAR, RESOURCES_EXAMPLE
from .util import safe_name


VOLUME_RELAX_INCAR = """SYSTEM = QHA constrained-volume relaxation
PREC = Accurate
IBRION = 2
ISIF = 4
NSW = 200
ENCUT = 520
EDIFF = 1E-8
EDIFFG = -1E-3
ISMEAR = 0
SIGMA = 0.01
LREAL = .FALSE.
LWAVE = .FALSE.
LCHARG = .FALSE.
"""


STATIC_INCAR = """SYSTEM = QHA static energy
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


PHONON_INCAR = """SYSTEM = QHA finite-displacement force
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


def init_qha_case(target: Path, phase_names: list[str]) -> Path:
    if len(phase_names) < 2:
        raise RunStateError("ph qha init 至少需要两个 --phases 名称")
    try:
        names = [safe_name(name, what="phase name") for name in phase_names]
    except ValueError as exc:
        raise RunStateError(str(exc)) from exc
    if len(names) != len(set(names)):
        raise RunStateError("--phases 中的名称不能重复")

    target = target.expanduser().resolve()
    try:
        project_name = safe_name(target.name, what="project.name")
    except ValueError as exc:
        raise RunStateError(str(exc)) from exc
    if target.exists() and not target.is_dir():
        raise RunStateError(f"目标路径不是目录: {target}")
    if target.exists() and any(target.iterdir()):
        raise RunStateError(f"目标目录非空，拒绝覆盖: {target}")

    phase_root = target / "inputs" / "phases"
    for name in names:
        directory = phase_root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "POSCAR").write_text(POSCAR, encoding="utf-8")
        (directory / "README.txt").write_text(
            f"请把 {name} 的参考结构写入 POSCAR；当前文件只是可解析的 SiO2 占位结构。\n",
            encoding="utf-8",
        )

    for stage, incar in (
        ("volume_relax", VOLUME_RELAX_INCAR),
        ("static", STATIC_INCAR),
        ("phonon", PHONON_INCAR),
    ):
        directory = target / "inputs" / "vasp" / stage
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "INCAR").write_text(incar, encoding="utf-8")
        (directory / "KPOINTS").write_text(KPOINTS, encoding="utf-8")
        (directory / "README.md").write_text(
            "把与 POSCAR 元素顺序一致的 POTCAR 放在这里。phonon-kit 不生成或分发 POTCAR。\n",
            encoding="utf-8",
        )

    dispatcher = target / "inputs" / "dispatcher"
    dispatcher.mkdir(parents=True, exist_ok=True)
    (dispatcher / "machine.json.example").write_text(MACHINE_EXAMPLE, encoding="utf-8")
    (dispatcher / "resources.json.example").write_text(RESOURCES_EXAMPLE, encoding="utf-8")
    models = target / "inputs" / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / ".gitkeep").write_text("", encoding="utf-8")

    phases = {
        name: {
            "structure": f"inputs/phases/{name}/POSCAR",
            "label": name,
            "phonon": {"supercell": [2, 2, 2]},
        }
        for name in names
    }
    config = {
        "schema_version": 1,
        "project": {"name": project_name, "runs_dir": "runs"},
        "phases": phases,
        "volume_grid": {"ratios": [0.88, 0.91, 0.94, 0.97, 1.0, 1.03, 1.06]},
        "volume_relaxation": {
            "fmax_ev_angstrom": 0.01,
            "max_deviatoric_stress_gpa": 0.1,
            "max_steps": 1000,
            "trajectory_interval": 10,
            "volume_tolerance_relative": 1.0e-5,
        },
        "phonon": {
            "supercell": [2, 2, 2],
            "displacement_angstrom": 0.01,
            "symmetry_tolerance": 1.0e-5,
            "mesh": [30, 30, 30],
            "significant_imaginary_thz": -0.1,
        },
        "qha": {
            "eos": "vinet",
            "temperature": {"min_k": 0, "max_k": 1000, "step_k": 10},
            "pressure": {"min_gpa": 0, "max_gpa": 20, "step_gpa": 0.25},
            "imaginary_policy": "exclude",
        },
        "methods": {
            "dpa4": {"type": "deepmd", "model": "builtin:dpa4", "device": "cuda:0", "enabled": True},
            "dft": {
                "type": "vasp",
                "enabled": False,
                "templates": {
                    "volume_relax": "inputs/vasp/volume_relax",
                    "static": "inputs/vasp/static",
                    "phonon": "inputs/vasp/phonon",
                },
                "executor": {
                    "type": "dpdispatcher",
                    "machine": "inputs/dispatcher/machine.json",
                    "resources": "inputs/dispatcher/resources.json",
                    "command": "mpirun -n 16 vasp_std",
                    "clean_remote_after_success": False,
                },
            },
        },
    }
    (target / "qha.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (target / ".gitignore").write_text(
        "runs/\ninputs/models/*\n!inputs/models/.gitkeep\n**/POTCAR\ninputs/dispatcher/machine.json\n",
        encoding="utf-8",
    )
    (target / "README.txt").write_text(
        "先替换各相 POSCAR，然后运行 ph qha plan qha.yaml 和 ph qha validate qha.yaml。\n",
        encoding="utf-8",
    )
    return target

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DeepMDMethod
from .providers.deepmd import run_worker
from .qha_config import VolumeRelaxationConfig


def relax_qha_volume(
    method: DeepMDMethod,
    structure: Path,
    workdir: Path,
    logdir: Path,
    relaxation: VolumeRelaxationConfig,
    target_volume: float,
) -> dict[str, Any]:
    payload = {
        "structure": str(structure),
        "model": str(method.model),
        "head": method.head,
        "device": method.device,
        "workdir": str(workdir),
        "target_volume_angstrom3": target_volume,
        "fmax": relaxation.fmax_ev_angstrom,
        "max_deviatoric_stress_gpa": relaxation.max_deviatoric_stress_gpa,
        "max_steps": relaxation.max_steps,
        "trajectory_interval": relaxation.trajectory_interval,
        "volume_tolerance_relative": relaxation.volume_tolerance_relative,
    }
    return run_worker(
        "qha-relax",
        payload,
        workdir / "worker-payload.json",
        logdir / f"qha-relax-{method.name}.log",
    )


def evaluate_qha_static(
    method: DeepMDMethod,
    structure: Path,
    workdir: Path,
    logdir: Path,
) -> dict[str, Any]:
    payload = {
        "structure": str(structure),
        "model": str(method.model),
        "head": method.head,
        "device": method.device,
    }
    return run_worker(
        "validate",
        payload,
        workdir / "worker-payload.json",
        logdir / f"qha-static-{method.name}.log",
    )

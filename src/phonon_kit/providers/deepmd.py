from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import DeepMDMethod
from ..errors import ExternalProgramError, IncompleteResultsError
from ..util import atomic_write_json, load_json


def device_environment(device: str) -> dict[str, str]:
    env = os.environ.copy()
    if device == "cpu":
        env["DEVICE"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.pop("LOCAL_RANK", None)
    elif device.startswith("cuda:"):
        gpu = device.split(":", 1)[1]
        if not gpu.isdigit():
            raise ExternalProgramError(f"无效 DeepMD device: {device}")
        env.pop("DEVICE", None)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["LOCAL_RANK"] = "0"
    else:
        raise ExternalProgramError(f"device 必须为 cpu 或 cuda:N: {device}")
    return env


def run_worker(operation: str, payload: dict, payload_path: Path, log_path: Path) -> dict:
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result_path = payload_path.with_name(payload_path.stem + "-result.json")
    payload = dict(payload)
    payload["result_file"] = str(result_path)
    atomic_write_json(payload_path, payload)
    env = device_environment(str(payload["device"]))
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            [sys.executable, "-m", "phonon_kit.deepmd_worker", operation, str(payload_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    if process.returncode != 0 or not result_path.is_file():
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        except OSError:
            pass
        raise ExternalProgramError(
            f"DeepMD {operation} 失败，日志: {log_path}\n{tail}"
        )
    return load_json(result_path)


def validate_method(method: DeepMDMethod, structure: Path, workdir: Path, logdir: Path) -> dict:
    payload = {
        "structure": str(structure),
        "model": str(method.model),
        "head": method.head,
        "device": method.device,
    }
    return run_worker("validate", payload, workdir / f"validate-{method.name}.json", logdir / f"validate-{method.name}.log")


def relax_structure(method: DeepMDMethod, structure: Path, workdir: Path, logdir: Path, *, variable_cell: bool, pressure_gpa: float, fmax: float, max_steps: int, trajectory_interval: int) -> dict:
    payload = {
        "structure": str(structure),
        "model": str(method.model),
        "head": method.head,
        "device": method.device,
        "workdir": str(workdir),
        "variable_cell": variable_cell,
        "pressure_gpa": pressure_gpa,
        "fmax": fmax,
        "max_steps": max_steps,
        "trajectory_interval": trajectory_interval,
    }
    return run_worker("relax", payload, workdir / "worker-payload.json", logdir / "relaxation.log")


@dataclass
class DeepMDProvider:
    method: DeepMDMethod
    phonopy_yaml: Path
    workdir: Path
    resultdir: Path
    logdir: Path

    @property
    def checkpoint_dir(self) -> Path:
        return self.workdir / "checkpoints"

    def prepare(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.resultdir.mkdir(parents=True, exist_ok=True)

    def run_or_submit(self, *, wait: bool = False) -> str:
        del wait
        self.prepare()
        payload = {
            "phonopy_yaml": str(self.phonopy_yaml),
            "checkpoint_dir": str(self.checkpoint_dir),
            "model": str(self.method.model),
            "head": self.method.head,
            "device": self.method.device,
        }
        run_worker("forces", payload, self.workdir / "worker-payload.json", self.logdir / f"{self.method.name}.log")
        return "ready"

    def collect(self) -> tuple[np.ndarray, np.ndarray | None]:
        from phonopy import load

        phonon = load(str(self.phonopy_yaml), produce_fc=False)
        cells = phonon.supercells_with_displacements
        forces: list[np.ndarray] = []
        energies: list[float] = []
        missing: list[str] = []
        for index, cell in enumerate(cells, start=1):
            path = self.checkpoint_dir / f"disp-{index:04d}.npz"
            if not path.is_file():
                missing.append(path.name)
                continue
            try:
                with np.load(path) as data:
                    force = np.asarray(data["forces"], dtype=float)
                    energy = float(data["energy"])
            except Exception as exc:
                raise IncompleteResultsError(f"损坏的 DeepMD checkpoint: {path}: {exc}") from exc
            if force.shape != (len(cell), 3) or not np.all(np.isfinite(force)):
                raise IncompleteResultsError(f"DeepMD checkpoint 力数组无效: {path}")
            forces.append(force)
            energies.append(energy)
        if missing:
            raise IncompleteResultsError(f"缺少 {len(missing)} 个 DeepMD 位移结果: {', '.join(missing[:5])}")
        force_array = np.asarray(forces, dtype=float)
        energy_array = np.asarray(energies, dtype=float)
        self.resultdir.mkdir(parents=True, exist_ok=True)
        np.save(self.resultdir / "forces.npy", force_array)
        np.save(self.resultdir / "energies.npy", energy_array)
        atomic_write_json(self.resultdir / "force-provider.json", {
            "type": "deepmd",
            "model": str(self.method.model),
            "head": self.method.head,
            "device": self.method.device,
            "n_displacements": len(forces),
        })
        return force_array, energy_array

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .config import DeepMDMethod
from .errors import ExternalProgramError
from .providers.deepmd import device_environment
from .util import atomic_write_json, load_json


def run_anh_forces(
    method: DeepMDMethod,
    phono3py_yaml: Path,
    checkpoint_dir: Path,
    log_path: Path,
    *,
    subtract_residual_forces: bool,
) -> dict:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path = checkpoint_dir.parent / "worker-payload.json"
    result_path = checkpoint_dir.parent / "worker-result.json"
    payload = {
        "phono3py_yaml": str(phono3py_yaml),
        "checkpoint_dir": str(checkpoint_dir),
        "model": str(method.model),
        "head": method.head,
        "device": method.device,
        "subtract_residual_forces": subtract_residual_forces,
        "result_file": str(result_path),
    }
    atomic_write_json(payload_path, payload)
    env = device_environment(method.device)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            [sys.executable, "-m", "phonon_kit.anh_worker", str(payload_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    if process.returncode != 0 or not result_path.is_file():
        try:
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        except OSError:
            tail = ""
        raise ExternalProgramError(f"DeepMD 三阶位移力计算失败，日志: {log_path}\n{tail}")
    return load_json(result_path)

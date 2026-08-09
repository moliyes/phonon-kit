from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .errors import ExternalProgramError, RetryableExternalError
from .qha_config import QHAVaspMethod
from .util import atomic_write_json


BACKWARD_FILES = ["vasprun.xml", "OUTCAR", "OSZICAR", "CONTCAR", "vasp.stdout", "vasp.stderr"]
LOCAL_METADATA = {"atom-map.json", "README.txt", "task.json"}


def _is_retryable(exc: Exception) -> bool:
    for attribute in ("status", "status_code", "http_status"):
        value = getattr(exc, attribute, None)
        try:
            if 500 <= int(value) <= 599:
                return True
        except (TypeError, ValueError):
            pass
    message = str(exc)
    return bool(
        re.search(r"(?:status[^0-9]{0,8}|HTTP\s*)5\d\d\b", message, re.IGNORECASE)
        or re.search(r"['\"]status['\"]\s*:\s*5\d\d", message)
    )


def submit_qha_tasks(
    method: QHAVaspMethod,
    method_workdir: Path,
    stage: str,
    task_dirs: list[Path],
    *,
    wait: bool,
) -> dict[str, Any]:
    try:
        from dpdispatcher import Machine, Resources, Submission, Task
    except Exception as exc:
        raise ExternalProgramError(f"无法导入 dpdispatcher: {exc}") from exc

    unfinished = [path for path in task_dirs if not (path / "vasprun.xml").is_file()]
    if not unfinished:
        return {"status": "ready", "stage": stage, "n_tasks": len(task_dirs), "n_submitted": 0}
    try:
        machine = Machine.load_from_json(str(method.executor.machine))
        resources = Resources.load_from_json(str(method.executor.resources))
        tasks = []
        for task_dir in unfinished:
            forward = sorted(
                path.name
                for path in task_dir.iterdir()
                if path.is_file() and path.name not in LOCAL_METADATA and path.name not in BACKWARD_FILES
            )
            tasks.append(
                Task(
                    command=method.executor.command,
                    task_work_path=str(task_dir.relative_to(method_workdir)),
                    forward_files=forward,
                    backward_files=BACKWARD_FILES,
                    outlog="vasp.stdout",
                    errlog="vasp.stderr",
                )
            )
        submission = Submission(
            work_base=str(method_workdir),
            machine=machine,
            resources=resources,
            task_list=tasks,
            forward_common_files=[],
            backward_common_files=[],
        )
        submission.generate_jobs()
        submission.submission_hash = submission.get_hash()
        serialized = submission.run_submission(
            exit_on_submit=not wait,
            clean=bool(method.executor.clean_remote_after_success),
        )
        ready = all((path / "vasprun.xml").is_file() for path in task_dirs)
        result = {
            "status": "ready" if ready else "submitted",
            "stage": stage,
            "submission_hash": submission.submission_hash,
            "n_tasks": len(task_dirs),
            "n_submitted": len(tasks),
            "job_states": [str(getattr(job, "job_state", "unknown")) for job in submission.belonging_jobs],
            "waited": wait,
            "serialized_returned": isinstance(serialized, dict),
        }
        state_dir = method_workdir / "dispatcher"
        state_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(state_dir / f"{stage}.json", result)
        return result
    except Exception as exc:
        if _is_retryable(exc):
            raise RetryableExternalError(
                f"DPDispatcher 暂时性传输/服务错误 ({method.name}/{stage})，可直接 resume: {exc}"
            ) from exc
        raise ExternalProgramError(f"DPDispatcher 提交/恢复失败 ({method.name}/{stage}): {exc}") from exc

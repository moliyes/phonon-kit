from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import VaspMethod
from .errors import ExternalProgramError
from .util import atomic_write_json


BACKWARD_FILES = ["vasprun.xml", "OUTCAR", "OSZICAR", "CONTCAR", "vasp.stdout", "vasp.stderr"]


def validate_dispatcher_files(method: VaspMethod) -> dict[str, Any]:
    try:
        from dpdispatcher import Machine, Resources

        machine = Machine.load_from_json(str(method.executor.machine))
        resources = Resources.load_from_json(str(method.executor.resources))
    except Exception as exc:
        raise ExternalProgramError(f"DPDispatcher 配置加载失败 ({method.name}): {exc}") from exc
    return {
        "machine_class": type(machine).__name__,
        "resources_class": type(resources).__name__,
        "group_size": getattr(resources, "group_size", None),
    }


def submit_vasp(method: VaspMethod, method_workdir: Path, *, wait: bool) -> dict[str, Any]:
    try:
        from dpdispatcher import Machine, Resources, Submission, Task
    except Exception as exc:
        raise ExternalProgramError(f"无法导入 dpdispatcher: {exc}") from exc

    jobs_dir = method_workdir / "jobs"
    task_dirs = sorted(path for path in jobs_dir.glob("disp-*") if path.is_dir())
    unfinished = [path for path in task_dirs if not (path / "vasprun.xml").is_file()]
    if not unfinished:
        return {"status": "ready", "n_tasks": len(task_dirs), "n_submitted": 0}
    try:
        machine = Machine.load_from_json(str(method.executor.machine))
        resources = Resources.load_from_json(str(method.executor.resources))
        tasks = []
        for task_dir in task_dirs:
            if (task_dir / "vasprun.xml").is_file():
                continue
            forward = sorted(
                path.name
                for path in task_dir.iterdir()
                if path.is_file() and path.name not in {"atom-map.json", "README.txt"} and path.name not in BACKWARD_FILES
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
        submission_hash = submission.submission_hash
        serialized = submission.run_submission(
            exit_on_submit=not wait,
            clean=bool(method.executor.clean_remote_after_success),
        )
        states = [str(getattr(job, "job_state", "unknown")) for job in submission.belonging_jobs]
        ready = all((path / "vasprun.xml").is_file() for path in task_dirs)
        result = {
            "status": "ready" if ready else "submitted",
            "submission_hash": submission_hash,
            "n_tasks": len(task_dirs),
            "n_submitted": len(tasks),
            "job_states": states,
            "waited": wait,
            "serialized_returned": isinstance(serialized, dict),
        }
        atomic_write_json(method_workdir / "dispatcher-state.json", result)
        return result
    except Exception as exc:
        raise ExternalProgramError(f"DPDispatcher 提交/恢复失败 ({method.name}): {exc}") from exc


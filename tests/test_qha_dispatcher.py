from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from phonon_kit.config import DispatcherConfig
from phonon_kit.qha_config import QHAVaspMethod, QHAVaspTemplates
from phonon_kit.qha_dispatcher import _is_retryable
from phonon_kit.qha_dispatcher import submit_qha_tasks


def test_qha_dispatcher_classifies_http_5xx():
    assert _is_retryable(RuntimeError("{'status': 502, 'details': {}}"))
    assert _is_retryable(RuntimeError("HTTP 503 Service Unavailable"))
    assert not _is_retryable(RuntimeError("permission denied"))


def test_qha_dispatcher_async_resume_uses_stable_hash(tmp_path: Path, monkeypatch):
    machine_path = tmp_path / "machine.json"
    resources_path = tmp_path / "resources.json"
    machine_path.write_text("{}")
    resources_path.write_text("{}")
    method = QHAVaspMethod(
        "dft",
        "vasp",
        QHAVaspTemplates(tmp_path, tmp_path, tmp_path),
        {},
        DispatcherConfig("dpdispatcher", machine_path, resources_path, "vasp_std", False),
    )
    method_work = tmp_path / "method"
    task_dir = method_work / "phases" / "alpha" / "volumes" / "v000" / "static"
    task_dir.mkdir(parents=True)
    (task_dir / "POSCAR").write_text("input")
    hashes: list[str] = []

    class FakeMachine:
        @classmethod
        def load_from_json(cls, path):
            return cls()

    class FakeResources(FakeMachine):
        pass

    class FakeTask:
        def __init__(self, **kwargs):
            self.task_work_path = kwargs["task_work_path"]

    class FakeSubmission:
        def __init__(self, work_base, task_list, **kwargs):
            self.work_base = Path(work_base)
            self.task_list = task_list
            self.belonging_jobs = [SimpleNamespace(job_state="finished")]
            self.submission_hash = ""

        def generate_jobs(self):
            return None

        def get_hash(self):
            return "stable-submission-hash"

        def run_submission(self, exit_on_submit, clean):
            del clean
            hashes.append(self.submission_hash)
            if not exit_on_submit:
                for task in self.task_list:
                    output = self.work_base / task.task_work_path / "vasprun.xml"
                    output.write_text("<modeling/>")
            return {}

    fake = SimpleNamespace(
        Machine=FakeMachine,
        Resources=FakeResources,
        Submission=FakeSubmission,
        Task=FakeTask,
    )
    monkeypatch.setitem(sys.modules, "dpdispatcher", fake)
    submitted = submit_qha_tasks(method, method_work, "static", [task_dir], wait=False)
    assert submitted["status"] == "submitted"
    ready = submit_qha_tasks(method, method_work, "static", [task_dir], wait=True)
    assert ready["status"] == "ready"
    assert hashes == ["stable-submission-hash", "stable-submission-hash"]

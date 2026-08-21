from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from phonon_kit.config import DispatcherConfig, VaspMethod
from phonon_kit.dispatcher import submit_vasp


def test_vasp_resume_keeps_complete_submission_after_partial_download(tmp_path: Path, monkeypatch):
    template = tmp_path / "template"
    template.mkdir()
    machine_path = tmp_path / "machine.json"
    resources_path = tmp_path / "resources.json"
    machine_path.write_text("{}")
    resources_path.write_text("{}")
    method = VaspMethod(
        "dft",
        "vasp",
        template,
        DispatcherConfig("dpdispatcher", machine_path, resources_path, "vasp_std"),
    )
    work = tmp_path / "work"
    first = work / "jobs" / "disp-0001"
    second = work / "jobs" / "disp-0002"
    for task in (first, second):
        task.mkdir(parents=True)
        (task / "POSCAR").write_text("input")

    calls: list[tuple[str, int]] = []

    class FakeLoader:
        @classmethod
        def load_from_json(cls, path):
            return cls()

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
            return "stable"

        def run_submission(self, exit_on_submit, clean):
            del clean
            calls.append((self.submission_hash, len(self.task_list)))
            if not exit_on_submit:
                for task in self.task_list:
                    (self.work_base / task.task_work_path / "vasprun.xml").write_text("<modeling/>")
            return {}

    monkeypatch.setitem(
        sys.modules,
        "dpdispatcher",
        SimpleNamespace(Machine=FakeLoader, Resources=FakeLoader, Submission=FakeSubmission, Task=FakeTask),
    )
    assert submit_vasp(method, work, wait=False)["status"] == "submitted"
    (first / "vasprun.xml").write_text("<modeling/>")
    assert submit_vasp(method, work, wait=True)["status"] == "ready"
    assert calls == [("stable", 2), ("stable", 2)]

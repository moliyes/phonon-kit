from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .errors import RunStateError
from .util import atomic_write_json, load_json, utc_now


TERMINAL_STATUS = {"completed"}


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def canonical(self) -> Path:
        return self.work / "canonical"

    @property
    def displacements(self) -> Path:
        return self.work / "displacements"

    def method_work(self, name: str) -> Path:
        return self.work / "methods" / name

    def method_results(self, name: str) -> Path:
        return self.results / name


class StateStore:
    def __init__(self, paths: RunPaths):
        self.paths = paths

    def load(self) -> dict[str, Any]:
        if not self.paths.state_file.is_file():
            raise RunStateError(f"缺少状态文件: {self.paths.state_file}")
        return load_json(self.paths.state_file)

    def save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        atomic_write_json(self.paths.state_file, state)

    def update(self, **changes: Any) -> dict[str, Any]:
        state = self.load()
        state.update(changes)
        self.save(state)
        return state

    def update_stage(self, name: str, value: Any = True) -> dict[str, Any]:
        state = self.load()
        state.setdefault("stages", {})[name] = value
        self.save(state)
        return state

    def update_method(self, name: str, **changes: Any) -> dict[str, Any]:
        state = self.load()
        method = state.setdefault("methods", {}).setdefault(name, {})
        method.update(changes)
        self.save(state)
        return state


def _run_entries(config: Config) -> list[tuple[int, RunPaths, dict[str, Any] | None]]:
    runs_dir = config.project.runs_dir
    if not runs_dir.is_dir():
        return []
    pattern = re.compile(rf"^{re.escape(config.project.name)}-(\d{{3,}})$")
    entries: list[tuple[int, RunPaths, dict[str, Any] | None]] = []
    for path in runs_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if not match or not path.is_dir():
            continue
        paths = RunPaths(path)
        try:
            state = load_json(paths.state_file)
        except (FileNotFoundError, ValueError):
            state = None
        entries.append((int(match.group(1)), paths, state))
    return sorted(entries, key=lambda item: item[0])


def latest_run(config: Config) -> tuple[RunPaths, dict[str, Any]] | None:
    entries = _run_entries(config)
    if not entries:
        return None
    _, paths, state = entries[-1]
    if state is None:
        raise RunStateError(f"最新运行目录状态损坏: {paths.root}")
    return paths, state


def matching_run(config: Config) -> tuple[RunPaths, dict[str, Any]]:
    fingerprint = config.fingerprint()
    for _, paths, state in reversed(_run_entries(config)):
        if state and state.get("config_hash") == fingerprint:
            return paths, state
    raise RunStateError("没有找到与当前配置完全匹配的运行；输入文件或 YAML 可能已改变")


def choose_run(config: Config, requested_methods: list[str], *, force_new: bool = False) -> tuple[RunPaths, dict[str, Any], bool]:
    entries = _run_entries(config)
    fingerprint = config.fingerprint()
    if entries and not force_new:
        _, paths, state = entries[-1]
        if state is None:
            raise RunStateError(f"最新运行目录状态损坏: {paths.root}")
        if state.get("config_hash") == fingerprint:
            prior_methods = state.get("selected_methods", [])
            if prior_methods != requested_methods:
                raise RunStateError(
                    "同一配置的最新运行使用了不同的 --only 方法集合；请沿用原集合或加 --new"
                )
            return paths, state, False
        if state.get("status") not in TERMINAL_STATUS:
            raise RunStateError(
                f"最新运行 {paths.root.name} 尚未完成但配置已变化；请用该运行目录 resume，或用 --new 新建运行"
            )

    index = (entries[-1][0] + 1) if entries else 1
    root = config.project.runs_dir / f"{config.project.name}-{index:03d}"
    paths = RunPaths(root)
    for directory in (paths.work, paths.results, paths.logs, paths.canonical, paths.displacements):
        directory.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema_version": 1,
        "project": config.project.name,
        "run_id": root.name,
        "config_path": str(config.path),
        "config_hash": fingerprint,
        "selected_methods": requested_methods,
        "status": "created",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "stages": {},
        "methods": {name: {"status": "pending"} for name in requested_methods},
        "errors": [],
    }
    StateStore(paths).save(state)
    return paths, state, True

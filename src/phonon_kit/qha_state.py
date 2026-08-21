from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import RunStateError
from .qha_config import QHAConfig
from .util import atomic_write_json, load_json, utc_now


@dataclass(frozen=True)
class QHARunPaths:
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

    def method_work(self, method: str) -> Path:
        return self.work / "methods" / method

    def phase_work(self, method: str, phase: str) -> Path:
        return self.method_work(method) / "phases" / phase

    def volume_work(self, method: str, phase: str, volume_id: str) -> Path:
        return self.phase_work(method, phase) / "volumes" / volume_id

    def method_results(self, method: str) -> Path:
        return self.results / method

    def phase_results(self, method: str, phase: str) -> Path:
        return self.method_results(method) / "phases" / phase


class QHAStateStore:
    def __init__(self, paths: QHARunPaths):
        self.paths = paths

    def load(self) -> dict[str, Any]:
        if not self.paths.state_file.is_file():
            raise RunStateError(f"缺少 QHA 状态文件: {self.paths.state_file}")
        return load_json(self.paths.state_file)

    def save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        atomic_write_json(self.paths.state_file, state)

    def update(self, **changes: Any) -> dict[str, Any]:
        state = self.load()
        state.update(changes)
        self.save(state)
        return state

    def update_method(self, method: str, **changes: Any) -> dict[str, Any]:
        state = self.load()
        state.setdefault("methods", {}).setdefault(method, {}).update(changes)
        self.save(state)
        return state

    def update_phase(self, method: str, phase: str, **changes: Any) -> dict[str, Any]:
        state = self.load()
        phase_state = (
            state.setdefault("methods", {})
            .setdefault(method, {})
            .setdefault("phases", {})
            .setdefault(phase, {})
        )
        phase_state.update(changes)
        self.save(state)
        return state

    def update_volume(self, method: str, phase: str, volume_id: str, **changes: Any) -> dict[str, Any]:
        state = self.load()
        volume_state = (
            state.setdefault("methods", {})
            .setdefault(method, {})
            .setdefault("phases", {})
            .setdefault(phase, {})
            .setdefault("volumes", {})
            .setdefault(volume_id, {})
        )
        volume_state.update(changes)
        self.save(state)
        return state


def volume_id(index: int, ratio: float) -> str:
    return f"v{index:03d}-r{ratio:.6f}"


def _entries(config: QHAConfig) -> list[tuple[int, QHARunPaths, dict[str, Any] | None]]:
    if not config.project.runs_dir.is_dir():
        return []
    pattern = re.compile(rf"^{re.escape(config.project.name)}-(\d{{3,}})$")
    entries: list[tuple[int, QHARunPaths, dict[str, Any] | None]] = []
    for path in config.project.runs_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if not match or not path.is_dir():
            continue
        paths = QHARunPaths(path)
        try:
            state = load_json(paths.state_file)
        except (FileNotFoundError, ValueError):
            state = None
        entries.append((int(match.group(1)), paths, state))
    return sorted(entries, key=lambda item: item[0])


def latest_qha_run(config: QHAConfig) -> tuple[QHARunPaths, dict[str, Any]] | None:
    entries = _entries(config)
    if not entries:
        return None
    _, paths, state = entries[-1]
    if state is None:
        raise RunStateError(f"最新 QHA 运行目录状态损坏: {paths.root}")
    return paths, state


def matching_qha_run(config: QHAConfig) -> tuple[QHARunPaths, dict[str, Any]]:
    fingerprint = config.fingerprint()
    for _, paths, state in reversed(_entries(config)):
        if state and state.get("config_hash") == fingerprint:
            return paths, state
    raise RunStateError("没有找到与当前 QHA 配置及输入文件完全匹配的运行")


def choose_qha_run(
    config: QHAConfig,
    selected_methods: list[str],
    *,
    force_new: bool = False,
) -> tuple[QHARunPaths, dict[str, Any], bool]:
    entries = _entries(config)
    fingerprint = config.fingerprint()
    if entries and not force_new:
        _, paths, state = entries[-1]
        if state is None:
            raise RunStateError(f"最新 QHA 运行目录状态损坏: {paths.root}")
        if state.get("config_hash") == fingerprint:
            if state.get("selected_methods") != selected_methods:
                raise RunStateError("同一 QHA 配置使用了不同 --only 方法集合；请沿用原集合或加 --new")
            return paths, state, False
        if state.get("status") != "completed":
            raise RunStateError(
                f"最新 QHA 运行 {paths.root.name} 尚未完成但配置已改变；请用该运行目录 resume，或使用 --new"
            )

    index = entries[-1][0] + 1 if entries else 1
    paths = QHARunPaths(config.project.runs_dir / f"{config.project.name}-{index:03d}")
    for directory in (paths.work, paths.results, paths.logs):
        directory.mkdir(parents=True, exist_ok=True)
    method_states: dict[str, Any] = {}
    for method in selected_methods:
        phase_states: dict[str, Any] = {}
        for phase_name, phase in config.phases.items():
            phase_states[phase_name] = {
                "status": "pending",
                "volumes": {
                    volume_id(index, ratio): {"ratio": ratio, "status": "pending"}
                    for index, ratio in enumerate(phase.volume_ratios)
                },
            }
        method_states[method] = {"status": "pending", "phases": phase_states}
    state = {
        "schema_version": 1,
        "workflow": "qha",
        "project": config.project.name,
        "run_id": paths.root.name,
        "config_path": str(config.path),
        "config_hash": fingerprint,
        "selected_methods": selected_methods,
        "status": "created",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "methods": method_states,
        "comparison": {"status": "pending"},
        "errors": [],
    }
    QHAStateStore(paths).save(state)
    return paths, state, True

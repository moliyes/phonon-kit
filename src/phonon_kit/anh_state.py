from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .anh_config import AnhConfig
from .errors import RunStateError
from .util import atomic_write_json, load_json, utc_now


TERMINAL_STATUS = {"completed"}


@dataclass(frozen=True)
class AnhRunPaths:
    root: Path

    @property
    def state_file(self) -> Path: return self.root / "state.json"
    @property
    def work(self) -> Path: return self.root / "work"
    @property
    def results(self) -> Path: return self.root / "results"
    @property
    def logs(self) -> Path: return self.root / "logs"
    @property
    def canonical(self) -> Path: return self.work / "canonical"
    @property
    def displacements(self) -> Path: return self.work / "displacements"
    def method_work(self, name: str) -> Path: return self.work / "methods" / name
    def method_results(self, name: str) -> Path: return self.results / name


class AnhStateStore:
    def __init__(self, paths: AnhRunPaths): self.paths = paths
    def load(self) -> dict[str, Any]:
        if not self.paths.state_file.is_file():
            raise RunStateError(f"缺少三阶状态文件: {self.paths.state_file}")
        return load_json(self.paths.state_file)
    def save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now(); atomic_write_json(self.paths.state_file, state)
    def update_stage(self, name: str, value: Any) -> None:
        state = self.load(); state.setdefault("stages", {})[name] = value; self.save(state)
    def update_method(self, name: str, **changes: Any) -> None:
        state = self.load(); state.setdefault("methods", {}).setdefault(name, {}).update(changes); self.save(state)


def _entries(config: AnhConfig) -> list[tuple[int, AnhRunPaths, dict[str, Any] | None]]:
    if not config.project.runs_dir.is_dir(): return []
    pattern = re.compile(rf"^{re.escape(config.project.name)}-(\d{{3,}})$")
    entries = []
    for path in config.project.runs_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if not match or not path.is_dir(): continue
        paths = AnhRunPaths(path)
        try: state = load_json(paths.state_file)
        except (FileNotFoundError, ValueError): state = None
        entries.append((int(match.group(1)), paths, state))
    return sorted(entries)


def latest_anh_run(config: AnhConfig) -> tuple[AnhRunPaths, dict[str, Any]] | None:
    entries = _entries(config)
    if not entries: return None
    _, paths, state = entries[-1]
    if state is None: raise RunStateError(f"最新三阶运行状态损坏: {paths.root}")
    return paths, state


def choose_anh_run(config: AnhConfig, *, force_new: bool = False) -> tuple[AnhRunPaths, dict[str, Any], bool]:
    entries = _entries(config); fingerprint = config.fingerprint()
    if entries and not force_new:
        _, paths, state = entries[-1]
        if state is None: raise RunStateError(f"最新三阶运行状态损坏: {paths.root}")
        if state.get("config_hash") == fingerprint: return paths, state, False
        if state.get("status") not in TERMINAL_STATUS:
            raise RunStateError(f"最新运行 {paths.root.name} 尚未完成但配置已变化；请指定该运行目录继续，或加 --new")
    index = entries[-1][0] + 1 if entries else 1
    paths = AnhRunPaths(config.project.runs_dir / f"{config.project.name}-{index:03d}")
    for directory in (paths.work, paths.results, paths.logs, paths.canonical, paths.displacements):
        directory.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": 1, "workflow": "anharmonic", "project": config.project.name,
        "run_id": paths.root.name, "config_path": str(config.path), "config_hash": fingerprint,
        "selected_methods": list(config.enabled_methods), "status": "created",
        "created_at": utc_now(), "updated_at": utc_now(), "stages": {},
        "methods": {name: {"status": "pending"} for name in config.enabled_methods}, "errors": [],
    }
    AnhStateStore(paths).save(state)
    return paths, state, True

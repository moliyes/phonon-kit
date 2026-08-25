from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import yaml

from .anh_config import AnhConfig, load_anh_config
from .util import atomic_write_text


SNAPSHOT_CONFIG = "config.resolved.yaml"


def create_anh_snapshot(config: AnhConfig, root: Path) -> AnhConfig:
    snapshot = root / SNAPSHOT_CONFIG
    if snapshot.is_file():
        return load_anh_config(snapshot)
    with tempfile.TemporaryDirectory(prefix=".anh-inputs-", dir=root) as temporary:
        temp_inputs = Path(temporary)
        final_inputs = root / "inputs"
        data = config.resolved_dict()
        data["project"]["runs_dir"] = ".."
        structure = temp_inputs / "structure" / "POSCAR"
        structure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.structure.file, structure)
        data["structure"]["file"] = "inputs/structure/POSCAR"
        if config.structure.born_file is not None:
            born = temp_inputs / "structure" / "BORN"
            shutil.copy2(config.structure.born_file, born)
            data["structure"]["born_file"] = "inputs/structure/BORN"
        else:
            data["structure"]["born_file"] = None
        kept_methods = {}
        for name, method in config.enabled_methods.items():
            model = temp_inputs / "models" / f"{name}{method.model.suffix.lower()}"
            model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(method.model, model)
            item = data["methods"][name]
            item["model"] = f"inputs/models/{model.name}"
            kept_methods[name] = item
        data["methods"] = kept_methods
        if (root / "inputs").exists():
            raise FileExistsError(f"运行输入快照已存在但配置快照缺失: {root / 'inputs'}")
        os.replace(temp_inputs, root / "inputs")
        atomic_write_text(snapshot, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return load_anh_config(snapshot)

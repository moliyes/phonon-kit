from pathlib import Path

from phonon_kit.config import load_config
from phonon_kit.initializer import init_case


def test_init_creates_ready_layout(tmp_path: Path):
    target = init_case(tmp_path / "case")
    config = load_config(target / "config.yaml")
    assert config.project.name == "sio2"
    assert (target / "inputs" / "POSCAR").is_file()
    assert (target / "inputs" / "vasp" / "INCAR").is_file()
    assert not (target / "inputs" / "vasp" / "POTCAR").exists()


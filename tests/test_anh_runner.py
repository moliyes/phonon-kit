from __future__ import annotations

from pathlib import Path

from phonon_kit.anh_config import load_anh_config
from phonon_kit.anh_runner import run_anh_config
from phonon_kit.anh_snapshot import create_anh_snapshot
from phonon_kit.anh_state import choose_anh_run



def write_anh_config(root: Path) -> Path:
    (root / "POSCAR").write_text("""Al
1.0
4.05 0 0
0 4.05 0
0 0 4.05
Al
1
Direct
0 0 0
""", encoding="utf-8")
    (root / "model.pth").write_bytes(b"model")
    path = root / "anh.yaml"
    path.write_text("""schema_version: 1
project:
  name: snapshot
  runs_dir: runs
structure:
  file: POSCAR
anharmonic:
  supercell: [1, 1, 1]
  mesh: [3, 3, 3]
  temperature_min_k: 300
  temperature_max_k: 300
  temperature_step_k: 100
  lifetime_temperature_k: 300
methods:
  dp:
    type: deepmd
    model: model.pth
    device: cpu
""", encoding="utf-8")
    return path


def test_explicit_run_uses_snapshot_after_case_inputs_change(tmp_path: Path, monkeypatch) -> None:
    config = load_anh_config(write_anh_config(tmp_path))
    paths, _, _ = choose_anh_run(config)
    snapshot = create_anh_snapshot(config, paths.root)
    original_snapshot_model = snapshot.methods["dp"].model

    config.structure.file.write_text("broken case input", encoding="utf-8")
    config.methods["dp"].model.unlink()
    seen: dict[str, Path] = {}

    def fake_validate(method, structure, workdir, logdir):
        seen["model"] = method.model
        seen["structure"] = structure
        return {"ok": True}

    monkeypatch.setattr("phonon_kit.anh_runner.validate_method", fake_validate)
    monkeypatch.setattr("phonon_kit.anh_runner.run_anh_forces", lambda *args, **kwargs: {"completed": 1, "total": 1})
    monkeypatch.setattr("phonon_kit.anh_runner.produce_force_constants", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr("phonon_kit.anh_runner.analyze_anh_method", lambda *args, **kwargs: {"thermodynamic_stability": True})
    monkeypatch.setattr("phonon_kit.anh_runner.compare_anh_methods", lambda *args, **kwargs: None)
    monkeypatch.setattr("phonon_kit.anh_runner.result_is_complete", lambda *args, **kwargs: False)

    run_paths, state, created = run_anh_config(config, paths=paths)
    assert created is False
    assert state["status"] == "completed"
    assert run_paths.root == paths.root
    assert seen["model"] == original_snapshot_model
    assert seen["model"].is_file()
    assert seen["structure"].is_relative_to(paths.root)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phonon_kit.config import DispatcherConfig, VaspMethod, load_config
from phonon_kit.errors import IncompleteResultsError
from phonon_kit.providers.vasp import VaspProvider, _parse_force_file, parse_incar, validate_vasp_template
from phonon_kit.structure import generate_displacements
from phonon_kit.util import load_json
from phonon_kit.vasp_input import has_vasp_kpoint_source, missing_vasp_template_inputs


def method(tmp_path: Path) -> VaspMethod:
    template = tmp_path / "vasp"
    template.mkdir(parents=True)
    (template / "INCAR").write_text("IBRION=-1\nNSW=0\nPREC=Accurate\nLREAL=.FALSE.\nEDIFF=1E-8\n")
    (template / "KPOINTS").write_text("KPOINTS")
    (template / "POTCAR").write_text("POTCAR")
    executor = DispatcherConfig("dpdispatcher", tmp_path / "machine.json", tmp_path / "resources.json", "vasp_std")
    return VaspMethod("dft", "vasp", template, executor)


def test_static_incar_validation(tmp_path: Path):
    item = method(tmp_path)
    assert validate_vasp_template(item) == []
    assert parse_incar(item.template_dir / "INCAR")["IBRION"] == "-1"


def test_kspacing_can_replace_kpoints(tmp_path: Path):
    item = method(tmp_path)
    (item.template_dir / "KPOINTS").unlink()
    with (item.template_dir / "INCAR").open("a", encoding="utf-8") as handle:
        handle.write("KSPACING = 0.20\n")
    assert has_vasp_kpoint_source(item.template_dir)
    assert missing_vasp_template_inputs(item.template_dir) == []


@pytest.mark.parametrize("value", ["0", "-0.2", "not-a-number"])
def test_invalid_kspacing_does_not_replace_kpoints(tmp_path: Path, value: str):
    item = method(tmp_path)
    (item.template_dir / "KPOINTS").unlink()
    with (item.template_dir / "INCAR").open("a", encoding="utf-8") as handle:
        handle.write(f"KSPACING = {value}\n")
    assert not has_vasp_kpoint_source(item.template_dir)
    assert missing_vasp_template_inputs(item.template_dir) == [
        "KPOINTS（或 INCAR 中的正数 KSPACING）"
    ]


def test_relaxing_incar_is_rejected(tmp_path: Path):
    item = method(tmp_path)
    (item.template_dir / "INCAR").write_text("IBRION=2\nNSW=20\n")
    with pytest.raises(ValueError, match="IBRION"):
        validate_vasp_template(item)


def test_prepare_vasp_jobs_with_atom_maps(config_path: Path, tmp_path: Path):
    config = load_config(config_path)
    displacements = tmp_path / "displacements"
    manifest = generate_displacements(config, config.structure.file, displacements)
    item = method(tmp_path / "dft-input")
    provider = VaspProvider(
        item,
        displacements / "phonopy_disp.yaml",
        tmp_path / "method-work",
        tmp_path / "method-result",
        tmp_path / "logs",
    )
    provider.prepare()
    jobs = sorted(provider.jobs_dir.glob("disp-*"))
    assert len(jobs) == manifest["n_displacements"]
    assert all((job / "POSCAR").is_file() for job in jobs)
    assert all((job / "POTCAR").is_file() for job in jobs)
    mapping = load_json(jobs[0] / "atom-map.json")
    assert sorted(mapping["written_to_canonical"]) == list(range(manifest["n_atoms_supercell"]))

    original = (jobs[0] / "INCAR").read_text(encoding="utf-8")
    (item.template_dir / "INCAR").write_text("IBRION=-1\nNSW=0\nEDIFF=1E-4\n", encoding="utf-8")
    provider.prepare()
    assert (jobs[0] / "INCAR").read_text(encoding="utf-8") == original


def test_vasprun_integrity_and_phonopy_parser(tmp_path: Path, monkeypatch):
    import phonopy.interface.vasp

    path = tmp_path / "vasprun.xml"
    path.write_text("<modeling></modeling>", encoding="utf-8")
    expected = np.arange(9, dtype=float).reshape(3, 3)
    monkeypatch.setattr(
        phonopy.interface.vasp,
        "parse_set_of_forces",
        lambda n, files, verbose=False: {"forces": [expected]},
    )
    assert np.array_equal(_parse_force_file(path, 3), expected)

    monkeypatch.setattr(phonopy.interface.vasp, "parse_set_of_forces", lambda n, files, verbose=False: [expected])
    assert np.array_equal(_parse_force_file(path, 3), expected)

    path.write_text("<modeling>", encoding="utf-8")
    with pytest.raises(IncompleteResultsError, match="损坏"):
        _parse_force_file(path, 3)

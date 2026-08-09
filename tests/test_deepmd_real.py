from __future__ import annotations

import os
from pathlib import Path

import pytest

from phonon_kit.config import DeepMDMethod
from phonon_kit.providers.deepmd import validate_method


@pytest.mark.integration
def test_existing_dpa3_cpu_smoke(tmp_path: Path):
    if os.environ.get("PHONON_KIT_RUN_DEEPMD_TESTS") != "1":
        pytest.skip("set PHONON_KIT_RUN_DEEPMD_TESTS=1 to run real DeepMD inference")
    model = Path("/home/begin/calypso-dp-search/models/dpa3_ft_n1600.pth")
    structure = Path("/home/begin/phonon_workflow_portable_8.9/structures/default_sio2/0_structure/POSCAR")
    if not model.is_file() or not structure.is_file():
        pytest.skip("local reference model/structure not found")
    method = DeepMDMethod("dpa3", "deepmd", model, str(model), "cpu")
    result = validate_method(method, structure, tmp_path / "work", tmp_path / "logs")
    assert result["n_atoms"] > 0
    assert result["force_max_ev_angstrom"] >= 0


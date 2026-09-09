from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .anh_config import AnhConfig
from .structure import ase_to_phonopy, phonopy_to_ase, read_structure, write_vasp_grouped
from .util import atomic_write_json


def _parse_born_with_vasp_units(primitive, filename: Path, symprec: float) -> dict[str, Any]:
    """Parse a Phonopy BORN file and restore the VASP NAC unit factor.

    Phonopy 4.4 ``parse_BORN`` returns Born charges and the dielectric tensor,
    but no longer includes the ``factor`` required by Phono3py's dynamical
    matrix.  BORN files used by phonon-kit follow the VASP convention.
    """
    from phonopy.file_IO import parse_BORN
    from phonopy.interface.calculator import get_calculator_physical_units

    nac_params = parse_BORN(
        primitive,
        symprec=symprec,
        filename=str(filename),
        lang="Rust",
    )
    nac_params.setdefault("factor", get_calculator_physical_units("vasp").nac_factor)
    return nac_params


def build_phono3py(config: AnhConfig, structure: Path):
    from phono3py import Phono3py
    from phonopy.structure.cells import PrimitiveMatrixAutoDefaultWarning

    atoms = read_structure(structure)
    kwargs: dict[str, Any] = {
        "unitcell": ase_to_phonopy(atoms),
        "supercell_matrix": np.diag(config.anharmonic.supercell),
        "primitive_matrix": "auto",
        "symprec": config.anharmonic.symmetry_tolerance,
        "log_level": 0,
    }
    if config.anharmonic.fc2_supercell is not None:
        kwargs["phonon_supercell_matrix"] = np.diag(config.anharmonic.fc2_supercell)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrimitiveMatrixAutoDefaultWarning)
        ph3 = Phono3py(**kwargs)
    if config.structure.born_file is not None:
        ph3.nac_params = _parse_born_with_vasp_units(
            ph3.primitive,
            config.structure.born_file,
            config.anharmonic.symmetry_tolerance,
        )
    return ph3


def _generate_displacements(config: AnhConfig, ph3) -> None:
    """Generate fc3 and, when requested, independent-distance fc2 datasets."""
    ph3.generate_displacements(distance=config.anharmonic.displacement_angstrom)
    if ph3.phonon_supercell_matrix is not None:
        ph3.generate_fc2_displacements(
            distance=config.anharmonic.resolved_fc2_displacement_angstrom,
            is_diagonal=config.anharmonic.fc2_is_diagonal,
        )


def displacement_plan(config: AnhConfig) -> dict[str, Any]:
    ph3 = build_phono3py(config, config.structure.file)
    _generate_displacements(config, ph3)
    n_fc3 = len(ph3.supercells_with_displacements)
    separate_fc2 = ph3.phonon_supercell_matrix is not None
    n_fc2 = len(ph3.phonon_supercells_with_displacements) if separate_fc2 else 0
    residual_per_model = int(config.anharmonic.subtract_residual_forces) * (2 if separate_fc2 else 1)
    n_models = len(config.enabled_methods)
    per_model = n_fc3 + n_fc2 + residual_per_model
    return {
        "supercell": list(config.anharmonic.supercell),
        "fc2_supercell": list(config.anharmonic.fc2_supercell or config.anharmonic.supercell),
        "fc2_uses_separate_displacements": separate_fc2,
        "n_atoms_unitcell": len(ph3.unitcell),
        "n_atoms_fc3_supercell": len(ph3.supercell),
        "n_atoms_fc2_supercell": len(ph3.phonon_supercell),
        "fc3_displacements": n_fc3,
        "fc2_displacements": n_fc2,
        "fc3_displacement_angstrom": config.anharmonic.displacement_angstrom,
        "fc2_displacement_angstrom": config.anharmonic.resolved_fc2_displacement_angstrom,
        "fc2_is_diagonal": config.anharmonic.fc2_is_diagonal,
        "residual_evaluations_per_model": residual_per_model,
        "models": list(config.enabled_methods),
        "force_evaluations_per_model": per_model,
        "total_force_evaluations": per_model * n_models,
        "mesh": list(config.anharmonic.mesh),
        "nac_enabled": config.structure.born_file is not None,
    }


def generate_anh_displacements(config: AnhConfig, canonical: Path, outdir: Path) -> dict[str, Any]:
    ph3 = build_phono3py(config, canonical)
    _generate_displacements(config, ph3)
    cells = ph3.supercells_with_displacements
    if not cells:
        raise RuntimeError("Phono3py 没有生成三阶位移超胞")
    outdir.mkdir(parents=True, exist_ok=True)
    yaml_path = outdir / "phono3py_disp.yaml"
    ph3.save(yaml_path, settings={"force_sets": False, "force_constants": False})
    write_vasp_grouped(phonopy_to_ase(ph3.unitcell), outdir / "POSCAR-unitcell")
    write_vasp_grouped(phonopy_to_ase(ph3.supercell), outdir / "SPOSCAR-fc3")
    separate = ph3.phonon_supercell_matrix is not None
    if separate:
        write_vasp_grouped(phonopy_to_ase(ph3.phonon_supercell), outdir / "SPOSCAR-fc2")
    n_fc2 = len(ph3.phonon_supercells_with_displacements) if separate else 0
    residual_count = int(config.anharmonic.subtract_residual_forces) * (2 if separate else 1)
    per_model = len(cells) + n_fc2 + residual_count
    manifest = {
        "supercell": list(config.anharmonic.supercell),
        "fc2_supercell": list(config.anharmonic.fc2_supercell or config.anharmonic.supercell),
        "fc2_uses_separate_displacements": separate,
        "n_atoms_unitcell": len(ph3.unitcell),
        "n_atoms_fc3_supercell": len(ph3.supercell),
        "n_atoms_fc2_supercell": len(ph3.phonon_supercell),
        "fc3_displacements": len(cells),
        "fc2_displacements": n_fc2,
        "fc3_displacement_angstrom": config.anharmonic.displacement_angstrom,
        "fc2_displacement_angstrom": config.anharmonic.resolved_fc2_displacement_angstrom,
        "fc2_is_diagonal": config.anharmonic.fc2_is_diagonal,
        "residual_evaluations_per_model": residual_count,
        "models": list(config.enabled_methods),
        "force_evaluations_per_model": per_model,
        "total_force_evaluations": per_model * len(config.enabled_methods),
        "mesh": list(config.anharmonic.mesh),
        "nac_enabled": config.structure.born_file is not None,
        "phono3py_yaml": str(yaml_path),
        "source_structure": str(canonical),
        "resolved_primitive_matrix": np.asarray(ph3.primitive_matrix, dtype=float).tolist(),
    }
    atomic_write_json(outdir / "manifest.json", manifest)
    return manifest


def load_anh_displacements(config: AnhConfig, yaml_path: Path, *, produce_fc: bool = False):
    from phono3py import load

    kwargs: dict[str, Any] = {"phono3py_yaml": str(yaml_path), "produce_fc": produce_fc, "is_nac": False}
    ph3 = load(**kwargs)
    if config.structure.born_file is not None:
        ph3.nac_params = _parse_born_with_vasp_units(
            ph3.primitive,
            config.structure.born_file,
            config.anharmonic.symmetry_tolerance,
        )
    return ph3

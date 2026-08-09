from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from .errors import IncompleteResultsError
from .providers.vasp import _parse_force_file
from .qha_config import QHAConfig, QHAVaspMethod
from .qha_dispatcher import submit_qha_tasks
from .qha_state import QHARunPaths, volume_id
from .qha_structure import canonicalize_primitive_structure, generate_qha_displacements, scale_structure
from .structure import read_structure, write_vasp_grouped
from .util import atomic_write_json, load_json


def _copy_template(directory: Path, task: Path) -> None:
    task.mkdir(parents=True, exist_ok=True)
    for source in sorted(path for path in directory.iterdir() if path.is_file()):
        destination = task / source.name
        if not destination.exists() or source.stat().st_mtime_ns > destination.stat().st_mtime_ns:
            shutil.copy2(source, destination)


def _restore_canonical_atoms(path: Path, mapping_path: Path):
    atoms = read_structure(path)
    mapping = load_json(mapping_path)
    written_to_canonical = np.asarray(mapping["written_to_canonical"], dtype=int)
    if sorted(written_to_canonical.tolist()) != list(range(len(atoms))):
        raise IncompleteResultsError(f"原子顺序映射不是双射: {mapping_path}")
    return atoms[np.argsort(written_to_canonical)]


def _vasp_atoms(path: Path):
    try:
        ET.parse(path)
        from ase.io import read

        return read(str(path), format="vasp-xml", index=-1)
    except Exception as exc:
        raise IncompleteResultsError(f"vasprun.xml 损坏或未完成: {path}: {exc}") from exc


def parse_e0_energy(path: Path) -> float:
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:
        raise IncompleteResultsError(f"vasprun.xml 损坏或未完成: {path}: {exc}") from exc
    values: list[float] = []
    for element in root.iter("i"):
        if element.attrib.get("name") == "e_0_energy" and element.text:
            try:
                values.append(float(element.text.strip()))
            except ValueError:
                continue
    if not values or not np.isfinite(values[-1]):
        raise IncompleteResultsError(f"无法从 VASP 结果读取最终 e_0_energy: {path}")
    return values[-1]


def target_volume(structure: Path, ratio: float) -> float:
    return float(read_structure(structure).get_volume()) * ratio


def prepare_relaxation_tasks(config: QHAConfig, method: QHAVaspMethod, paths: QHARunPaths) -> list[Path]:
    tasks: list[Path] = []
    for phase_name, phase in config.phases.items():
        reference = paths.phase_work(method.name, phase_name) / "reference" / "POSCAR-primitive"
        if not reference.is_file():
            canonicalize_primitive_structure(phase.structure, reference, phase.phonon)
        for index, ratio in enumerate(phase.volume_ratios):
            vid = volume_id(index, ratio)
            volume_root = paths.volume_work(method.name, phase_name, vid)
            scaled = volume_root / "POSCAR-scaled"
            if not scaled.is_file():
                scale_structure(reference, scaled, ratio)
            task = volume_root / "relaxation"
            _copy_template(method.template_for(phase_name, "volume_relax"), task)
            if not (task / "vasprun.xml").is_file():
                write_vasp_grouped(read_structure(scaled), task / "POSCAR", task / "atom-map.json")
            atomic_write_json(
                task / "task.json",
                {"stage": "volume_relax", "phase": phase_name, "volume_id": vid, "ratio": ratio},
            )
            tasks.append(task)
    return tasks


def collect_relaxation_tasks(
    config: QHAConfig,
    method: QHAVaspMethod,
    paths: QHARunPaths,
) -> dict[tuple[str, str], dict[str, Any]]:
    results: dict[tuple[str, str], dict[str, Any]] = {}
    missing: list[Path] = []
    for phase_name, phase in config.phases.items():
        for index, ratio in enumerate(phase.volume_ratios):
            vid = volume_id(index, ratio)
            task = paths.volume_work(method.name, phase_name, vid) / "relaxation"
            xml = task / "vasprun.xml"
            if not xml.is_file() or not (task / "CONTCAR").is_file():
                missing.append(xml)
                continue
            vasp_atoms = _vasp_atoms(xml)
            canonical = _restore_canonical_atoms(task / "CONTCAR", task / "atom-map.json")
            reference = paths.phase_work(method.name, phase_name) / "reference" / "POSCAR-primitive"
            wanted_volume = target_volume(reference, ratio)
            actual_volume = float(canonical.get_volume())
            volume_error = abs(actual_volume - wanted_volume) / wanted_volume
            forces = np.asarray(vasp_atoms.get_forces(), dtype=float)
            stress = np.asarray(vasp_atoms.get_stress(voigt=True), dtype=float)
            deviatoric = stress.copy()
            deviatoric[:3] -= np.mean(stress[:3])
            fmax = float(np.linalg.norm(forces, axis=1).max())
            dev_gpa = float(np.max(np.abs(deviatoric)) * 160.21766208)
            failures: list[str] = []
            if volume_error > config.volume_relaxation.volume_tolerance_relative:
                failures.append(f"相对体积误差 {volume_error:.3g}")
            if fmax > config.volume_relaxation.fmax_ev_angstrom:
                failures.append(f"最大力 {fmax:.6g} eV/A")
            if dev_gpa > config.volume_relaxation.max_deviatoric_stress_gpa:
                failures.append(f"最大偏应力 {dev_gpa:.6g} GPa")
            relaxed = task / "POSCAR-relaxed-canonical"
            write_vasp_grouped(canonical, relaxed)
            summary = {
                "converged": not failures,
                "target_volume_angstrom3": wanted_volume,
                "volume_angstrom3": actual_volume,
                "volume_error_relative": volume_error,
                "fmax_ev_angstrom": fmax,
                "stress_ev_angstrom3": stress.tolist(),
                "max_deviatoric_stress_gpa": dev_gpa,
                "relaxed_structure": str(relaxed),
                "failure_reasons": failures,
            }
            atomic_write_json(task / "summary.json", summary)
            if failures:
                raise RuntimeError(f"VASP 固定体积弛豫未通过 ({phase_name}/{vid}): {'; '.join(failures)}")
            results[(phase_name, vid)] = summary
    if missing:
        raise IncompleteResultsError(f"VASP 固定体积弛豫尚缺 {len(missing)} 个结果；第一个: {missing[0]}")
    return results


def prepare_static_tasks(config: QHAConfig, method: QHAVaspMethod, paths: QHARunPaths) -> list[Path]:
    tasks: list[Path] = []
    for phase_name, phase in config.phases.items():
        for index, ratio in enumerate(phase.volume_ratios):
            vid = volume_id(index, ratio)
            volume_root = paths.volume_work(method.name, phase_name, vid)
            relaxed = volume_root / "relaxation" / "POSCAR-relaxed-canonical"
            if not relaxed.is_file():
                raise IncompleteResultsError(f"缺少弛豫结构，不能准备静态任务: {relaxed}")
            task = volume_root / "static"
            _copy_template(method.template_for(phase_name, "static"), task)
            if not (task / "vasprun.xml").is_file():
                write_vasp_grouped(read_structure(relaxed), task / "POSCAR", task / "atom-map.json")
            atomic_write_json(
                task / "task.json",
                {"stage": "static", "phase": phase_name, "volume_id": vid, "ratio": ratio},
            )
            tasks.append(task)
    return tasks


def prepare_phonon_tasks(config: QHAConfig, method: QHAVaspMethod, paths: QHARunPaths) -> list[Path]:
    tasks: list[Path] = []
    for phase_name, phase in config.phases.items():
        for index, ratio in enumerate(phase.volume_ratios):
            vid = volume_id(index, ratio)
            volume_root = paths.volume_work(method.name, phase_name, vid)
            relaxed = volume_root / "relaxation" / "POSCAR-relaxed-canonical"
            phonon_root = volume_root / "phonon"
            manifest_path = phonon_root / "manifest.json"
            if not manifest_path.is_file():
                generate_qha_displacements(relaxed, phase.phonon, phonon_root)
            manifest = load_json(manifest_path)
            for disp_index in range(1, int(manifest["n_displacements"]) + 1):
                task = phonon_root / "jobs" / f"disp-{disp_index:04d}"
                _copy_template(method.template_for(phase_name, "phonon"), task)
                if not (task / "vasprun.xml").is_file():
                    shutil.copy2(phonon_root / f"POSCAR-{disp_index:04d}", task / "POSCAR")
                    shutil.copy2(phonon_root / f"atom-map-{disp_index:04d}.json", task / "atom-map.json")
                atomic_write_json(
                    task / "task.json",
                    {"stage": "phonon", "phase": phase_name, "volume_id": vid, "displacement": disp_index},
                )
                tasks.append(task)
    return tasks


def collect_static_tasks(
    config: QHAConfig,
    method: QHAVaspMethod,
    paths: QHARunPaths,
) -> dict[tuple[str, str], float]:
    energies: dict[tuple[str, str], float] = {}
    missing: list[Path] = []
    for phase_name, phase in config.phases.items():
        for index, ratio in enumerate(phase.volume_ratios):
            vid = volume_id(index, ratio)
            xml = paths.volume_work(method.name, phase_name, vid) / "static" / "vasprun.xml"
            if not xml.is_file():
                missing.append(xml)
                continue
            energy = parse_e0_energy(xml)
            atomic_write_json(xml.parent / "summary.json", {"energy_ev_cell": energy, "energy_source": "e_0_energy"})
            energies[(phase_name, vid)] = energy
    if missing:
        raise IncompleteResultsError(f"VASP 静态能尚缺 {len(missing)} 个结果；第一个: {missing[0]}")
    return energies


def collect_phonon_tasks(
    config: QHAConfig,
    method: QHAVaspMethod,
    paths: QHARunPaths,
) -> dict[tuple[str, str], np.ndarray]:
    results: dict[tuple[str, str], np.ndarray] = {}
    missing: list[Path] = []
    for phase_name, phase in config.phases.items():
        for index, ratio in enumerate(phase.volume_ratios):
            vid = volume_id(index, ratio)
            phonon_root = paths.volume_work(method.name, phase_name, vid) / "phonon"
            manifest = load_json(phonon_root / "manifest.json")
            n_displacements = int(manifest["n_displacements"])
            n_atoms = int(manifest["n_atoms_supercell"])
            forces: list[np.ndarray] = []
            for disp_index in range(1, n_displacements + 1):
                task = phonon_root / "jobs" / f"disp-{disp_index:04d}"
                xml = task / "vasprun.xml"
                if not xml.is_file():
                    missing.append(xml)
                    continue
                raw = _parse_force_file(xml, n_atoms)
                mapping = load_json(task / "atom-map.json")
                written_to_canonical = np.asarray(mapping["written_to_canonical"], dtype=int)
                if sorted(written_to_canonical.tolist()) != list(range(n_atoms)):
                    raise IncompleteResultsError(f"原子顺序映射不是双射: {task / 'atom-map.json'}")
                canonical = np.empty_like(raw)
                canonical[written_to_canonical] = raw
                forces.append(canonical)
            if len(forces) == n_displacements:
                array = np.asarray(forces, dtype=float)
                resultdir = paths.phase_results(method.name, phase_name) / "volumes" / vid
                resultdir.mkdir(parents=True, exist_ok=True)
                np.save(resultdir / "forces.npy", array)
                results[(phase_name, vid)] = array
    if missing:
        raise IncompleteResultsError(f"VASP 声子力尚缺 {len(missing)} 个结果；第一个: {missing[0]}")
    return results


def submit_stage(
    method: QHAVaspMethod,
    paths: QHARunPaths,
    stage: str,
    tasks: list[Path],
    *,
    wait: bool,
) -> dict[str, Any]:
    return submit_qha_tasks(method, paths.method_work(method.name), stage, tasks, wait=wait)

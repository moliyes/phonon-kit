from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import VaspMethod
from ..dispatcher import submit_vasp
from ..errors import IncompleteResultsError
from ..structure import phonopy_to_ase, write_vasp_grouped
from ..util import atomic_write_json, load_json


def parse_incar(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.split("!", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.split(";"):
            if "=" in part:
                key, value = part.split("=", 1)
                values[key.strip().upper()] = value.strip()
    return values


def validate_vasp_template(method: VaspMethod) -> list[str]:
    warnings: list[str] = []
    incar = parse_incar(method.template_dir / "INCAR")
    if incar.get("IBRION", "-1").split()[0] != "-1":
        raise ValueError(f"{method.name}: 声子位移任务必须设置 IBRION = -1")
    try:
        nsw = int(float(incar.get("NSW", "0").split()[0]))
    except ValueError as exc:
        raise ValueError(f"{method.name}: 无法解析 INCAR 中的 NSW") from exc
    if nsw != 0:
        raise ValueError(f"{method.name}: 声子位移任务必须设置 NSW = 0")
    if incar.get("LREAL", ".FALSE.").upper() not in {".FALSE.", "F", "FALSE"}:
        warnings.append("建议设置 LREAL = .FALSE. 以获得可靠声子力")
    if incar.get("PREC", "").lower() not in {"accurate", "high", "single"}:
        warnings.append("建议设置 PREC = Accurate")
    if "EDIFF" not in incar:
        warnings.append("建议显式设置严格的 EDIFF（例如 1E-8）")
    return warnings


def _parse_force_file(path: Path, n_atoms: int) -> np.ndarray:
    try:
        ET.parse(path)
    except Exception as exc:
        raise IncompleteResultsError(f"vasprun.xml 损坏或未完整写出: {path}: {exc}") from exc
    try:
        from phonopy.interface.vasp import parse_set_of_forces

        parsed = parse_set_of_forces(n_atoms, [str(path)], verbose=False)
    except Exception as exc:
        raise IncompleteResultsError(f"Phonopy 无法读取 VASP 力: {path}: {exc}") from exc
    if not parsed:
        raise IncompleteResultsError(f"Phonopy 未从文件读取到力: {path}")
    forces = np.asarray(parsed[0], dtype=float)
    if forces.shape != (n_atoms, 3) or not np.all(np.isfinite(forces)):
        raise IncompleteResultsError(f"VASP 力数组无效: {path}, shape={forces.shape}")
    return forces


def _parse_energy(path: Path) -> float:
    try:
        from ase.io import read

        atoms = read(str(path), format="vasp-xml", index=-1)
        return float(atoms.get_potential_energy())
    except Exception:
        return float("nan")


@dataclass
class VaspProvider:
    method: VaspMethod
    phonopy_yaml: Path
    workdir: Path
    resultdir: Path
    logdir: Path

    @property
    def jobs_dir(self) -> Path:
        return self.workdir / "jobs"

    def prepare(self) -> None:
        from phonopy import load

        phonon = load(str(self.phonopy_yaml), produce_fc=False)
        cells = phonon.supercells_with_displacements
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        template_files = sorted(path for path in self.method.template_dir.iterdir() if path.is_file())
        for index, cell in enumerate(cells, start=1):
            task = self.jobs_dir / f"disp-{index:04d}"
            task.mkdir(parents=True, exist_ok=True)
            for source in template_files:
                destination = task / source.name
                if not destination.exists() or source.stat().st_mtime_ns > destination.stat().st_mtime_ns:
                    shutil.copy2(source, destination)
            write_vasp_grouped(phonopy_to_ase(cell), task / "POSCAR", task / "atom-map.json")
            (task / "README.txt").write_text(
                "Static VASP force calculation for a Phonopy displaced supercell. Do not relax this structure.\n",
                encoding="utf-8",
            )
        atomic_write_json(self.workdir / "manifest.json", {
            "type": "vasp",
            "template_dir": str(self.method.template_dir),
            "n_displacements": len(cells),
            "jobs_dir": str(self.jobs_dir),
        })

    def readiness(self) -> tuple[int, int]:
        tasks = sorted(path for path in self.jobs_dir.glob("disp-*") if path.is_dir())
        ready = sum((path / "vasprun.xml").is_file() for path in tasks)
        return ready, len(tasks)

    def run_or_submit(self, *, wait: bool = False) -> str:
        self.prepare()
        ready, total = self.readiness()
        if ready == total and total:
            return "ready"
        result = submit_vasp(self.method, self.workdir, wait=wait)
        return str(result["status"])

    def collect(self) -> tuple[np.ndarray, np.ndarray | None]:
        from phonopy import load

        phonon = load(str(self.phonopy_yaml), produce_fc=False)
        cells = phonon.supercells_with_displacements
        missing = [
            self.jobs_dir / f"disp-{index:04d}" / "vasprun.xml"
            for index in range(1, len(cells) + 1)
            if not (self.jobs_dir / f"disp-{index:04d}" / "vasprun.xml").is_file()
        ]
        if missing:
            raise IncompleteResultsError(
                f"VASP 结果未完成，缺少 {len(missing)}/{len(cells)} 个 vasprun.xml；第一个: {missing[0]}"
            )
        forces: list[np.ndarray] = []
        energies: list[float] = []
        reordered = 0
        for index, cell in enumerate(cells, start=1):
            task = self.jobs_dir / f"disp-{index:04d}"
            raw = _parse_force_file(task / "vasprun.xml", len(cell))
            mapping = load_json(task / "atom-map.json")
            written_to_canonical = np.asarray(mapping["written_to_canonical"], dtype=int)
            if sorted(written_to_canonical.tolist()) != list(range(len(cell))):
                raise IncompleteResultsError(f"原子顺序映射不是双射: {task / 'atom-map.json'}")
            canonical = np.empty_like(raw)
            canonical[written_to_canonical] = raw
            if not np.array_equal(written_to_canonical, np.arange(len(cell))):
                reordered += 1
            forces.append(canonical)
            energies.append(_parse_energy(task / "vasprun.xml"))
        force_array = np.asarray(forces, dtype=float)
        energy_array = np.asarray(energies, dtype=float)
        self.resultdir.mkdir(parents=True, exist_ok=True)
        np.save(self.resultdir / "forces.npy", force_array)
        if np.any(np.isfinite(energy_array)):
            np.save(self.resultdir / "energies.npy", energy_array)
            returned_energies: np.ndarray | None = energy_array
        else:
            returned_energies = None
        atomic_write_json(self.resultdir / "force-provider.json", {
            "type": "vasp",
            "n_displacements": len(forces),
            "n_reordered": reordered,
            "force_source": "vasprun.xml parsed by phonopy.interface.vasp",
        })
        return force_array, returned_energies

# Single-Structure Configuration

Use this reference when creating, reviewing, or modifying an ordinary
`config.yaml`. The source-of-truth user reference is
`docs/config-reference.zh-CN.md` under the injected project root. If this file
and the installed program disagree, inspect `src/phonon_kit/config.py` and follow
the installed version.

## Contents

- [Selection checklist](#selection-checklist)
- [Schema and paths](#schema-and-paths)
- [Structure and optional relaxation](#structure-and-optional-relaxation)
- [Finite-displacement settings](#finite-displacement-settings)
- [DeepMD methods](#deepmd-methods)
- [VASP and DPDispatcher](#vasp-and-dpdispatcher)
- [Validation and cost](#validation-and-cost)

## Selection checklist

Before writing YAML, establish:

1. The input structure and whether it is already relaxed at the intended pressure.
2. Whether the result is DPA-only, VASP-only, or a same-displacement comparison.
3. The finite-displacement supercell and a plan to test its convergence.
4. The model path, output head when applicable, and `cpu` or `cuda:N` device.
5. The temperature range and q mesh required for vibrational thermodynamics.
6. For VASP, the static-force template, POTCAR order, remote executor, and whether
   the user intends submission now or only configuration preparation.

Do not infer a production-quality supercell, k mesh, ENCUT, or relaxation pressure
from composition alone. Offer a starting point and label it as such when no
convergence evidence exists.

## Schema and paths

The top-level schema is strict:

```yaml
schema_version: 1
project:
  name: sio2
  runs_dir: runs
structure:
  file: inputs/POSCAR
relaxation: {}
phonon: {}
methods: {}
```

- Only schema version `1` is supported.
- Relative paths resolve against the YAML directory, never the shell directory.
- `project.name` becomes the numbered run prefix, such as `sio2-001`.
- `project.runs_dir` defaults to `runs`.
- Unknown fields and quoted pseudo-booleans fail validation.
- Input files, including model and VASP template contents, participate in the
  configuration fingerprint.
- Units are Å, eV, eV/Å, GPa, THz, and K unless a result column says otherwise.

At least one method must be enabled. Multiple DeepMD methods may be enabled, but
only one VASP method is currently allowed. Command-line `--only` selects a subset
of enabled methods and becomes part of run identity.

## Structure and optional relaxation

`structure.file` is read by ASE; if it contains multiple frames, the last frame
is used. Require atoms and a nonsingular three-dimensional periodic cell. The
workflow writes a canonical POSCAR into the run rather than altering the input.

Optional relaxation defaults:

```yaml
relaxation:
  enabled: false
  method: dpa4
  variable_cell: true
  pressure_gpa: 0.0
  fmax_ev_angstrom: 0.01
  max_steps: 1000
  trajectory_interval: 10
```

- `method` must name an enabled DeepMD method; VASP relaxation is not supported
  in this ordinary workflow.
- `variable_cell: true` uses `FrechetCellFilter + LBFGS` and positive pressure
  means compression. `false` optimizes atoms with a fixed cell.
- `fmax_ev_angstrom` must be positive; `max_steps` and
  `trajectory_interval` must be positive integers.
- The sparse trajectory includes initial, interval, and final frames.
- Failure to converge stops before displacement generation.
- All selected DPA and DFT force methods share this one relaxed reference, which
  is necessary for direct method comparison.

If the structure was already carefully relaxed by a chosen reference method,
leaving relaxation disabled avoids silently changing the comparison geometry.

## Finite-displacement settings

```yaml
phonon:
  supercell: [2, 2, 2]
  displacement_angstrom: 0.01
  primitive: auto
  symmetry_tolerance: 1.0e-5
  band:
    path: auto
    points_per_segment: 101
  mesh: [30, 30, 30]
  thermal:
    temperature_min_k: 0
    temperature_max_k: 1000
    temperature_step_k: 10
    imaginary_policy: exclude
    significant_imaginary_thz: -0.1
```

- `supercell` and `mesh` are three positive integers; general 3x3 matrices are
  not supported.
- `displacement_angstrom` and `symmetry_tolerance` must be positive.
- `primitive` and `band.path` currently accept only `auto`; SeeK-path defines the
  common high-symmetry path.
- `points_per_segment >= 2` changes plot resolution, not force-call count.
- The mesh affects DOS and thermodynamic integration, not displacement count.
- Temperature begins at or above 0 K, has a positive step, and ends no lower than
  it begins.
- `imaginary_policy` currently accepts only `exclude`.
- `significant_imaginary_thz` must be nonpositive and controls warning severity;
  all negative frequencies are excluded at 0 THz regardless of this threshold.

Treat `0.01 Å`, `[2,2,2]`, and `[30,30,30]` as defaults, not universal convergence
proof. Test supercell size for force-constant range and mesh density for thermal
integrals. Do not loosen symmetry tolerance merely to reduce displacement count.

## DeepMD methods

```yaml
methods:
  dpa4:
    type: deepmd
    model: builtin:dpa4
    device: cuda:0
    head: null
    enabled: true
```

- `model` accepts `builtin:dpa4` or a DeepMD ASE-loadable `.pt2`, `.pt`, `.pth`,
  or `.pb` file.
- `builtin:dpa4` resolves from installed package data and its checksum, not from
  the case directory. It is a frozen GPU `.pt2` artifact.
- `device` defaults to `cuda:0`; supported syntax is `cpu` or `cuda:N`, subject to
  the model/runtime actually supporting it.
- A `.pt2` head is frozen: omit `head` or set it to `null`. A checkpoint format
  may accept a named head only when the model does.
- `enabled` defaults to true.
- The file suffix is only preliminary validation. Real validation checks the type
  map and finite energy, forces, and stress on the actual structure.

Use distinct method names for pretrained and fine-tuned models. Do not assume a
model is accurate because it loads or because the calculation completes.

## VASP and DPDispatcher

An ordinary VASP method is finite-displacement static-force calculation only:

```yaml
methods:
  dft:
    type: vasp
    template_dir: inputs/vasp
    enabled: true
    executor:
      type: dpdispatcher
      machine: inputs/dispatcher/machine.json
      resources: inputs/dispatcher/resources.json
      command: "mpirun -n 16 vasp_std"
      clean_remote_after_success: false
```

The template must include `INCAR`, `KPOINTS`, and `POTCAR`. Require `IBRION=-1`
and `NSW=0`. The program checks obvious mistakes but does not choose or modify
ENCUT, k points, smearing, spin, precision, or convergence settings. POTCAR
species order must match the structure's first-occurrence species order.

`executor.type` is currently `dpdispatcher`. Machine and resource JSON paths are
relative to the YAML. The command must load the remote environment and execute
VASP. Keep remote cleanup false until recovery is trusted. Never print credentials
from machine JSON or add POTCAR/credential files to version control.

The program stores an explicit atom mapping for each displaced VASP task and
restores canonical Phonopy order while parsing `vasprun.xml`. A present OUTCAR is
not enough: collection requires a complete, parseable `vasprun.xml`.

## Validation and cost

With explicit authority, validate after changing a structure, model, head,
device, VASP template, POTCAR, or dispatcher file:

```text
ph validate config.yaml
ph validate config.yaml --only dpa4 dft
```

Validation checks the schema, structure, model type map, real finite inference,
VASP static settings, POTCAR ordering, and dispatcher files. It does not prove
scientific convergence or model accuracy.

Use Phonopy's generated displacement count as the number of force evaluations per
method. The supercell atom count multiplies every evaluation's cost. Multiple
DeepMD methods share geometries but perform separate force calls; one VASP method
adds one remote calculation per displaced supercell.

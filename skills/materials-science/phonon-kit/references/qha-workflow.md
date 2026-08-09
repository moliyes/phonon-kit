# Multiphase QHA Workflow

Use this reference for `qha.yaml`, `ph qha` commands, volume-grid planning, fixed-
volume relaxation, and P-T phase diagrams. Consult
`docs/qha-config-reference.zh-CN.md` and `src/phonon_kit/qha_config.py` under the
injected project root when installed-version behavior must be confirmed.

## Contents

- [Physical contract](#physical-contract)
- [Configuration inheritance](#configuration-inheritance)
- [Volume relaxation and phonons](#volume-relaxation-and-phonons)
- [QHA grid and normalization](#qha-grid-and-normalization)
- [DeepMD and VASP stages](#deepmd-and-vasp-stages)
- [Planning and acceptance](#planning-and-acceptance)

## Physical contract

This workflow compares two or more solid polymorphs with the same reduced
composition using

`G(P,T) = min_V [U(V) + F_vib(V,T) + P V]`.

It does not implement variable-composition hulls, chemical potentials, electronic
free energy, explicit anharmonicity, melting, liquids, or kinetic accessibility.
Each phase input must be a real candidate topology; generated placeholders from
`ph qha init` are parseable scaffolding, not scientific phase structures.

Different phases may contain different atom counts. The workflow canonicalizes a
primitive cell per phase, performs QHA in primitive-cell units, then normalizes
cross-phase Gibbs energies to `eV/reduced formula unit`. Confirm all reduced
formulas match before spending compute.

Each force method owns an independent phase-volume series. Never reuse DPA-relaxed
geometries as DFT-relaxed geometries or vice versa when comparing phase diagrams.

## Configuration inheritance

Core shape:

```yaml
schema_version: 1
project:
  name: sio2_qha
  runs_dir: runs
phases:
  quartz:
    structure: inputs/phases/quartz/POSCAR
    volume_ratios: [0.88, 0.91, 0.94, 0.97, 1.00, 1.03, 1.06]
    phonon:
      supercell: [2, 2, 2]
volume_grid:
  ratios: [0.88, 0.91, 0.94, 0.97, 1.00, 1.03, 1.06]
volume_relaxation: {}
phonon: {}
qha: {}
methods: {}
```

- At least two phases are required and all must have the same reduced formula.
- `phases.<name>.structure` is required. `label` and Matplotlib-compatible `color`
  are optional presentation fields.
- Each phase inherits `volume_grid.ratios` unless it supplies
  `volume_ratios`. Require at least five unique, strictly increasing, positive
  points. Seven to eleven is a practical starting range, not a guarantee.
- Each phase inherits global `phonon` fields and may override them independently.
  Different cells commonly need different supercells and meshes.
- All paths resolve relative to `qha.yaml`; the schema rejects unknown fields.

Ratios mean `V/Vref`; initial lattice vectors scale by `ratio^(1/3)`. The input
volume centers the grid. It need not first be fully variable-cell relaxed, but a
bad center can cause every fitted minimum to approach or leave a grid boundary.

## Volume relaxation and phonons

Defaults:

```yaml
volume_relaxation:
  fmax_ev_angstrom: 0.01
  max_deviatoric_stress_gpa: 0.1
  max_steps: 1000
  trajectory_interval: 10
  volume_tolerance_relative: 1.0e-5
phonon:
  supercell: [2, 2, 2]
  displacement_angstrom: 0.01
  symmetry_tolerance: 1.0e-5
  mesh: [30, 30, 30]
  significant_imaginary_thz: -0.1
```

DPA uses `FrechetCellFilter(constant_volume=True) + LBFGS` to relax atomic
coordinates and cell shape. VASP uses `ISIF=4`. Acceptance requires target-volume
tolerance, maximum atomic force, and maximum deviatoric stress; hydrostatic stress
need not be zero at a constrained volume.

The DPA trajectory retains initial, interval, and final frames. An unconverged
volume point must not proceed to static energy or phonons. Fixed-volume relaxation
may alter symmetry, so the actual number of displaced cells can differ from the
plan estimate.

The phonon displacement, tolerance, and mesh constraints match the ordinary
workflow. Converge supercell and q mesh separately per phase. Do not compare a
small primitive phase with a large conventional phase using superficially equal
supercell multipliers without checking real-space size and atom count.

## QHA grid and normalization

```yaml
qha:
  eos: vinet
  temperature:
    min_k: 0
    max_k: 1000
    step_k: 10
  pressure:
    min_gpa: 0
    max_gpa: 20
    step_gpa: 0.25
  imaginary_policy: exclude
```

- `eos` accepts `vinet`, `birch_murnaghan`, or `murnaghan`.
- Temperature begins at or above 0 K; temperature and pressure steps are positive
  and must exactly divide their configured ranges.
- The implementation computes one additional temperature internally for a
  numerical derivative but only reports through `max_k`.
- `imaginary_policy` currently accepts only `exclude`; Phonopy uses a 0 THz
  cutoff. Significant imaginary modes continue through analysis but mark the
  phase and diagram unstable/diagnostic.
- A phase's fitted equilibrium volume must lie within its sampled interval. If it
  lies outside, that phase's P-T point is invalid. Because all phases must be valid
  to select a winner, the phase-map point is no-data.
- A fitted minimum near a volume boundary remains computable but is marked
  `near_volume_edge`; expand the grid before trusting a boundary there.

Static energies, volumes, phonon free energies, and `PV` must share the same cell
basis within each fit. Cross-phase comparison then uses the program's
`eV/formula unit` arrays. Do not manually compare raw `e-v.dat` energies between
differently sized primitive cells without applying their formula-unit counts.

## DeepMD and VASP stages

DeepMD fields match the ordinary workflow: model, device, optional supported head,
and enabled flag. `builtin:dpa4` is GPU-only and cannot set a head. For every
method, phase, and volume, DeepMD performs fixed-volume relaxation, one static
energy evaluation, every finite-displacement force evaluation, and QHA analysis.
Checkpointing skips completed volumes and displacement forces on resume.

QHA VASP requires three templates, each containing `INCAR`, `KPOINTS`, and
`POTCAR`:

```yaml
methods:
  dft:
    type: vasp
    templates:
      volume_relax: inputs/vasp/volume_relax
      static: inputs/vasp/static
      phonon: inputs/vasp/phonon
    phase_templates: {}
    executor: {}
```

- `volume_relax`: `ISIF=4`, `IBRION` in 1/2/3, and `NSW>0`.
- `static`: `IBRION=-1`, `NSW=0`; its final `e_0_energy` supplies U(V).
- `phonon`: `IBRION=-1`, `NSW=0`; it supplies displaced-supercell forces.
- POTCAR content must be consistent across stages and phases.
- `phase_templates.<phase>` may override all three directories for a phase that
  needs a different KPOINTS or cell-specific template.
- The software neither generates POTCAR nor modifies scientific INCAR settings.

The remote stages are ordered: all fixed-volume relaxations, then static energies
and phonon-force jobs. Running without `--wait` submits the available stage and
returns. A later `resume` reconstructs the same DPDispatcher submission and may
submit the next stage; never treat it as a passive poll. HTTP 5xx transport errors
become retryable state so the same run can resume after connectivity recovers.

## Planning and acceptance

`ph qha plan qha.yaml` is read-only: it normalizes structures and estimates each
phase's reference/target volumes, symmetry-reduced displacements, DeepMD work,
and VASP relaxation/static/phonon task counts. Run it before validation or
submission and call out large supercells, broad grids, and asymmetric pressure
coverage.

Before running, require:

1. Same reduced composition and intentional phase identities.
2. At least five volume points per phase with likely minima inside the grid.
3. Per-phase supercell and q-mesh rationale.
4. A pressure range supported by compression-side volumes.
5. Real model inference or valid three-template VASP setup.
6. User acknowledgement of expected compute/submission scale.

After completion, require every expected volume point, per-phase QHA grid, and
phase diagram. Report invalid and near-edge grid counts, significant imaginary
modes, the formula-unit normalization, and phase-boundary interpolation. A clean
workflow status alone is not scientific acceptance.

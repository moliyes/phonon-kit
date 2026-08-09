# Results and Recovery

Use this reference to interpret `ph status`, inspect run state, diagnose
DeepMD/VASP interruptions, collect returned calculations, or regenerate plots.
Start from CLI summaries and small JSON summaries; inspect large logs or arrays
only when an invariant is unresolved.

## Contents

- [Run identity and safe inspection](#run-identity-and-safe-inspection)
- [Ordinary phonon states and results](#ordinary-phonon-states-and-results)
- [QHA states and results](#qha-states-and-results)
- [VASP and dispatcher recovery](#vasp-and-dispatcher-recovery)
- [Failure playbooks](#failure-playbooks)
- [Completion checks](#completion-checks)

## Run identity and safe inspection

Both workflows use numbered directories and atomically written state:

```text
runs/<project>-001/
├── config.resolved.yaml
├── state.json
├── validation.json
├── work/
├── results/
└── logs/
```

Inspect with the correct CLI first:

```text
ph status config.yaml
ph status runs/project-001
ph qha status qha.yaml
ph qha status runs/project-001
```

Status accepts either a configuration file or a concrete run directory. A config
selects the latest numbered run by project prefix; a directory reads that exact
state. Use the directory form when the user identifies a historical run.

Then read `config.resolved.yaml`, followed by `state.json` only for details not in
status. Do not recursively read all job logs or load `.npy/.npz` merely to answer
a phase/status question. Stop once summary invariants explain the state.

For a completed QHA run, do not begin with `find -maxdepth`: phase result summaries
are deeper than three levels and a shallow listing can falsely suggest an empty
results directory. After `ph qha status`, read these deterministic small files:

```text
results/<method>/phases/<phase>/summary.json
results/<method>/phase_diagram/summary.json
```

They contain completion, volume counts, imaginary warnings, and coverage counts.
Avoid the full `state.json` per-volume mapping unless one summary is missing or
contradictory.

Identity rules:

- First run creates `name-001`.
- Same config and incomplete state resumes that run.
- Same config and completed state returns its existing results.
- Changed config after completion creates the next number.
- Changed config while the newest run is incomplete is rejected unless the user
  explicitly requests an independent `--new` run.
- A different `--only` set is rejected for an existing matching run; preserve the
  method list or explicitly create a new run.

Config fingerprints include resolved settings and relevant input content. Never
edit `state.json`, `config.resolved.yaml`, or delete outputs to force a match.

## Ordinary phonon states and results

Top-level status commonly becomes:

- `created`: run state initialized.
- `running`: normalization, relaxation, displacement force calls, or analysis.
- `waiting_dft`: a VASP method is submitted or awaiting complete results.
- `failed`: preprocessing or one selected method failed.
- `completed`: every selected method completed; comparison may be independent.

Method states include `pending`, `running`, `submitted`, `waiting`, `analyzing`,
`failed`, and `completed`. Status reports the symmetry-reduced displacement count
and VASP readiness as `ready/total`.

Work artifacts:

- `work/canonical/POSCAR` and optional `POSCAR-relaxed` define the shared geometry.
- `work/displacements/phonopy_disp.yaml` defines the one canonical displacement
  set reused by every method.
- `work/methods/<method>/checkpoints/disp-xxxx.npz` stores independent DeepMD
  results; resume recomputes only missing or invalid checkpoints.
- VASP tasks live under `work/methods/<method>/jobs/disp-xxxx/` and contain an
  explicit `atom-map.json`.

Completed method results normally include:

- `forces.npy`, optionally `energies.npy`, and `force-provider.json`.
- `FORCE_SETS`, `force_constants.hdf5`, and `phonopy_params.yaml`.
- `band.yaml`, `band_data.npz`, `band_path.json`, and total DOS.
- thermal YAML/CSV, mesh frequencies, three PNG plots, and `summary.json`.
- With DFT plus DPA, `results/comparison/` contains metrics and force/frequency
  comparison plots.

`plot` re-analyzes existing force arrays and rewrites plots/results. It does not
recompute forces, but it is still a mutating action and requires an explicit user
request.

## QHA states and results

QHA top-level state normally moves through `created`, `running`, `waiting`,
`failed`, and `completed`. Method status adds a stage such as:

- `volume_phonons` for local DeepMD volume work.
- `volume_relax` for the first VASP batch.
- `static`, `phonon`, or `static_and_phonon` for later VASP stages.
- `phase_diagram` or `analyzing_phase_diagram` for final analysis.
- `retryable` when a recognized external HTTP/service failure can be retried.
- `completed` with `stage=completed` after its phase diagram exists.

Each method has phase states, and each phase has deterministic volume IDs such as
`v000-r0.880000`. Status reports completed/total volumes. DeepMD volume states may
move through `relaxing`, `static`, `phonon`, and `completed`; VASP also records
stage-specific readiness.

QHA work is organized as:

```text
work/methods/<method>/phases/<phase>/
├── reference/POSCAR-primitive
└── volumes/<volume-id>/{relaxation,static,phonon}/
```

Per-volume results contain forces, Phonopy artifacts, and
`volume_summary.json`. Per-phase results include:

- `volume_points.csv`: target/actual volume, cell/primitive/formula-unit energy,
  relaxation quality, minimum frequency, imaginary fraction, and validity.
- `e-v.dat`: primitive-cell static E-V input used by Phonopy-QHA.
- `qha_grid.csv/npz`: P-T Gibbs energy, equilibrium volume, bulk modulus,
  validity, and near-edge flags.
- `static_eos.png`, `qha_thermodynamics.png`, and `summary.json`.

Method phase-diagram results include `phase_map.csv/npz`, interpolated
`phase_boundaries.csv`, three phase/Gibbs PNGs, and `summary.json`. With multiple
completed methods, `results/comparison/` reports stable-label and boundary
agreement; it does not align absolute energy zeroes between methods.

`ph qha plot` uses existing per-phase `qha_grid.npz` files. It cannot repair
missing forces, volume points, or invalid EOS data.

## VASP and dispatcher recovery

For ordinary phonons, `run` prepares/submits all displaced cells and usually
returns without waiting. `resume` queries the same submission, downloads complete
results, parses forces, and analyzes once all XML files are ready.

For QHA, VASP is dependency staged:

1. Fixed-volume relaxation.
2. Static energies plus finite-displacement phonon forces from relaxed structures.
3. Per-volume QHA, per-phase fits, and phase diagram.

A QHA `resume` may submit the next stage, so only run it with explicit authority.
Use `--wait` only when the user explicitly wants continuous remote waiting.

Manual collection is appropriate only when the user has already returned complete
`vasprun.xml` files to the exact prepared job directories. `collect` parses and
analyzes but does not submit missing jobs. It changes state and results, so require
an explicit request.

Treat these categories separately:

- `submitted` or `waiting` with incomplete readiness: not yet a calculation
  failure.
- HTTP 5xx, upload failure, or QHA `retryable`: transport/service failure; retain
  local submission state and resume after connectivity returns. This does not by
  itself reveal whether the remote service received, queued, or ran the jobs.
- Truncated/unparseable `vasprun.xml`: result transfer or incomplete VASP output.
- Complete XML with electronic/ionic failure: scientific calculation failure;
  inspect the specific job and VASP settings before resubmission.

Never assume OUTCAR existence means completion. Do not print machine credentials,
API tokens, or POTCAR content while diagnosing.

## Failure playbooks

### No run found

Confirm the correct YAML and `project.runs_dir`, then inspect the numbered folders.
Do not start a run unless asked. A config path may point to another case with the
same project name but a different runs directory.

### Configuration changed

Compare the editable YAML and the run's resolved snapshot. If the run is
incomplete, restore the exact original inputs to resume or, only when requested,
start a separate `--new` run. If the old run completed, ordinary `run` with the
changed config creates a new numbered run automatically.

### DeepMD interruption

Check status, method/volume checkpoints, and the latest method log. Resume only
with authority; completed displacement checkpoints and completed QHA volumes are
reused. If a model/device changed, do not mix checkpoints—use a new run.

### Relaxation failure

Read the saved relaxation summary: convergence flag, force, stress when relevant,
step count, and volume error. Diagnose the starting structure, model domain,
target pressure/volume, and thresholds. Do not manufacture downstream phonons
from an unconverged geometry.

### VASP waiting or HTTP 5xx

Report stage, readiness, total tasks, and latest error. For a 5xx with 0/N ready,
state that submission/recovery failed and no parseable result was collected. Stop
there when this answers the question: do not read manifests or dispatcher state
merely for reassurance. Do not claim that VASP started or did not start unless an
authoritative remote status proves it; local `n_submitted`, serialization flags,
and undocumented numeric job-state values are insufficient. Do not call it a
VASP physics failure and do not retry without explicit user authority.

### Analysis or plotting failure

Confirm complete force arrays/volume grids and inspect the analysis traceback.
Fix configuration or code only if requested. Replot cannot fill missing upstream
data. Preserve numeric results while repairing presentation.

### QHA grey or suspicious regions

Read each phase summary, `volume_points.csv`, and `qha_grid.csv`. Report invalid
and near-edge counts by phase. Extend only the deficient compression or expansion
side based on fitted equilibrium volumes, then create an explicitly requested new
run; do not overwrite the completed grid.

Keep the two coverage states distinct: `invalid_grid_points > 0` means an
equilibrium minimum fell outside the sampled interval and is masked as no-data;
`near_volume_edge_grid_points > 0` with zero invalid points means the minimum is
still inside but close to an edge. Do not call the latter extrapolation.

## Completion checks

For ordinary phonons:

1. Top-level status is completed.
2. Every selected method is completed and has exactly the displacement count
   reported by the canonical manifest.
3. Force constants, band, DOS, thermal files, PNGs, and method summary exist.
4. If multiple reference types completed, comparison outputs exist or their
   independent failure is disclosed.
5. Minimum frequency and stability flags are reported.

For QHA:

1. Top-level and every selected method are completed.
2. Every phase reports all configured volume points completed.
3. Per-volume and per-phase summaries exist and normalization is consistent.
4. Every phase has a QHA grid; the method has phase map, boundaries, plots, and
   summary.
5. Invalid and near-edge grid counts and imaginary-mode flags are explicitly
   reported. `completed` does not override these scientific warnings.

Retain an entire run directory for exact resume/reproduction. For archival results
alone, retain the editable YAML, all inputs, resolved config, state, logs relevant
to provenance, and complete `results/` rather than isolated PNGs.

---
name: phonon-kit
description: Run Phonopy phonons and DeepMD/VASP QHA phase diagrams.
version: 0.1.0
author: moliyes, Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [materials-science, phonons, Phonopy, DeepMD, VASP, QHA, phase-diagram]
    category: materials-science
    related_skills: [calypso-dp-search]
    requires_tools: [terminal]
    config:
      - key: phonon_kit.project_root
        description: Path to the phonon-kit source checkout
        default: ~/phonon-kit
        prompt: phonon-kit project directory
---

# Phonon Kit Skill

Operate the `ph` CLI for resumable finite-displacement phonons, vibrational
thermodynamics, DeepMD/VASP comparison, multiphase QHA, and P-T phase diagrams.
Separate workflow completion from scientific validity: a completed calculation
with significant imaginary modes or inadequate volume coverage remains diagnostic.

## When to Use

- Create or change a single-structure Phonopy calculation using DeepMD/DPA or VASP.
- Create or change a multiphase QHA and P-T phase-diagram calculation.
- Validate models, structures, supercells, VASP templates, CUDA, or DPDispatcher.
- Inspect status, failures, checkpoints, VASP collection, plots, phonon spectra,
  thermal properties, QHA volume points, Gibbs grids, or phase boundaries.
- Explain imaginary modes, convergence, model agreement, QHA validity, or handoff
  from a CALYPSO structure search.
- Do not use for VASP-DFPT, force-field training, variable-composition convex
  hulls, explicit anharmonicity, melting, or chemical-potential phase diagrams.

## Prerequisites

1. Read the injected `phonon_kit.project_root` Skill setting. If it is absent or
   invalid, use `search_files(target='files')` to locate a checkout whose
   `pyproject.toml` names `phonon-kit`; ask the user if multiple checkouts match.
2. Check the CLI with
   `terminal(command="command -v ph && ph --version")`. Do not assume the active
   shell's `python` owns `ph`; its launcher must use an environment compatible
   with DeepMD when a DeepMD method is requested.
3. If `ph` is absent, read `install.sh` and the project README. Install or repair
   it only after the user explicitly asks.
4. Require a compatible NVIDIA runtime for `builtin:dpa4`; the bundled frozen
   `.pt2` model is GPU-only. VASP workflows additionally require valid templates,
   POTCAR, DPDispatcher configuration, credentials, and a working remote executor.

## How to Run

Use Hermes tools as the interaction surface:

- Read YAML, JSON, CSV, and logs with `read_file`; locate artifacts with
  `search_files`.
- Change YAML with `patch`, preserving comments and unrelated fields. Never edit
  a run's `config.resolved.yaml` or `state.json` to alter workflow identity.
- Use `terminal(command="...", workdir="...")` for CLI checks.
- Treat `ph status`, `ph qha status`, and `ph qha plan` as read-only diagnostics.
- Run `init`, modify configuration, perform real-inference `validate`, invoke
  `run` or `resume`, collect VASP results, or regenerate plots only when the user
  explicitly requests that action. A request to diagnose is not authority to
  submit, resume, collect, replot, or infer with a model.
- Start DPA runs, QHA runs, and any command using `--wait` with
  `background=true, notify_on_complete=true`. Track it with `process`; do not
  launch another process for the same run while one may still be active.
- Remember that `resume` can continue local DPA calculations or submit the next
  VASP stage. It is not a read-only status check.

## Quick Reference

| Intent | Hermes invocation |
|---|---|
| Create one-structure case | `terminal(command="ph init <case-dir>")` |
| Validate one-structure case | `terminal(command="ph validate <config.yaml> [--only ...]")` |
| Run one-structure case | `terminal(command="ph run <config.yaml> [--only ...] [--new] [--wait]", background=true, notify_on_complete=true)` |
| Resume one-structure case | `terminal(command="ph resume <config.yaml> [--wait]", background=true, notify_on_complete=true)` |
| Collect local VASP results | `terminal(command="ph collect <config.yaml>")` |
| Inspect one-structure status | `terminal(command="ph status <config-or-run-dir>")` |
| Replot one-structure results | `terminal(command="ph plot <config.yaml>")` |
| Create QHA case | `terminal(command="ph qha init <case-dir> --phases <phase...>")` |
| Estimate QHA work | `terminal(command="ph qha plan <qha.yaml>")` |
| Validate QHA case | `terminal(command="ph qha validate <qha.yaml> [--only ...]")` |
| Run QHA | `terminal(command="ph qha run <qha.yaml> [--only ...] [--new] [--wait]", background=true, notify_on_complete=true)` |
| Resume QHA | `terminal(command="ph qha resume <qha.yaml> [--wait]", background=true, notify_on_complete=true)` |
| Collect QHA VASP results | `terminal(command="ph qha collect <qha.yaml>")` |
| Inspect QHA status | `terminal(command="ph qha status <config-or-run-dir>")` |
| Replot QHA results | `terminal(command="ph qha plot <qha.yaml>")` |

Both status commands accept a configuration file or a concrete run directory.
Quote every path that may contain spaces.

Load references only when needed:

- Before creating or changing a single-structure YAML, load
  `skill_view("phonon-kit", "references/configuration.md")`.
- For QHA configuration, work estimates, VASP stages, or phase-map construction,
  load `skill_view("phonon-kit", "references/qha-workflow.md")`.
- For status, errors, checkpoints, DFT transfer, recovery, collection, or replot,
  load `skill_view("phonon-kit", "references/results-and-recovery.md")`.
- For imaginary modes, thermodynamics, convergence, phase-boundary claims,
  method comparison, or CALYPSO handoff, load
  `skill_view("phonon-kit", "references/scientific-interpretation.md")`.

## Procedure

1. **Classify the workflow.** Use ordinary `ph` commands for one structure's
   harmonic phonons; use `ph qha` only for two or more same-composition phases
   over multiple volumes. Never pass `config.yaml` to QHA or `qha.yaml` to the
   ordinary workflow.
2. **Classify the request.** Separate read-only diagnosis, scientific advice,
   initialization, configuration change, validation, run, resume, collection,
   and replotting. Obtain explicit authority for every mutating or costly action.
3. **Resolve the case.** Resolve all relative paths from the YAML directory, not
   the shell working directory. Identify selected methods, input structures,
   expected run prefix, model/device, and whether VASP submission is in scope.
4. **Inspect before changing.** Run the correct status command and inspect the
   latest `state.json` plus `config.resolved.yaml` when a run exists. Do not alter
   old results. A changed incomplete run requires the original inputs or an
   explicitly requested `--new` run.
5. **Configure scientifically.** For a single structure, establish relaxation,
   supercell, displacement, mesh, temperature range, and force providers. For
   QHA, additionally establish real phase structures, common reduced formula,
   volume coverage, per-phase supercells, EOS, and P-T grid. Do not silently
   invent a scientific parameter whose choice controls the conclusion.
6. **Estimate before QHA.** Run `ph qha plan` and report phases, volume points,
   expected displaced supercells, local model evaluations, and VASP task counts.
   Warn that fixed-volume relaxation can change symmetry and the final count.
7. **Validate with authority.** After initialization or changes to structures,
   models, device, templates, POTCAR, or dispatcher files, run the corresponding
   validate command if the user asked to validate or proceed with computation.
   Validation includes real DeepMD energy, force, and stress inference.
8. **Start once.** When explicitly requested, launch the selected run/resume
   command in the background. Preserve the exact `--only` method set across
   recovery. Use `--new` only for an explicitly requested independent rerun.
9. **Diagnose from summaries first.** Use CLI status, resolved configuration, and
   result `summary.json` files before reading large logs or arrays. For VASP,
   distinguish submitted/waiting/retryable transfer state from a failed physical
   calculation. If status already reports a 5xx, stage, and `ready/total`, stop:
   do not inspect manifests or dispatcher internals unless the user asks or a
   contradiction remains. A 5xx with 0/N proves only that no parseable result was
   collected; it does not prove whether the remote service received or ran jobs.
   For QHA, run status first, then read the known paths
   `results/<method>/phases/<phase>/summary.json` and
   `results/<method>/phase_diagram/summary.json`. Do not use a shallow directory
   listing to decide results are absent and do not read the full per-volume state
   mapping when these summaries answer the question.
10. **Verify numerical products.** For ordinary phonons, require every selected
    method completed and check forces, force constants, band, DOS, thermal data,
    plots, and comparison when applicable. For QHA, require all selected phases
    and volumes completed, then check per-phase QHA grids and the phase diagram.
11. **Report scientific qualifiers.** Always report minimum frequencies or
    imaginary-mode warnings, volume coverage and invalid/edge points for QHA,
    normalization basis, method/model, and convergence checks still outstanding.
    Map flags literally: phase-level `imaginary_modes_excluded=true` means at
    least one sampled volume had excluded significant imaginary modes, not that
    the equilibrium structure is proven unstable. With `invalid_grid_points=0`,
    say explicitly that no fitted minimum was outside the grid; near-edge points
    remain in range and are coverage warnings, not extrapolation. Use “triggered
    the imaginary-mode warning” and “no significant imaginary mode found on this
    sampled mesh,” not “completely unstable,” “dynamically stable,” “clean,” or
    “only numerical noise” unless independent evidence establishes that claim.

## Pitfalls

- `builtin:dpa4` resolves from package data; it is not relative to the case. It
  cannot use CPU and cannot set a non-null `head`.
- `.pt2` has a frozen head. Other `.pt`, `.pth`, and `.pb` files may use `head`
  only when the model supports it; confirm by real inference.
- The ordinary optional relaxation is shared by all selected force methods and
  must use an enabled DeepMD method. QHA methods instead relax every phase-volume
  series independently on their own potential-energy surface.
- `--only` is part of run identity. Reusing the same configuration with a
  different method subset is rejected unless the user explicitly requests a new
  run.
- An ordinary DFT run needs one static-force template. QHA DFT needs three
  distinct template roles: fixed-volume relaxation, static energy, and phonon
  forces. The software never generates POTCAR or repairs INCAR automatically.
- An HTTP 5xx or `retryable` QHA dispatcher error is usually safe to resume after
  transport recovery; do not label it a phonon or VASP convergence failure
  without examining the relevant result. Do not infer remote job state from an
  upload error, local serialization flag, or undocumented numeric job-state code.
- Negative frequencies are excluded from thermal integration at 0 THz. This can
  produce finite curves and a completed status even when the phase is not
  dynamically stable. Frequency magnitude alone cannot identify a genuine
  instability, numerical artifact, or model failure; report plausible causes and
  required checks rather than assigning one cause as fact.
- A QHA phase map is invalid wherever any candidate phase lacks a valid Gibbs
  minimum. Near-edge points are warnings, and extrapolated minima are no-data,
  not stable-phase predictions. `near_volume_edge` means an in-range minimum close
  to a boundary; never describe it as extrapolated when `invalid=false`.
- Stable-phase selection requires valid Gibbs values for **all configured phases**
  at the same P-T point. Never describe “at least one valid phase” as sufficient.
- Do not compare raw total energies from phases with different cell sizes. Use
  the program's `eV/formula unit` Gibbs values and keep each method internally
  consistent.

## Verification

- Confirm `ph --version`, the correct workflow status, selected method set, run
  directory, and configuration fingerprint behavior.
- For ordinary completed methods, confirm `forces.npy`, `FORCE_SETS`,
  `force_constants.hdf5`, `phonopy_params.yaml`, `band.yaml`, thermal outputs,
  plots, and `summary.json` exist.
- For completed QHA methods, confirm every expected volume is complete; every
  phase has `volume_points.csv`, `e-v.dat`, `qha_grid.csv/npz`, and summary; and
  `phase_diagram/phase_map.csv`, boundaries, plots, and summary exist.
- If VASP is selected, report readiness counts and require complete parseable
  `vasprun.xml` results rather than treating OUTCAR presence as completion.
- Read stability and coverage flags from result summaries. Disclose significant
  imaginary modes, excluded modes, invalid grid points, and near-edge grid points
  even when the workflow status is `completed`.
- State the result's force provider, normalization basis, temperature/pressure
  range, and unresolved supercell, q-mesh, DFT, model-domain, or anharmonic
  convergence limitations.

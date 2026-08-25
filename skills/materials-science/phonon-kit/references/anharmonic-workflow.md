# Anharmonic Phono3py Workflow

Load this reference for third-order force constants, three-phonon scattering,
linewidth, lifetime, RTA conductivity, NAC, or `ph anh` recovery.

## Command selection

```text
ph anh init CASE
ph anh plan [anh.yaml]
ph anh run [anh.yaml|run-dir] [--new]
ph anh status [anh.yaml|run-dir]
ph anh plot [anh.yaml|run-dir]
```

`plan` and `status` are read-only. `run`, `init`, configuration changes, and
`plot` require explicit user authority. Run long calculations in the background.
There is no `resume`: rerun `ph anh run`, preferably with the concrete run
directory after case inputs have changed.

## Configuration decisions

- Require an already relaxed periodic structure. `ph anh` never relaxes it.
- `supercell` controls fc3. `fc2_supercell: null` reuses the fc3 dataset; an
  explicit value generates separate fc2 displacements and may be larger because
  harmonic interactions can be longer ranged.
- Systematic displacement defaults to 0.03 Angstrom. Compare a nearby amplitude
  for formal work instead of choosing it only to reduce force noise.
- `mesh` controls scattering cost, not DeepMD displacement count. Converge it.
- `subtract_residual_forces` defaults false. When true, each model evaluates its
  perfect fc3 supercell and, if different, perfect fc2 supercell, then subtracts
  those force arrays from every corresponding displaced structure.
- `structure.born_file` enables shared BORN/NAC data. Confirm it belongs to the
  same primitive structure and atom order.
- Only DeepMD methods are accepted. All enabled models share structure,
  displacements, primitive matrix, NAC, and mesh. `.pt2` cannot set `head`.

## Cost and checkpoints

Always run `ph anh plan` first. Report fc3 and separate fc2 displacement counts,
atoms per supercell, residual evaluations, model count, total force predictions,
and mesh. Systematic fc3 can generate thousands of structures even when harmonic
Phonopy needs only a few.

Each displaced structure has an NPZ checkpoint under
`work/methods/<model>/checkpoints/{fc3,fc2}/`. RTA writes official per-grid-point
Phono3py gamma HDF5 files under `scattering/`. Re-running computes only missing
force or q-point files, then reads all gamma files to rebuild kappa.

## Physical interpretation

- `gamma` is imaginary self-energy and half linewidth in THz.
- `linewidth_thz = 2 * gamma_thz`.
- `lifetime_ps = 1 / (4*pi*gamma_thz)`; non-positive gamma is `NaN`.
- Conductivity is intrinsic three-phonon RTA only. It excludes isotope, boundary,
  electron, and four-phonon scattering.
- `kappa_trace_over_3` is `(xx+yy+zz)/3`, not an error metric.
- With significant imaginary modes, completed kappa is diagnostic. Never call the
  structure stable or the conductivity trustworthy solely because files exist.
- Without an external reference, model plots show comparison, difference, or
  change. They do not establish accuracy or improvement.

## Verification

For every completed model require `forces_fc3.npy`, any separate `forces_fc2.npy`,
`fc2.hdf5`, `fc3.hdf5`, `phono3py_params.yaml`, `kappa.hdf5`, conductivity and
lifetime CSV, `kappa.png`, `gamma.png`, `lifetime.png`, and `summary.json`.
Report minimum frequency, significant-imaginary count, NAC state, raw-versus-
residual force convention, mesh, temperature range, and unresolved convergence.

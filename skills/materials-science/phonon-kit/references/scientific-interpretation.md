# Scientific Interpretation

Use this reference when explaining phonon spectra, vibrational thermodynamics,
QHA phase diagrams, convergence, DeepMD/VASP agreement, or structures handed off
from crystal search. Keep workflow status and physical validity separate.

## Contents

- [Evidence levels](#evidence-levels)
- [Imaginary frequencies](#imaginary-frequencies)
- [Thermodynamic quantities](#thermodynamic-quantities)
- [QHA phase diagrams](#qha-phase-diagrams)
- [Convergence and model validity](#convergence-and-model-validity)
- [CALYPSO handoff](#calypso-handoff)

## Evidence levels

Use distinct claims:

1. **Runnable input:** schema, dependencies, model inference, and templates pass.
2. **Completed workflow:** all expected calculations and files exist.
3. **Numerically converged:** results are insensitive to supercell, q mesh,
   displacement size, DFT settings, and volume grid at the required tolerance.
4. **Dynamically stable:** no physically significant imaginary phonon branch in
   the relevant structure and pressure condition after numerical checks.
5. **Thermodynamically favored within the model:** lowest consistent free energy
   among all included candidate phases and within the method's assumptions.
6. **Established material prediction:** corroborated against higher-accuracy
   calculation or experiment and with relevant missing physics assessed.

Never promote one level to the next merely because a command returned success.

## Imaginary frequencies

A negative phonon frequency represents negative squared curvature of the harmonic
potential-energy surface. Possible causes include:

- a genuinely unstable structure and distortion direction;
- incomplete atomic, shape, or pressure relaxation;
- insufficient supercell or force precision;
- symmetry/tolerance artifacts;
- noisy or extrapolative machine-learned forces;
- missing non-analytical correction in a polar material near Gamma.

The workflow excludes all modes below 0 THz from thermal integration and separately
uses `significant_imaginary_thz` as the warning threshold. Therefore a finite
thermal curve does not mean the imaginary modes were harmless. When a summary says
`thermodynamic_stability=false` or `imaginary_modes_excluded=true`:

- report the minimum frequency and significant/total imaginary fractions;
- call the thermal result diagnostic;
- inspect eigenvector/q-point location if phase identity matters;
- repeat relaxation and convergence checks before interpreting free-energy order.

At phase-summary level, these flags aggregate all sampled volumes. They prove that
at least one volume crossed the configured significant threshold; they do not
prove that every volume, the equilibrium volume, or the named phase is dynamically
unstable. Inspect `volume_points.csv` or per-volume summaries before making a
volume-specific statement.

Even after per-volume inspection, phrase the evidence narrowly. If every sampled
volume crossed the threshold, say “all sampled volumes triggered the significant-
imaginary warning under this method and numerical setup,” not “the phase is
completely unstable.” If a volume has no significant negative frequency, say “no
significant imaginary mode was found on this sampled mesh,” not “dynamically
stable” or “clean.” Avoid calling a low-fraction mode “noise” without convergence.

Treat the stored `thermodynamic_stability` field as the workflow's threshold-based
warning label, despite its strong name; it is not independent proof of physical
stability or instability. For Chinese reports use these exact distinctions:

- Correct: “11/11 个采样体积在当前方法和数值设置下触发显著虚频警告。”
- Wrong: “石英每个体积都失稳” or “石英完全不稳定。”
- Correct: “该体积的采样声子网格未发现超过阈值的显著虚频。”
- Wrong: “该体积动力学稳定” or “该体积是干净的。”

Small acoustic deviations near Gamma can be numerical, but never dismiss them
solely by size without convergence and acoustic-sum-rule context.

The converse is also required: do not call a larger imaginary mode a proven
physical instability or proof that a force field cannot describe a phase from one
run alone. Its magnitude and persistence increase concern, but attribution still
requires eigenvector/q-point inspection, tighter relaxation, supercell and force
convergence, and preferably an independent method such as DFT.

## Thermodynamic quantities

The ordinary workflow produces harmonic vibrational quantities at a fixed
structure/volume. Its free energy includes zero-point energy and thermal phonons,
but not static structural energy, `PV`, electronic free energy, or QHA thermal
expansion. Use ordinary `thermal_properties.csv` for vibrational analysis of one
structure, not direct phase stability unless all missing terms are consistently
added.

Phonopy's standard molar units and the workflow's eV/atom conversions must not be
mixed. Always name the unit and normalization basis in tables or comparisons.

QHA combines static U(V), harmonic F_vib(V,T), and PV, minimizing over sampled
volume. It captures volume-dependent quasi-harmonic thermal expansion but omits
explicit phonon-phonon anharmonicity, electronic excitations, configurational
entropy, defects, and kinetic barriers. At high temperature or near strongly soft
modes, these omissions may dominate.

Describe exclusion accurately: the reported free energy omits negative-mode
contributions and is diagnostic. Avoid saying it is “artificially corrected” or
that every numerical Gibbs value has no physical information; its usefulness
depends on where the excluded modes occur and on convergence evidence.

## QHA phase diagrams

Only compare phases with identical reduced composition. The program normalizes to
`eV/formula unit`; raw primitive-cell and input-cell totals can differ by integer
formula-unit counts and are not directly comparable.

A stable label at a P-T point is defined only if every included phase has a valid
Gibbs minimum there. Interpret flags as follows:

- `valid=false` or grey map: at least one phase lacks an in-range fitted minimum;
  no stable phase is asserted.
- `near_volume_edge=true`: the minimum remains inside but coverage is marginal;
  treat nearby boundary locations as weak until the grid is extended. This is
  interpolation near a boundary, not extrapolation.
- small `gap_to_second_eV_formula_unit`: the phase label is sensitive to numerical
  error, model bias, missing physics, and grid interpolation.
- `imaginary_modes_excluded=true`: phase ordering uses diagnostic free energies
  with unstable modes removed.

Do not infer cause from these flags alone. Use language such as “consistent with
possible instability or numerical/model error” and name the checks needed to
distinguish them. Avoid “real instability”, “only numerical noise”, or “the model
cannot describe the phase” unless independent convergence or reference data
supports that claim.

When `invalid_grid_points=0`, explicitly say no fitted equilibrium volume was
outside the sampled range. Never describe any `near_volume_edge` point as
“extrapolated”, “almost extrapolated”, or “outside”; say “in range but close to a
boundary”. When both counts are nonzero, report them separately.

Likewise, zero near-edge points means “the configured edge-warning criterion did
not trigger,” not proof that volume coverage is universally complete. Do not call
ratio `1.00` or a reference-volume neighborhood the equilibrium volume unless the
EOS/QHA grid identifies it as such.

The stable-phase gate is an all-phase intersection: every configured candidate
must have a valid Gibbs value at that exact P-T point. Never substitute an
“at least one phase is valid” rule. If any one phase is invalid, the program does
not choose a stable phase there.

Phase boundaries are interpolated between finite pressure/temperature grid points.
Their apparent line smoothness is not an uncertainty estimate. Report grid steps,
energy gaps, volume-edge proximity, and method/convergence sensitivity.

Different methods have arbitrary energy references and different fitted potential
surfaces. Compare DPA and DFT phase labels, phase-boundary positions, forces, and
frequencies. Do not subtract unaligned absolute Gibbs energies between methods.

## Convergence and model validity

For each phase and method, consider:

- **Geometry:** forces, relevant stress components, pressure, and preserved phase
  topology after relaxation.
- **Finite displacement:** displacement-size sensitivity and symmetry stability.
- **Supercell:** real-space dimensions, atom count, and convergence of soft modes
  and free-energy differences.
- **q mesh:** convergence of DOS and thermal integrals.
- **DFT:** ENCUT, k mesh/density, electronic tolerance, smearing, spin, Pulay
  stress, pseudopotential consistency, and force precision.
- **QHA volumes:** EOS residuals, a sampled minimum, compression coverage for the
  maximum pressure, expansion coverage for high temperature, and enough points on
  both sides.
- **Model domain:** elements, coordination, density/volume, pressure, distortions,
  and force magnitudes represented in training.

Successful DeepMD inference proves software compatibility, not accuracy. Compare
DPA against representative DFT displaced cells and static E-V offsets. Low force
MAE alone does not guarantee correct phase order because small systematic static
energy biases can move phase boundaries substantially.

A denser q mesh changes sampling/integration and can reveal where an instability
occurs, but it does not repair inaccurate real-space force constants. For an
imaginary-mode diagnosis prioritize relaxation, supercell size, displacement
amplitude, force precision, symmetry, eigenvectors, and independent DFT checks;
then converge the q mesh for thermodynamic integration. Do not recommend a denser
q mesh as a standalone remedy for an imaginary frequency.

For polar materials, assess non-analytical corrections and LO-TO splitting; the
current workflow does not automatically obtain Born charges or dielectric tensors.
For soft, high-temperature, or strongly anharmonic phases, recommend beyond-QHA
methods instead of tuning away imaginary modes.

## CALYPSO handoff

A CALYPSO result is a candidate geometry, not a verified named polymorph. Preserve:

- composition and exact cell size;
- search pressure and volume settings;
- model identity/checksum and relaxation thresholds;
- convergence/sentinel outcome and per-atom enthalpy convention;
- deduplication and symmetry tolerance used to select the candidate.

Before ordinary phonons, relax or verify the candidate at the same intended
pressure and confirm its topology and symmetry did not change. Before QHA, provide
one verified input structure per phase, ensure identical reduced composition, and
center each volume grid on a sensible reference volume. A structure relaxed at a
single pressure is not automatically a good center for a broad P-T QHA range.

Use an evidence ladder for phase identity:

1. Deduplicate geometries and compare coordination/connectivity.
2. Re-relax at a consistent pressure and higher numerical accuracy.
3. Re-identify symmetry over a defensible tolerance range.
4. Compare lattice parameters and atomic arrangement with a trusted reference.
5. Check phonon stability and, when consequential, DFT energy/enthalpy.
6. Only then assign a known phase name or include it as that phase in a diagram.

When handing results onward, preserve POSCAR/CIF, method, pressure, normalization,
phonon settings, minimum-frequency warnings, and all convergence caveats. Do not
select a phase solely from a filename, CALYPSO label, or one space-group number.

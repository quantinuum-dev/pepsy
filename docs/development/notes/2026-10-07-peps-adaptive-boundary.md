# Adaptive boundary caps before PEPS sweeps

The user requested automatic norm/overlap convergence checks before each
`PepsOptimizer` sweep fit, bounded by a configurable maximum, with warnings
and continuation when convergence is not established. This extends the
[earlier reliability corrections](2026-10-07-peps-sweep-reliability.md).

## Implemented policy

`boundary_convergence=True` is the default. Both state norms and the complex
overlap are recomputed on unchanged warm-start/target states, in x and y,
using the actual fitting boundary engine and compression policy. Norm changes
must meet relative tolerance; normalized complex overlap changes must meet
absolute-plus-relative tolerance. This avoids accepting a constant fidelity
ratio whose numerator and denominator have not stabilized. Scalars retain
their base-10 exponents; tensor environments are not cached between probes.

The requested fit/normalization/evaluation caps form the initial minimum.
Caps double up to a default maximum of 512; default tolerances are 1e-5
relative and 1e-8 absolute for normalized overlap. The higher cap in an
agreeing pair is selected. Later fits test a lower probe against the retained
floor before increasing it, avoiding unconditional doubling each time step.
Selected caps govern fitting, normalization, and evaluation. Target and
warm-start norms are reused; explicit custom normalization still runs.

At the maximum, nonconvergence warns and is recorded, then fitting continues.
Unusable nonpositive/nonfinite norms still fail. Existing local-update and
independent postcheck safeguards remain. Pre-fit agreement is an empirical
test and does not guarantee later local environments or a full trajectory.
Global mode and standalone `SweepOptimizer` are unchanged. Custom boundary
stores/options currently require opting out rather than comparing mismatched
policies. Records expose all probe scalars, changes, selected caps, status,
and a separate timing phase. Nonfinite diagnostic scalars serialize as strings.

The example runner forwards enable/disable, maximum, relative and absolute
tolerances. Execution-policy identity is now 5, separating old saved runs.

## Validation

- New adaptive tests: **21 passed**, including real Torch complex64/128 fits,
  exact-reference direct/DMRG/Quimb-MPS probes, cap reuse, false-convergence
  cancellation, bounded nonconvergence, target ownership, custom normalization,
  and nonfinite diagnostics.
- Fixed-cap legacy suites: **284 passed**. These now explicitly opt out of
  adaptive checks where they test fixed-cap numerical or contraction budgets.
- Example runner adaptive controls and optimizer suites: **39 passed**,
  including default forwarding and effective-cap assertions for real fits.
- Earlier adaptive/public API/package-layout selection: **73 passed, 1 failed**.
  The only failure is `test_package_version_matches_installed_distribution`:
  runtime and installed metadata say 0.4.0; source project says 0.5.0. No
  dependency reinstall or source version change was made for this mismatch.
- Ruff is unavailable in the active cloudspace environment. Diff whitespace
  checks pass. No full-suite claim is made.

A bounded A100 test used a seeded 3x3 D=2 Torch complex128 state, one RZZ
target, SciPy five iterations per slice, and six local fits. Both direct and
DMRG (`eff`) compression selected (8,8) after probes (2,2), (4,4), (8,8).
Exact final fidelity was 0.9860510016484156 (direct) and
0.9860510016484157 (eff); exact output norms differed from one by less than
3e-15. Total times were 2.93s and 4.88s; calibration used 0.289s and 2.369s.
These are small-case validations, not a 5x6 accuracy or performance result.
Temporary evidence: `/tmp/pepsy_adaptive_cuda_probe.{py,log,json}`.

## Dependency audit

Installed versions remain Quimb 1.15.1.dev75+g4112e304a,
Autoray 0.11.1.dev9+g1291702f9, Cotengra 0.8.3.dev7+g1d7fd333f,
Symmray 0.4.1.dev11+g1a3481803, Torch 2.11.0. Inspected boundary metric
signatures and scaled-scalar helpers. Reviewed official
[Quimb notes](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[notes](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray).
The abelian-array documentation remained inaccessible. Classification:
adopt existing supported contraction/scaled-scalar APIs; defer unrelated
upstream upgrades. No installed libraries changed. Native Symmray adaptive
fits were not separately validated in this task; dense Quimb-MPS was tested.

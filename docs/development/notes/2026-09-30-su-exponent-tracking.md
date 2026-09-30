# 2026-09-30 — Preserve SU gauge scales in the network exponent

The user requested bounded simple-update gauges with their physical scale
retained in tensor-network exponent metadata. The immediate motivation was
the reproducible 3x3 roughening overflow documented in
[the exact-norm diagnosis](../../../history/2026-09-30-3x3-absorbed-norm-overflow.md).

## Implementation

`pepsy.operators.gate_simple(..., renorm=False, strip_exponent=True)`:

- Reuses the existing native/backend-aware `renorm_gauge` scale extraction.
  Dividing a gauge by positive RMS s adds log10(s) to the parent TN exponent.
  The norm's squared scaling is therefore recovered as 10**(2*exponent).
- Normalizes existing internal gauges at the start of a nonempty stream and
  each updated gauge after every adjacent gate and routed SWAP. A zero RMS
  uses scale one; the extraction floor is zero so tiny nonzero scales are
  retained. Scale reductions are detached, but gauge data and gradients remain.
- Requires `renorm=False`, avoiding physical scale loss. Nonzero absolute
  cutoffs are rejected because their threshold depends on scalar gauge;
  relative modes and zero cutoff are supported. The new option defaults off.
- Adds no decomposition, BP solve, or full gauge-equilibration sweep.
  It tracks gauge scale only; it is not a guarantee against arbitrary input
  overflow or ill-conditioned singular-value ratios.

The existing `renorm_gauge` API remains unchanged; its index-based body is
shared with initialization of the new mode. Core tensor/gauge ownership and
the existing out-of-place contract are retained. In particular, callers who
need the original state must also retain a copy of its gauge dictionary.

The roughening PEPS engine now enables this option alongside `renorm=False`
and records it in the normalization metadata. Measurement copies retain the
exponent. Public gate docs, the package changelog, and the runner guide were
updated. `SimpleUpdateGen`'s separate gate-option allowlist was not extended;
this task's integration is the direct gate API used by roughening.

## Upstream audit

**Adopt:** Quimb's existing network `exponent`, full contractions with
`strip_exponent=True`, and D2BP/loop-cluster exponent propagation. Reuse Pepsy's
existing scalar gauge helper rather than copying upstream gating internals.
The installed `TensorNetwork.strip_exponent`, `equalize_norms`,
`distribute_exponent`, D2BP contraction methods, and SU gate implementation
were inspected. SU does not itself retain a discarded `renorm=True` factor,
so the wrapper uses raw `renorm=False` followed by explicit scale bookkeeping.

Reviewed official sources:

- [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html):
  exponent-aware contractions and BP/norm scale handling.
- [Autoray source](https://github.com/jcmgray/autoray): backend dispatch.
- [Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
  [changelog](https://cotengra.readthedocs.io/en/latest/changelog.html).
- [Symmray source](https://github.com/jcmgray/symmray). Its requested
  `abelian_arrays.html` documentation page was unavailable; used the installed
  block operations and existing native scale tests instead. The attempted
  Quimb `examples/ex_simple_update.html` page was also unavailable.

Installed versions: Quimb 1.15.1.dev66+ge927f06e1, Autoray
0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f, Symmray
0.4.1.dev7+g83fb22865, Torch 2.6.0+cu124. The environment was not modified.

## Validation

- Existing `test_gate.py`, `test_gauge_scale.py`, `test_simple_update_gen.py`:
  175 passed, 1 skipped (CuPy GPU check, CUDA hidden for this CPU validation).
- New `test_gate_simple_scale.py`, public API, package layout, and existing
  3x3 BP reference tests: 73 passed, 1 failed. The sole failure is the
  preexisting `test_package_version_matches_installed_distribution`: installed
  metadata 0.4.0 differs from checkout pyproject 0.5.0. Local imports resolve
  to this checkout. This mismatch was already observed in the prior audit.
- All 14 new gate-scale tests passed: dense adjacent/routed amplitudes,
  truncated-state equivalence, source isolation, unit-RMS gauges, finite
  logarithmic norms outside float range, Torch gradients, policy rejection,
  native U1/U1U1/Z2 fermionic PEPO norm equivalence and type preservation.
- A first diagnostic used MPS input for a D2BP cluster test and hit an existing
  Quimb structured-MPS cluster issue (`contract()` returned TensorNetwork1D,
  not a scalar). The large-scale norm test uses a PEPS, matching this task;
  MPS cluster compatibility was not changed or claimed fixed.
- Roughening `tests/test_peps.py`: 35 passed. The previously failing default
  gauge-interval cases now pass. The 3x3 matrix covers NumPy/Torch CPU, D=2/4,
  gauge intervals 0/1, first-step dense amplitudes, t=4/6 exact/C=9 norms,
  C=0/4 wrapper agreement, local Z, imbalance, and measurement isolation.
- Repository-wide Pepsy Ruff, changed examples files' Ruff, and both diff
  whitespace checks passed. Full repository numerical suites were not run.

## Run evidence

For 3x3, dt=0.25, complex128 NumPy, no periodic gauge equilibration, exact
full Cotengra contractions agree with every pre-failure norm from the old
implementation: max relative differences 1.49e-14 (D=2) and 1.20e-14 (D=4).

| D | t | Exact squared norm | Exact norm | TN exponent |
| --- | --- | ---: | ---: | ---: |
| 2 | 4 | 0.10767396391115552 | 0.32813711145061836 | -0.04687327551877911 |
| 2 | 6 | 0.07173840781570597 | 0.26784026548617734 | -0.13152843379517398 |
| 4 | 4 | 0.17046475131736238 | 0.41287377165104877 | 0.49981127407146164 |
| 4 | 6 | 0.08800392273753682 | 0.29665455118291517 | 0.35532716792562735 |

The largest core entries over all 24 steps are about 60.7 (D=2) and 1.75e5
(D=4), instead of the prior 1e158/1e178 blow-up. Gauge RMS stays one. These
are norms of the truncated SU state, not exact untruncated time evolution.

An actual `run_roughening.py` CLI run completed for 3x3 D=4, boundary chi=32,
dt=0.25, depth=24, J=-1, hx=1, hz=0, diagonal x+y wall, snake mapping,
delta_theta=0, NumPy complex128. It saved local observables, C=0/4 norms,
and 16 samples at each of t=4 and t=6 with default gauge interval zero.
Both BP solves converged. Every stored sample amplitude was checked against
the corresponding exact state amplitude; max error 1.51e-16. Log weights
were finite and normalized weights summed to one. Manifest is complete.

Artifacts: `/tmp/peps_su_exponent_cli_20260930_ehMBjl/`, including `cli.log`,
`run/` outputs and metadata, and `norm_validation.json` for all norm/scale
steps and comparison errors. No GPU/5x6 validation or production launch was
performed. Existing GPU jobs were untouched.

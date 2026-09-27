# 2026-09-26 — PEPS sampler automatic cutoff and precision checks

Status: implemented in the working tree on `develop` / `80f451a`.
Scope: user's requested small-case validation with `cutoff="auto"` and
`cutoff_mode="auto"`, following MpsOptimizer for complex64 and complex128.
No commits, publishing, shared-environment changes, or production-job changes.

## Implemented policy

- **Adopt:** reuse `pepsy._internal.cutoff.dtype_auto_cutoff`, also used by
  MpsOptimizer. `PepsSampler` now defaults to automatic cutoff: complex64 and
  float32 resolve to 1e-6; complex128 and float64 resolve to 1e-12. Resolution
  occurs after private backend conversion and repeats on `refresh()`.
- **Adopt:** automatic cutoff mode (also None) resolves to `rsum2`, using the
  existing Quimb boundary-store mode validator. Explicit numeric cutoff and
  Quimb mode strings remain available. Invalid strings/non-finite/negative
  values and Boolean numeric cutoffs fail explicitly.
- Quimb future environments and conditioned ket compression receive both
  resolved settings. The FIT ket guess receives the same settings, and
  `FIT.run_eff` receives them explicitly. DMRG future preparation receives
  the requested cutoff policy and resolved mode through `CompBdy`.
- **One-site FIT limitation:** its current fixed-rank updates have no
  singular-value truncation; chi determines the represented future rank.
  The conditioned FIT ket rank is set by its compressed guess. Passing
  cutoff options does not silently change FIT block size or iteration policy.
- Updated the [API](../../api/sampling/samplers.md), changelog, and
  [example](../../../examples/peps_sampling.py). Source PEPS tensors, backend,
  dtype, and device remain preserved in the numerical regressions.

## Numerical cases

Ran 32 deterministic cases: NumPy and Torch CUDA; complex64 and complex128;
Quimb future + Quimb ket compression and DMRG future + FIT ket compression.
Shapes were 1×3, 2×3, and 3×2 at D=2 with chi=16, chi_prime=8, plus active
ket truncation on 2×3 with chi=4, chi_prime=1. Each case enumerated the entire
proposal distribution and checked eight grouped samples against queried
likelihoods and original-state amplitudes. The exact reference contracted a
complex128 copy of each realized input state, including complex64 inputs.

All 32 cases passed. Maximum errors across NumPy/CUDA were:

| Check | complex64 | complex128 |
| --- | --- | --- |
| Probability versus exact Born law, ample bond caps | 8.07e-8 | 2.23e-16 |
| Proposal normalization, including chi_prime=1 | 2.17e-7 | 3.34e-16 |
| Sampled log probability versus likelihood query | 1.10e-6 | 1.78e-15 |
| Relative L2 sampled-amplitude error | 1.63e-7 | 2.25e-16 |

With active chi_prime=1 truncation, proposal probabilities differ from the
Born law, as expected: maximum absolute differences were 0.02524 (complex64
cases) and 0.01156 (complex128 cases). The two dtype groups need not generate
identical random tensors; these numbers are not a precision comparison of a
single fixed state. Proposal normalization and sampled/query consistency
remain accurate. No speed benchmark was performed.

## Controlled truncation and corner cases

- A 2×2 state leaves a conditioned ket `|00> + 1e-4 |11>` after the first
  measured row. `rsum2` auto discards the weak branch for complex64 and keeps
  it for complex128, for both Quimb and FIT ket compression. Explicit `rel`
  or cutoff=0 keeps it for both precisions.
- A 2×3 weighted GHZ state with amplitude ratio 0.01 isolates the cached
  future, with ket compression disabled. Auto rsum2 loses its weak branch
  for complex64, while explicit rel keeps it. Complex128 preserves the branch
  for both modes. This verifies future-mode forwarding independently.
- Thus a cutoff is a local singular-value criterion, **not** a bound on global
  Born-probability error. Future double-layer singular weights are different
  from single-ket Schmidt weights. As documented previously, importance
  weights cannot recover proposal support lost to truncation.
- Conversion complex128→complex64 resolves auto to 1e-6; refresh after source
  precision changes resolves again. Explicit numeric cutoffs stay fixed.
- Native NumPy, Torch CPU, and JAX CPU precision regressions enumerate 2×2
  distributions for exact, Quimb, and DMRG/FIT modes; they verify amplitudes,
  proposal logs, tensor signatures, and source preservation. JAX complex128
  tests use a scoped public x64 context, with an older-API fallback in tests.

## Validation

- New precision/policy selection before the additional future test:
  **29 passed**, 101 deselected, 10 existing Quimb compatibility warnings.
- Full focused PEPS suite: **132 passed** in 201.95 s, including both new
  future-boundary cases; 25 existing Quimb mode/method warnings.
- Existing sampler/public API/layout: **162 passed, 1 failed**.
- Default smoke: **153 passed, 1 failed**.
- Both failures are the existing
  `test_package_version_matches_installed_distribution`: installed/runtime
  0.4.0 versus pyproject 0.5.0. No metadata was modified.
- Ruff src/tests/example and whitespace checks passed; final links and
  focused-suite results are recorded in the handoff. No full-suite pass is
  claimed.

Temporary evidence: `/tmp/peps_sampler_auto_cases.py/.json/.log`,
`peps_sampler_auto_targeted.log`, `peps_sampler_auto_focused.log`,
`peps_sampler_auto_broader.log`, and `peps_sampler_auto_smoke.log`.
Before-edit source snapshot: `/tmp/peps_sampler_before_auto_cutoff.py`.

## Installed upstream capabilities

Reused the same-task official-source audit recorded in the
[final sweep audit](peps_sampler_final_sweep_audit.md), and rechecked installed
versions/signatures. Versions are unchanged: NumPy 2.5.2, Quimb
1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305, Cotengra
0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1, Symmray 0.4.1.dev7+g83fb22865,
Torch 2.6.0+cu124, JAX 0.10.2. Inspected public MPS compress/tensor_split/
compute_y_environments signatures and installed CompBdy/FIT.run_eff dispatch.
The first new JAX test used an obsolete experimental context name; corrected
it to prefer installed public `jax.enable_x64` before rerunning successfully.

**Compatibility shim:** none in production. Existing Quimb mode-to-method
FutureWarnings remain. **Defer:** FIT algorithm/block-size changes, native
Symmray PEPS sampling, JAX GPU, large-D accuracy/performance claims, and any
proposal mixture to guarantee support. These are outside this cutoff task.

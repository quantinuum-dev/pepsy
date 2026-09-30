# 2026-09-26 — PEPS sampler relative-cutoff scaling fix

- Scope: implement the authorized follow-up to the
  [maturity audit](peps_sampler_maturity_comparison.md), preserving the cached
  future sweep and whole-row projection/absorption order.
- Branch / baseline: `develop` / `a13031b`.
- Status: implementation and regressions in the working tree; not committed
  or published. No dependency, installed library, source PEPS, or production
  job was modified.

## Implemented

`pepsy.sampling.peps.PepsSampler` rescales private proposal tensors before
relative-cutoff compression. Each array is divided by its largest absolute
entry before computing its Frobenius norm, then divided by that norm. Zero
arrays are preserved. These operations stay on the inferred array backend,
dtype, and device, and preserve autodiff connections.

Only `rel`, `rsum1`, and `rsum2` enable this scaling. Their decisions depend on
singular-value ratios, so removing a positive overall network scalar does not
change the intended cutoff. `abs`, `sum1`, and `sum2` retain the previous
compression scale. The code does not modify Quimb's decomposition registry.

The policy is applied to:

1. A private copy of the norm network used to prepare Quimb future boundaries.
   The existing `QuimbMpsBoundaryStore(equalize_norms=True)` option also keeps
   intermediate boundary contractions and canonicalization scales bounded.
2. A conditioned ket boundary immediately before Quimb compression or creation
   of the compressed FIT initial guess. The full FIT target and its guess use
   the same rescaled target.

The original private `_ket` and `_norm` remain available at their original
scale. Returned amplitudes still come from `_ket`. This change does not
rescale a source PEPS, its evolution gauges, or a reported physical norm.
DMRG future preparation retains its existing normalization policy.

Future environments are prepared once per constructor/`refresh()`, from high
`y` toward low `y`. Draws and likelihood queries traverse increasing `y`, then
increasing `x`, with a separately conditioned ket boundary for each sampled
prefix. Each sampled row is fully fixed before its projected ket is
absorbed/compressed for the next row; the final row needs no boundary update.
Repeated draws do not rebuild the future cache.

## Evidence

The new weighted-GHZ 2×3 regression has Born probabilities `(0.8, 0.2)` while
scaling each tensor in its top row by `1`, `1e5`, or `1e-5`. Before the fix the
NumPy complex64 large-scale case fails. A separate conditioned-MPS regression
uses known singular values proportional to `(1, 0.5)`, at scales `1`, `1e20`,
and `1e-20`; before the fix the large complex64 case produces invalid
normalization. Both tests cover complex64 and complex128 on NumPy, Torch CPU,
and JAX CPU. Existing regressions still distinguish legitimate small-weight
truncation under automatic complex64 `rsum2` from explicit `rel` retention.

The audit's original 5×6 D=2 case was rerun after implementation: generate a
complex128 PEPS with seed 101, cast the same tensor data to complex64, set
χ=32, χ′=16, Quimb future boundaries, automatic cutoffs, greedy contraction,
32 shots, seed 800. Each process used one CPU thread; no GPU run was added.

| Backend | Fixed ESS / N | Future ranks at rows 0, 1 | Maximum rho Hermiticity defect |
| --- | ---: | --- | ---: |
| NumPy complex64 | 0.9999996621 | 14, 15 | 9.67e-7 |
| Torch complex64 | 0.9999992715 | 14, 15 | 8.01e-7 |

The earlier Torch result was ESS/N=0.854099102, ranks `(1, 1)`, and a
Hermiticity defect of approximately 0.4293. These are bounded correctness
probes, not a new performance benchmark or a guarantee of exact Born sampling
at finite χ/χ′.

Regressions also cover:

- physical amplitude scale and unchanged source arrays;
- scale-sensitive `abs`/`sum1`/`sum2` rank choices for Quimb and FIT;
- one future preparation across serial draws, repeated batches, and queries,
  followed by an explicit rebuild after a source physical filter;
- a Torch normalized-state probability gradient against its analytic value.

## Upstream decision and compatibility

**Adopt:** existing public Quimb boundary equalization and Autoray namespace
operations, gated by the relative-cutoff policy. No global compatibility shim,
upstream patch, dtype promotion, or dependency upgrade is needed.

The immediately preceding audit's installed-version and official-source review
was reused in the unchanged environment: NumPy 2.5.2;
Quimb 1.15.1.dev66+ge927f06e1; Autoray 0.11.1.dev3+g1b476b305;
Cotengra 0.8.3.dev7+g1d7fd333f; Cotengrust 0.2.1;
Symmray 0.4.1.dev7+g83fb22865; Torch 2.6.0+cu124; JAX 0.10.2.
See the audit for official-source links and the unavailable Symmray arrays
page. Dense NumPy, Torch, and JAX are affected; native Symmray PEPS remain
explicitly unsupported by this sampler.

Installed `TensorNetwork2D.compute_environments` accepts `equalize_norms` and
forwards it through public boundary contraction. Its MPS implementation scales
contracted tensors and canonicalization/compression intermediates, tracking
scalars in network exponents. `MatrixProductState.compress` accepts the
existing cutoff options through `**compress_opts`. The sampler does its ket
rescaling before that call and normalizes the resulting ket as before.

## Validation and limits

The complete PEPS sampler suite passes: **159 tests**, including NumPy,
Torch CPU, and JAX CPU in complex64 and complex128. Ruff and diff checks pass.
The public API/layout selection has 58 passes and one existing installed-version
metadata failure; smoke has 162 passes, one skip, and the same metadata failure.
The shared environment reports 0.4.0 while checkout metadata declares 0.5.0.
The entire repository suite was not rerun. Details are in the
[session handoff](https://github.com/quantinuum-dev/pepsy/blob/develop/history/2026-09-26-peps-sampler-boundary-scaling.md).

This fix does not promise arbitrary-scale raw amplitude/rho contractions.
Absolute cutoff modes intentionally retain their scale dependence. Finite
χ/χ′ can still lose proposal support; increasing caps or checking convergence
is required for that separate approximation issue. Native GPU batch-axis
sampling and the earlier performance proposals remain unimplemented.

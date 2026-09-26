# 2026-09-26 — Final PEPS sampler sweep and corner-case audit

Status: implemented and validated in the working tree on `develop` / `80f451a`.
The user requested a final review of the simple sequential algorithm and its
corner cases. This supersedes the default-cache policy in the earlier
[performance implementation](peps_sampler_performance_implementation.md).
No changes were committed, staged, published, or made to production jobs.

## Final algorithm and dimensions

- **χ = marginal_chi:** cached future double-layer boundary compression.
- **χ′ = sample_chi:** compression of the conditioned single ket boundary.
- Prepare only the future side, opposite to the sampling direction. The
  Quimb provider uses `start_sweep(..., update_side="left")`; DMRG constructs
  only `Y*_r` boundaries with the existing public lazy BdyMPS option.
- At each site, combine the conditioned prefix, current row, and cached future
  and contract a `d × d` rho. Sample its normalized real diagonal and fix both
  physical legs. A joint row draw is factored into these site conditionals;
  no `d**Lx × d**Lx` density matrix is allocated.
- Only after a complete row is fixed, absorb that single ket layer and
  truncate to χ′. The last row needs no further boundary update. Future
  environments remain unchanged for all shots; prefixes carry separate ket
  boundaries once their configurations differ.
- Accumulate log probabilities; retain scaled mantissa/exponent results and
  independently contract the original private ket for amplitudes.

The implementation orders sites with increasing x within fixed y, then
increasing y. Calling the slices columns corresponds to exchanging axis names.
In Quimb's environment representation the outermost future row can remain
exact/factored before compression, so χ does not bound its original PEPS bonds.
With `ket_compression=None`, χ′ does not cap the represented ket-boundary bonds.
With no future environment, identity caps define a different approximate
proposal. See the [API](../../api/sampling/samplers.md) for current behavior.

## Fixes made

1. **Simple default:** `row_cache_max_bytes=0`; positive values explicitly opt
   into the previously implemented guarded dense transfer cache. Disabled
   caching does not spend work estimating unused allocations, and reports
   `estimated_cache_bytes=None`.
2. **One future direction:** eliminated unused Quimb opposite-side preparation
   and eager construction of unrelated DMRG boundaries. Missing required future
   entries now raise instead of silently changing to identity environments.
3. **Stable explicit likelihoods:** added natural-log `log_probability(config)`.
   Exact and default boundary queries use public exponent-stripped contractions;
   the common rho scale cancels in normalization. Exact likelihoods accumulate
   log probabilities rather than multiplying them in the input float dtype.
   `probability(config)` exponentiates the final log, possibly underflowing only
   at that final Python-float conversion. Query diagnostics describe scaled
   rhos; ordinary sampling diagnostics retain the contraction scale.
4. **Input and geometry guards:** validate the whole configuration before any
   zero-branch early exit; reject fractional/out-of-range physical values,
   non-finite cutoffs, Boolean cutoff/count parameters, and integer tensor data
   without an explicit converter. Boundary sweeps reject periodic edges;
   exact contraction remains available for periodic PEPS.

## Corner cases checked

| Case | Finding / behavior |
| --- | --- |
| 1x1, 1x3, 3x1, 2x3, 3x2 | Quimb and DMRG proposals agree with exact Born probabilities when cutoffs avoid truncation |
| Unequal physical dimensions, including d=1 | Range checks and grouped draws work; row-major order is respected |
| Zero-probability prefix followed by an invalid value | Full configuration validation now rejects the invalid value |
| Rare probability approximately 10^-480 | Finite log likelihood; ordinary probability converts to zero as expected |
| Float32 likelihood below its representable range | NumPy/Torch/JAX query agrees with analytic log and Python-float probability |
| Missing future cache | Explicit error; no silent identity substitution |
| Periodic boundary sweep | Explicitly rejected; exact mode still works |
| Nonstandard lattice tag scheme | Existing norm builder rejects unsupported tags clearly; no tag rewriting added |
| Integer product tensors | Require explicit floating/complex conversion; source tensors preserved |
| χ′ compression timing | Exactly once after each complete nonfinal row; represented ket bonds obey the cap |
| Shared future environments | Tags, indices, and tensor data unchanged after sampling and likelihood queries |

### Proposal support can be lost by truncation

A concrete 2x2 state fixes the first sampled row to 00 and leaves a Bell pair
in the second row: only configurations 0000 and 0011 have amplitude 1/sqrt(2).
With an exact future (χ=4), χ′=1 gives q=(1,0), while χ′=2 gives q=(1/2,1/2).
This is a real consequence of ket-boundary truncation, not a normalization or
log-arithmetic bug. Importance weights cannot recover an unsampled nonzero
amplitude. The API now states the support requirement explicitly. No hidden
probability floor, uniform proposal mixture, or clipping of material negative
probabilities was introduced. These would change the requested algorithm.

Explicit log queries do not make every raw amplitude or ill-scaled tensor
contraction overflow-proof. Dense transfer caches materialize ordinary arrays;
their opt-in likelihood path does not have full exponent-stripping protection.
Positive trace/negative diagonal validation remains strict. Native Symmray,
JAX GPU/sharding, and large production high-D validation remain outside the
implemented/tested scope; prior eager-JAX startup limitations still apply.

## Validation performed in this task

Activated the same py312 project environment, `PYTHONPATH=src`, one BLAS/OpenMP
thread. JAX was restricted to CPU; its focused tests used two CPU devices.

- `python -m pytest -xq -o addopts='' tests/test_peps_sampler.py`:
  **77 passed** in 117.38 s. Includes earlier backend, cache-budget, stable-rho,
  gradient/source, and grouping checks plus the new corner regressions.
- Additional sampler/public API/layout selection: **161 passed, 1 failed**.
- Repository default smoke: **152 passed, 1 failed**.
- Both broader failures are the unchanged installed metadata/runtime 0.4.0
  versus pyproject 0.5.0 mismatch in
  `test_package_version_matches_installed_distribution`. No full-suite pass is
  claimed and no environment metadata was modified.
- Native Torch CPU complex128 and CUDA complex128/complex64, across exact,
  Quimb future, DMRG/FIT future, and identity future: **12/12 passed**.
  Maximum amplitude errors on the unnormalized probe were 1.47e-14 and 1.73e-5;
  maximum relative error between sampled and explicitly queried proposals was
  2.10e-15 and 8.24e-7, respectively. Source tensors/device/dtype were preserved.
- Before/after differential probe with active truncation: 3x3 D=2, χ=2, χ′=1,
  eight shots, one FIT iteration. Quimb and DMRG produced the same seeded
  configurations; maximum relative likelihood differences were 1.87e-15 and
  3.04e-15. One-shot cold setup timings were not a controlled speed benchmark
  and are not reported as speedups.
- Ruff over src/tests, local documentation links, and `git diff --check` pass.

Temporary evidence: `/tmp/peps_sampler_corner_probe.py/.log`,
`peps_sampler_before_final_audit.py`, `peps_sampler_final_differential.py/.log`,
`peps_sampler_support_probe.py`, `peps_sampler_final_focused.log`,
`peps_sampler_final_broader.log`, `peps_sampler_final_smoke.log`, and
`peps_sampler_final_cuda.py/.json/.log`. The code and tests are the maintained
artifacts; temporary probe files may disappear.

## Upstream audit and decisions

Installed versions remained NumPy 2.5.2, Autoray 0.11.1.dev3+g1b476b305,
Quimb 1.15.1.dev66+ge927f06e1, Torch 2.6.0+cu124, JAX 0.10.2,
Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1,
Symmray 0.4.1.dev7+g83fb22865. Inspected installed public contraction,
compression, selection, cyclic-edge detection, lazy-boundary, and directional
provider signatures/source. No dependencies or installed sources were changed.

Rechecked official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The requested
[Symmray array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
returned an internal error; used the repository and installed implementation.

- **Adopt:** existing public one-direction provider/lazy-boundary APIs and
  `tensor_contract(strip_exponent=True)` for explicit likelihood queries.
- **Compatibility shim:** no new shim; retain previous scoped JAX placement.
  The installed Quimb provider emits a `mode`→`method` FutureWarning through
  its existing compatibility argument. Eleven such warnings appeared in the
  focused suite; numerical checks pass and no installed code was patched.
- **Defer:** native boundary batch axes, a proposal mixture guaranteeing full
  support, custom tag normalization, broader topology support in boundary mode,
  and globally scaled amplitude/contraction representations.

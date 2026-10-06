# 2026-10-06 — JAX defaults for global PEPS cleanup

Implemented the configuration requested after the
[JAX/JIT review](2026-10-06-peps-global-jax-jit-review.md).
Branch `develop`, baseline `8f7c896`; changes remain uncommitted.

## Behavior

`PepsOptimizer(mode="global")` resolves global optimizer options before
constructing its loss defaults. Selecting `autodiff_backend="jax"` supplies
`jit_fn=True`, loss `cutoff=0.0`, and loss `strip_exponent=False`. This also
handles shared `optimizer_options` and per-run global optimizer controls.
Explicit `jit_fn`, `loss_kwargs`, and legacy `loss_opt` values retain their
existing precedence. Other backends, outer measurements/normalization,
boundary caps, solver selection, and iteration budgets are unchanged.

This changes the PEPS driver's defaults, not standalone GlobalOptimizer or
the exponent conversion implementation. Explicit exponent stripping with
JAX remains unsafe for gradients; positive SVD cutoff with JIT remains
unsupported. Disabling stripping gives up overflow/underflow protection.
These limitations are documented in the public PEPS API guide.

Classification: **adopt** the explicit supported configuration established
by the preceding real contraction and gradient probes. No dependency shim,
upstream code modification, or backend registration was introduced.

## Environment and compatibility

Reused the immediately preceding unchanged-environment audit of installed
Quimb JaxHandler, TNOptimizer loss/gradient methods, and SVD rank selection.
Version check: JAX 0.10.2; Quimb 1.15.1.dev79+gb5e316200;
Autoray 0.11.1.dev9+g1291702f9; Cotengra 0.8.3.dev7+g1d7fd333f;
Symmray 0.4.1.dev11+g1a3481803. Numerical checks explicitly use CPU and
complex128 with JAX x64 enabled, single-threaded BLAS/OMP.

## Validation

- Six orchestration cases cover JAX/Torch defaults, explicit loss overrides,
  the legacy loss alias, explicit JIT opt-out, and unchanged outer settings.
- Two optional integration cases exercise the actual global driver on a
  3x3 D=2 PEPS with a real RZZ gate, using NumPy/JAX inputs. They check JIT
  selection, directional gradients against finite differences, improved
  fidelity, normalized output, unchanged input, bond cap, dtype, and backend.
- An initial test harness used a removed JAX experimental x64 context;
  corrected it to the existing configuration fixture pattern. The initial
  other domain cases passed; the numerical cases were rerun after that fix.
- Final results across focused runs: **273 passed**: 173 PEPS/global/
  safeguard cases; 2 new JAX integration cases (64.63 seconds); 98 batching,
  public API, and package-layout cases. Warnings include existing NumPy shape
  deprecations, Quimb split defaults, and Torch rank-deficient QR diagnostics.
- `python -m ruff check src tests`, local documentation links, and
  `git diff --check` passed.

Native symmetry, GPU, and the full repository suite were not validated by
this change. Earlier broader results belong to their dated task records.

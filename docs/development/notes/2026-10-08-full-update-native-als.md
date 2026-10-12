# Full-update native ALS and 4×4 Ising validation — 2026-10-08

Working-tree follow-up on `develop`, baseline `fe11e24`. This records measured
short-time behavior, not a long-time accuracy guarantee. Nothing was committed
or published. See the [API guide](../../api/optimizers/peps.md) for controls and
the [review handoff](../../../history/2026-10-08-full-update-review-fixes.md)
for the earlier normalization, acceptance, and environment-cache fixes.

## Upstream and backend audit

Installed versions: Torch 2.6.0+cu124, Quimb 1.15.1.dev90+g6a3906cbe,
Cotengra 0.8.3.dev8+g8954240f2, Cotengrust 0.2.1,
Autoray 0.11.1.dev14+g014a3f69a, Symmray 0.4.1.dev15+g0374aaa3c.
The upstream documentation/source audit from this active review was reused;
the Symmray abelian-array documentation URL was unavailable, so official
repository and installed source were inspected instead.

- **Adopt:** Pepsy's shared reduced-pair preparation, positive projection,
  FIT tolerance resolver and reduced ALS. Quimb's public
  `tensor_network_fit_als` accepts prebuilt `tnAA`/`tnAB` overlap networks and
  updates their live tensor views. Its native backend namespace performs the
  local solves. Pepsy now controls complete sweeps when monitoring is enabled,
  reuses those networks, and performs QR/LQ regauging between sweeps. No
  upstream source was modified or copied.
- **Adopt:** Pepsy's existing boundary MPS cuts and reusable Cotengra optimizer.
  Exact partial strip contractions add a second cache level; their source and
  mutation checks invalidate dependent contractions. Cotengra receives the
  same optimizer object for planning and contraction.
- **Defer:** differentiable full-update, native Symmray full-update, GPU
  validation and long-time/high-bond convergence studies. The existing
  full-update contract remains non-differentiable dense Torch complex arrays.

The reduced norm is Hermitianized as `(N + N.H)/2`, then its negative
eigenvalues are clipped to zero. Independent gauges use its square root as
described in [Lubasch et al., Sec. III B](https://arxiv.org/pdf/1405.3259).
With `N = root.H @ root`, the two independent leg unfoldings both use QR;
this is the conjugate/transposed convention of the paper's QR/LQ construction.
Singular environment gauges are skipped. The shared weighted-QR solver also
regauges after each local solve.

## Automatic stopping

Full-update defaults to `rtol="auto"`: 1e-9 for complex128, 1e-5 for complex64.
Let `C` be the squared environment-weighted residual and `T` the target norm.
After a full left/right sweep, stop if `0 <= C/T <= rtol` or
`abs(C_previous - C)/T <= rtol`. `rtol=None` or zero disables these tolerance
checks; `max_iterations` bounds the solve. The cost-change criterion denotes
local stagnation and does not certify a globally optimal fit.

Reports contain the resolved tolerance, actual complete-sweep count,
convergence flag and stopping reason. The underlying solver status is retained
separately when a better SVD guess is kept. A retained guess already within
residual tolerance reports convergence even if a roundoff-worse ALS candidate
was rejected. `success` denotes a usable update, not convergence.

Shared dense reduced ALS exposes this behavior with
`monitor_convergence=True`; its default behavior is unchanged. Unsupported
autodiff/Symmray monitoring raises explicitly. No array is moved to NumPy for
the PEPS evolution, norm projection or ALS solve. Scalar diagnostic reads
remain host synchronization points. Independent reference vectors below use
NumPy/SciPy outside the optimizer.

## 4×4 transverse-field Ising experiment

Open boundaries, 16 spins, 24 nearest-neighbor bonds,
`H = -sum_<ij> Z_i Z_j - sum_i X_i`, initial state `|+>^16`, Torch complex128
on CPU. Each symmetric second-order step applies all X half steps, all ZZ
gates, then all X half steps. ZZ gates traverse columns then rows. Fixed-cap
runs use direct boundary compression with zero cutoff, chi 32 for D=2 and
chi 64 for D=4. ALS uses automatic tolerance and at most 20 sweeps. Global
acceptance checks are disabled; positive-environment best-candidate selection
and final state normalization remain enabled.

Two independent references use all 65,536 amplitudes: exact application of the
same Trotter gates, and SciPy sparse `expm_multiply` for continuous evolution.
The former isolates truncation error; the latter also includes Trotter error.

| Run | End time | Steps / dt | Infidelity vs exact Trotter | Infidelity vs continuous | Energy difference vs exact Trotter |
| --- | ---: | --- | ---: | ---: | ---: |
| Real, D=2, cached | 0.4 | 4 / 0.1 | 1.6530050e-3 | 2.5613865e-3 | +8.04050e-3 |
| Real, D=2, fresh environments | 0.4 | 4 / 0.1 | 1.6530050e-3 | 2.5613865e-3 | +8.04050e-3 |
| Real, D=4, cached | 0.4 | 4 / 0.1 | 1.4694594e-8 | 5.2794964e-4 | -6.29943e-5 |
| Real, D=2, adaptive chi <= 32 | 0.2 | 2 / 0.1 | 1.6877904e-6 | 1.9229738e-4 | -7.22533e-5 |
| Imaginary, D=2, cached | 0.15 | 3 / 0.05 | 4.3106491e-8 | 3.5794699e-6 | +6.77260e-4 |

All five runs match their first Trotter step to numerical precision, remain
finite and normalized within 4e-15, retain Torch dtype/device, and respect
the requested bond cap. Imaginary-time energy falls from -16 through
-17.9991111 and -19.4523393 to -20.6067360. All 48 adaptive gate checks report
boundary convergence. No simulation warnings were recorded.

The final cached/fresh D=2 vectors have phase-aligned norm distance 4.35e-9,
maximum component difference 2.38e-10 and infidelity at machine roundoff.
Fixed-cap real-time runs record 64 strip-cache hits versus 128 rebuilds and
379 outer-boundary hits versus 194 rebuilds. Adaptive fresh confirmation
replaces boundary arrays, so that experiment records zero strip-cache hits;
this is valid invalidation, not a claim that every gate reuses its strip.
ALS requires at most three sweeps for D=2 real time and one for these D=4 and
imaginary-time steps. Reusable Cotengra planning can produce small differences
between repeat runs; the table uses the final run.

At D=4, continuous-time error is dominated by the Trotter approximation
(exact Trotter versus continuous infidelity 5.2823586e-4 at t=0.4). D=2 has
visible truncation error by that time. This supports the implementation on
these short runs; it does not establish long-time, critical-ground-state,
large-D, GPU or all-boundary-engine accuracy.

Temporary reproducer: `/tmp/pepsy_itf_4x4_review.py --case CASE`, with cases
`real_d2_cached`, `real_d2_fresh`, `real_d4_cached`, `real_d2_adaptive`, and
`imag_d2_cached`. JSON records and final vectors are under
`/tmp/pepsy_itf_4x4_results/`. Run with the activated project environment,
`PYTHONPATH=src`, CPU-only execution and one BLAS/OpenMP/Numba thread.

## Regression checks

New regressions guard bulk Torch `.numpy()`/`.cpu()` calls throughout boundary
construction, ALS and normalization for both solvers and complex dtypes;
check exact reconstructed states and automatic stopping; check fixed-budget
opt-out; preserve lazy open environments with monitoring; and verify that
scaling the norm environment from 1e-12 to 1e12 leaves stopping and the fitted
state unchanged. Final combined PEPS and shared reduced-update validation:
**416 passed**, 66 warnings, 109.73 seconds. Default smoke: **94 passed**,
two warnings, 29.57 seconds. Ruff and whitespace checks pass. This includes
the earlier acceptance, normalization, positive-metric and cache regressions.
Logs: `/tmp/pepsy_final_fullupdate_regression.log` and
`/tmp/pepsy_final_review_smoke.log`.

The full repository suite was attempted and interrupted after reporting
24 failures, 184 passes and three skips. All 24 failed identifiers reproduced
on an isolated unchanged `HEAD` archive: `test_bp_compression.py` and
`test_bp_open_series.py` produce 24 failures and 42 passes there. These are BP
fixed-point convergence failures, including sequential post-compression BP,
simultaneous BP and open-loop-series preconditions. No clean full-suite result
is claimed, and their tolerances were not weakened. Logs:
`/tmp/pepsy_full_review_suite.log` and
`/tmp/pepsy_review_baseline_failures.log`.

# JAX L-BFGS: mixed-precision and best-state review

**Fixed:** loss-cache precision mismatches, stale restoration across equal
losses, and nonfinite parameter acceptance in the native JAX L-BFGS runner.
Baseline: `98ff2aaa`; the [API guide](../../api/solvers/gradient.md#native-jax-l-bfgs)
owns the resulting contract. The fixes accompany the
[session handoff](../../../history/2026-11-16-jax-lbfgs-review.md);
Git records the published revision.

## Reproduced findings

| Case | Baseline behavior | Correction |
| --- | --- | --- |
| Float64 parameters, float32 scalar loss | Compiled `lax.cond` fails because cached and initial losses have different dtypes | Use one real working precision for coordinates, objective values and Optax buffers |
| Float32 parameters, float64 scalar loss with offset `1e10` | Reports `final_loss=1e10` for returned parameters whose measured loss is `10000000006.530098`; can stop with stale best parameters | Preserve scalar-loss precision throughout the line search and best tracker |
| Float64 quadratic with offset `1e10`, weights `(1, .001)` | Reports `gradient_tol=1e-8` but restores an earlier equal-loss point with gradient norm about `5.51e-6` | Prefer the latest accepted eligible point when best losses tie; ties do not count as significant improvements |
| Nonfinite unused initial leaf | Returns `gradient_tol` with a NaN parameter | Reject nonfinite initial parameters before objective evaluation |

Inspected installed Optax 0.2.6 `zoom_linesearch` and
`scale_by_zoom_linesearch`: both internal value buffers and the external
cache use the first parameter leaf's real dtype. Replacing only the external
cache dtype is insufficient. **Adopt:** keep a common real working precision
selected from parameter coordinates and the measured scalar-loss dtype.
The objective and returned results receive each leaf in its original dtype;
no objective contraction is silently promoted. All coordinates and optimizer
state stay on the source JAX device. A mixed-precision solve may allocate
larger optimizer state. No upstream source was copied or changed.

Parameter casts are checked at acceptance: a finite float64 working
coordinate can overflow a user's float32 leaf. An `atan` objective remains
finite with zero gradient at infinity, so checking only the loss/gradient
would allow a false convergence. The regression rejects this trial and keeps
the finite initial point.

The environment and upstream audit from the
[feature implementation](2026-11-16-native-lbfgs-mpo-dmt.md#decisions-and-upstream-audit)
remain applicable: JAX 0.8.2, Optax 0.2.6, CPU; no dependency changes.

## Validation

- `tests/test_jax_lbfgs.py`, `tests/test_gradient_solver.py`, and
  `tests/test_gradient_solver_jax.py`: **183 passed**, 19.09 seconds. The
  native L-BFGS file now has 43 cases, including both scalar-loss precision
  directions, signed large offsets, equal-loss restoration, original-dtype
  overflow rejection, nonfinite unused parameters, and complex two-site MPO
  contraction fits. Existing search evaluation counts and trace-count checks
  remain passing.
- Isolated process with two CPU devices: float32 parameters on `TFRT_CPU_1`,
  float64 loss, result remains on `TFRT_CPU_1`, loss `5.88e-14`,
  `gradient_tol`. This validates non-default CPU placement, not GPU execution.
- API/layout: **54 passed, one existing failure** from installed distribution
  `0.5.0` versus checkout `0.6.0`.
- Ruff over `src tests` and `git diff --check` pass. No new full-suite run:
  this follow-up changes only the private native solver, and the previous
  full-suite attempt has a documented unrelated BP/Autoray dispatch failure.
- Fresh state per solve, per-chunk host reporting and lack of guaranteed
  compilation reuse between separate solves retain their documented scope.

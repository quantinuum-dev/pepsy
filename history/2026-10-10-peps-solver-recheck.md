# 2026-10-10 — Review of cached site solves and whole-column L-BFGS

- Scope: recheck the existing ALS/CG/L-BFGS implementations and fix confirmed
  problems, without changing solver defaults or adding automatic switching.
- Branch / baseline: `develop`, `b3e7985`.
- Commit status: working-tree edits only; earlier uncommitted work preserved.

## Finding and correction

The optional CG-to-direct fallback merged the failed CG report over the final
direct report, overwriting its true residual. A reproduced accepted direct
solve incorrectly reported a residual of about 2.61. The returned tensor was
unaffected. The [site solver](../src/pepsy/optimizers/sweep/_als.py) now keeps
the final `relative_residual` and stores the failed iterate's residual as
`cg_relative_residual`. Both explicit and matrix-free fallback paths have
regressions asserting these distinct meanings. The API guide and changelog
describe the correction.

## Review and additional checks

- Rechecked explicit Hermitian and matrix-free norm actions, RHS/index order,
  complex analytic gradients, backend conversion, solver dispatch, and the
  existing forward/backward directional cursor.
- Extended interior-site independent dense least-squares comparisons to both
  L-BFGS variants, both axes, and both boundary engines. Finite-budget L-BFGS
  is checked against optimal physical fidelity rather than gauge-sensitive
  tensor coordinates; existing strict CG/direct comparisons remain intact.
- Added invalid-writeback rollback checks: subsequent cached solves and the
  final state agree with a run that rejects before any writeback. All four
  iterative choices and both axes are covered.
- Added singular supported quadratic and exact warm-start checks for L-BFGS.
- Added whole-column SciPy/Torch L-BFGS checks that explicitly prohibit
  dispatch to one-site ALS and compare the reported loss with exact fidelity.
  An initial five-step Torch test missed its improvement target with the
  shared `lr=0.01` default. The test now explicitly requests `lr=1` and
  `line_search_fn="strong_wolfe"`; the original fidelity threshold is retained.
  This is test configuration, not a change to production defaults.

## Validation

- ALS suite: **140 passed**, 29 warnings in 20.15 seconds, including native
  Torch CUDA/CuPy tests; no skips. Intentional zero-candidate rejection tests
  produce zero-normalization warnings from Cotengra.
- Whole-column L-BFGS follow-up: **4 passed**, 2 warnings in 3.55 seconds.
  Bare SciPy `lbfgs` currently warns about the unused shared `lr` default.
- Broader regression: **460 other checks passed**, covering sweep safeguards,
  contraction performance, PEPS batching/driver, boundary engines, strip/layer
  refinement, public API/package layout, module extraction, and gradient
  solver regressions. That run also included the four whole-column tests
  before their explicit Torch settings were corrected: its raw result was
  462 passed / 2 failed, 56 warnings in 99.50 seconds. Both failures were the
  short-budget Torch improvement checks described above; all four passed in
  the follow-up. Final coverage is **604 distinct passing checks** across
  these runs (140 ALS + 460 other regressions + 4 whole-column checks).
- Final Ruff (`src tests`), `git diff --check`, and review-journal link checks
  passed. No full repository suite was run.

The unchanged-environment upstream audit from the
[previous ALS review](2026-10-10-peps-als-review.md) was reused. No dependency,
contraction, compression, or backend policy changed. Classification: adopt
existing solver/backend APIs and correct local reporting.

## Limits

The default is still explicit `dense-cg`. Automatic ALS-to-whole-column
L-BFGS switching remains a proposal. These are small-system correctness
checks, not a new 4-by-4 D=4 trajectory benchmark or a speed ranking. The full
repository suite was not run; no notebook output was changed.

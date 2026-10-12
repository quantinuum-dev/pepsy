# 2026-10-10 — Review and publication of PEPS sweep solvers

- Scope: user requested review, commit, and push of the new PEPS sweep
  implementation and its downstream comparison notebooks.
- Branch / baseline: `develop`, `3afa925`; examples `main`, `75b4bdf`.
- Commit status: this entry accompanies the reviewed implementation commit.
  Publication is authorized for both repositories; final remote verification
  belongs to the session handoff.

## Review

Reviewed the explicit and matrix-free Hermitian site equations, analytic
L-BFGS gradients, cached forward/backward traversal, rollback, matrix guards,
public solver routing, and diagnostics. The default remains explicit-matrix
CG with no diagonalization or automatic fallback. The compressed-norm and
fallback-report fixes are described in the
[solver review](2026-10-10-peps-solver-recheck.md) and
[live-campaign correction](2026-10-10-peps-compressed-norm-fix.md).

The examples commit includes the four sweep methods, checkpointed campaign
runner, exact-overlap acceptance, and separate comparison plot cells. Corrected
a stale README claim about the number and layout of figures. Existing notebook
outputs were preserved; validation executed temporary copies under `/tmp`.
The running campaign continues from its frozen source without changes.

## Fresh validation

- **614 package tests passed**, 84 warnings, 121.67 seconds: the three ALS
  modules, sweep safeguards/performance, PEPS driver/batching, boundary-engine
  numerics, strip/layer refinement, public API/layout/module extraction, and
  gradient-solver regressions. No skips or failures.
- **13 example tests passed**, 54.10 seconds: independent trajectories,
  exact-overlap accounting, restart equivalence, and scheduling.
- Both notebooks executed successfully: `comparison.ipynb` (56 code cells,
  53 displays) and `variational_comparison.ipynb` (11 code cells, 9 displays).
- Ruff passed for package `src tests` and examples `FullUpdate`; staged diff
  checks and the new PEPS journal/note links passed.
- The earlier unchanged-environment upstream audit was reused. The full
  repository suite and a thread-count performance benchmark were not run.

Only this task's sources, tests, documentation, and notebooks are included.
The unrelated MPO publication-journal edit, local simulation archives, and
Git-ignored live checkpoints are excluded. Both branches matched their fetched
upstream tips before these commits.

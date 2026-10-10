# 2026-10-10 — ALS locality review and explicit-matrix iterative default

- Scope: recheck one-site row/column updates; distinguish D/chi costs; follow
  the user's clarified preference for explicit N, Hermitianization, then an
  iterative solve, with matrix-free optional.
- Branch / baseline: `develop`, `b3e7985`; previous ALS work was already dirty.
- Commit status: working-tree changes only; nothing staged, committed or pushed.

## Final implementation

The earlier [ALS entry](2026-10-10-peps-column-als.md) describes the initial
spectral solver. The final default is now `linear_solver="dense-cg"`: assemble
N from cached strip environments, Hermitianize, and use native CG without
SVD, eigendecomposition, direct factorization, shift, or automatic fallback.
`"cg"` selects matrix-free contraction actions, `"dense"` direct factorization,
and `"pinv"` the old positive-support eigensolve. Optional shifts/fallback
are explicit. All paths retain original-overlap acceptance and one-site updates.

New `_als_cg.py` owns balanced environment halves and native CG. Fixed operator
paths prevent hidden full-matrix constant folding. Small iterative scalar
and cursor kernels use optimal contraction paths. CG checks true residuals
and curvature; failures preserve tensors. Single-precision objective
contractions use double-precision temporary arrays on the same backend/device.
Mutable Torch/CuPy exponent scalars are copied to Python scalars before adding
temporary operator scaling, preventing accidental changes to shared networks.

The [cost audit](../docs/development/notes/2026-10-10-sweep-als-costs.md) derives
explicit assembly O(D^12), dense CG O(k D^8), and matrix-free CG O(k D^10) at
chi=D^2. Whole-column Torch/autodiff L-BFGS has scalar-objective cost O(L D^10),
while current non-Torch sweep gradients use finite differences. No gradient
routing, boundary policy, saved notebook output, or long-time run was changed.

## Validation

The expanded tests cover independent interior one-site reference solves on
both axes/engines, explicit and matrix-free CG, default no-factorization
behavior, native CPU/GPU arrays, singular/indefinite equations, shifts,
exponent scaling, guarded fallback, and public driver summaries.
- Expanded ALS selection: **87 tests**, included in the broader run below.
- PEPS/sweep/refinement/API regression selection: **490 passed**, 54 warnings,
  109.42 seconds. This includes both ALS files, sweep safeguards/performance,
  boundary numerics, PEPS driver/batching, strip/layer refinement, public API,
  package layout, and module extraction. Log:
  `/tmp/pepsy_als_review_regression.log`.
- Ruff `src tests` and `git diff --check` passed. The full repository suite
  was not run. Earlier 441-test results belong to the initial spectral policy.

The upstream audit from the same active task/environment was reused. Public
Quimb contraction expressions and native Autoray linear algebra are used;
the SciPy-backed TNLinearOperator wrapper was inspected but not used because
its public matrix application coerces inputs to NumPy. No dependency edits.

## Limits

Hermitianization does not ensure positivity; CG may reject an indefinite or
poorly conditioned approximate environment. The default iteration cap is 200.
The explicit metric-size guard remains 1024. Matrix-free is optional and has
no general runtime win established. The cost note distinguishes tested
kernel timings from unmeasured trajectories; the full repository suite was
not run.

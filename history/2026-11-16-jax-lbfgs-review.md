# 2026-11-16 — JAX L-BFGS follow-up review

- Scope: review `GradientOptimizer(solver="jax-lbfgs")` again and fix
  reproducible correctness issues. Used the repository maintainer guidance.
- Branch / baseline: `develop`, `98ff2aaa`, already pushed by the prior task.
- Publication: the user authorized committing and pushing these fixes on
  2026-11-16. This handoff accompanies the fix commit; Git records its final
  revision. Unrelated sampling/test-harness/journal edits and their changelog
  entry remain outside the commit.

## Findings and fixes

- Optax zoom types internal value buffers from the first coordinate leaf.
  Mixed scalar-loss and parameter precision caused either a compiled carry
  mismatch or silently rounded cached losses. Use common real precision
  internally, retaining original parameter dtypes at objective/return boundaries.
- Equal rounded best losses restored an old point despite later gradient
  convergence. Restore the latest accepted eligible minimum; keep significant
  improvement and patience tracking strict.
- Reject nonfinite initial leaves even if the objective ignores them. Check
  accepted trials after casting to original dtypes, avoiding overflow masked
  by a finite saturating objective.
- Added 20 native-solver regressions and clarified input/working precision,
  memory and restoration semantics in the API guide and changelog. The native
  solver file now contains 43 test cases.

## Validation

- All three solver modules: **183 passed** in 19.09 seconds, including
  exact objective-evaluation counts, complex/mixed precision and two-site
  MPO contraction fits. No DMT or other optimizer implementation changed.
- Non-default placement verified in an isolated two-CPU-device process;
  result remains on device 1. GPU execution was not tested.
- API/layout: **54 passed, one known installed-version mismatch** (`0.5.0`
  versus checkout `0.6.0`). Full suite was not repeated for this scoped review;
  its prior unrelated BP dispatch failure is recorded in the earlier handoff.
- Ruff, local Markdown links and diff whitespace checks passed at handoff.
- [Reproductions and rationale](../docs/development/notes/2026-11-16-jax-lbfgs-review.md).

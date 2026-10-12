# 2026-09-28 — Shared FIT tolerance policy

- Scope: continue the user's readability and duplication cleanup.
- Branch / baseline: `develop` / `f410fa0`, preserving the uncommitted
  [FIT](2026-09-28-fit-readability.md) and
  [four-domain](2026-09-28-bp-vmc-mpo-pepo-readability.md) passes.
- Commit status: working-tree changes; not staged, committed, or pushed.

## Changes

- Consolidated three identical FIT tolerance parsers in the existing private
  cutoff utility. MPS, tree, and stabilizer-MPS retain their method hooks.
  Backend dtype is still read only for `"auto"`; explicit values and `None`
  keep the original conversion, errors, and behavior.
- Corrected contradictory MPS/tree and standalone FIT documentation about
  non-unitary stopping.
  MPS retains automatic stopping; tree disables it for non-unitary or
  unknown-target-norm updates unless an explicit numeric tolerance is given.
  Added concrete stabilizer FIT thresholds beside its truncation controls.
- Clarified Torch `local_observables` input/return shapes, amplitude ownership,
  gradient policy, and the distinction from saved-sample statistics. Its
  executable body is unchanged.
- Numerical kernels, public signatures, defaults, dependency profiles,
  tests, and CI selections are unchanged. No source modules were added.

## Validation

- Existing focused suites: `test_optimize_mps`, `test_mps_compression_modes`,
  `test_tree_fit_messages`, `test_tree_api_consistency`, `test_stabilizer_tn`
  → **778 passed, 1 skipped, 13 warnings in 14.70s**. The skip requires CuPy.
- 540 direct baseline comparisons across three optimizers, nine dtype
  representations, and twenty inputs match return values, exception types,
  messages and causes, and lazy dtype access counts.
- Shared parser AST equals each original body after replacing the optimizer
  dtype lookup with its argument. Optimizer signatures, decorators, and other
  method bodies are unchanged. Existing call sites still dispatch through
  the optimizer hooks.
- Ruff, the two CI mypy targets, all 29 local Markdown links in changed
  guides/handoffs, and whitespace checks passed.
- Full combined working-tree suite:
  `MPLBACKEND=Agg python -m pytest -q -ra -o addopts=''` →
  **5,168 passed, 129 skipped, 791 warnings in 499.64s (8m19s)**, exit 0.
  Skips require unavailable CuPy/CUDA/Metal, multiple MPI ranks, or two
  configured XLA host devices. Those configurations remain unvalidated.
- Default smoke: `MPLBACKEND=Agg python -m pytest -q` → **89 passed**, two
  compatibility-alias warnings, **16.80s**, exit 0. Final whitespace review
  passed; nothing is staged, committed, or pushed.

## Limits

This consolidates scalar policy; it does not unify domain-specific stopping
schedules or change the meaning of truncation versus convergence tolerances.
The preceding compatibility audit is reused for this continuing task;
no upstream API call or numerical default changes in this pass. No runtime
speedup or hosted CI result is claimed.

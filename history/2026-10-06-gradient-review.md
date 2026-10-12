# 2026-10-06 — Gradient optimizer review

- Scope: user requested a careful review of the gradient optimizer, its
  integrations, bugs, and greater use of “auto,” interpreted as Autoray.
  An optional clarification was sent; the review proceeded with that stated
  assumption. No implementation/refactor was requested or performed.
- Branch / baseline: `develop` / `2abb1e8`.
- Status: new review note and this handoff are uncommitted; all pre-existing
  working edits preserved. Nothing staged, committed, pushed, or published.

## Findings

[Detailed report and reproductions](../docs/development/notes/2026-10-06-gradient-optimizer-review.md).

Eight confirmed issues: native JAX complex gradient direction; Torch/FD Adam
terminal update and result mismatch; false SciPy convergence on invalid loss;
ignored SciPy/TNC budgets including PEPS maxeval integration; trust-constr
callback crash; silently ignored JAX bounds; qMERA's negative-energy mismatch
with NLopt nonnegative best selection; unchecked imaginary native JAX loss.

Autoray can consolidate shared array operations and device-resident snapshots.
Native Torch currently invokes the host packer even without bounds (six calls
in a four-step probe). Preserve explicit autodiff, gradient convention,
ownership, and host-solver adapters rather than attempting an all-Autoray
optimizer abstraction. Reuse the existing backend namespace helper.

## New validation

- Solver suites: **118 passed**, 12.59 s; explicit CPU rerun **118 passed**,
  12.42 s, with JAX preallocation disabled.
- PEPS safeguard/performance suites: **45 passed**, 7.67 s.
- Four qMERA optimizer tests: **4 passed**, 99 deselected, 6.32 s.
- Small independent probes reproduced the reported defects, including an
  actual PEPS SciPy call receiving maxiter=30 after requesting maxeval=1 and
  an actual negative-energy qMERA NLopt solve reporting best_loss=inf.
- No source/test changes; documentation links and whitespace checked.
- An initial Autoray capability probe triggered JAX default GPU preallocation
  OOM messages but completed and exited. Final numerical solver probes used
  explicit CPU selection. No production run or GPU performance validation.
- Full suite not run. Passing existing tests does not invalidate the uncovered
  failures; the report identifies the missing behavioral coverage.

## Scope boundaries

Applied maintainer guidance and the qMERA guide to inspect the energy caller;
reviewed solver API, closest tests, and previous solver handoffs. Read-only
searches identified Gaugy callers; no sibling modifications or complete Gaugy
validation. PEPS global optimization uses Quimb TNOptimizer through a separate
adapter and is not automatically affected by all gradient-module findings.

Recommended next implementation order: gradient/result correctness first,
invalid-loss and budget handling second, then backend-array consolidation.
These remain proposals subject to the user's requested scope.

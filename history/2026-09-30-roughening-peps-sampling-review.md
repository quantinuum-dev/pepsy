# 2026-09-30 — Roughening PEPS sample defaults and review

- Scope: user requested norms and 4096 Z samples only at t=6 by default,
  batched local-expectation verification, and a sampler-settings review.
- Branch: Pepsy develop; downstream examples main. Existing concurrent work
  was preserved. This task's edits remain uncommitted and were not published.

Updated downstream scheduling, help, documentation, and regressions. Verified
that all site-Z terms enter one Quimb `compute_local_expectation` call via
Pepsy and that imbalance reuses the returned values. Reviewed Pepsy's current
sampler without editing its implementation. Automatic cutoffs remain owned
by Pepsy; sampling caps and positivity repair define the weighted proposal.

A bounded planning-only probe estimates roughly 9.75e8 contraction operations
per full-D=4 9x10 amplitude. This is not a runtime or production-convergence
measurement. Recommend boundary-cap convergence, weighted-observable checks,
ESS/repair diagnostics, and throughput measurement before increasing chunks.

Downstream validation: 318 passed, 12 existing MPS failures; all 47 PEPS and
39 sweep tests passed. Changed-file Ruff and diff checks passed; six unrelated
benchmark lint issues persist. No simulation production job was launched.

See the [review and validation details](../../pepsy_examples/experiments/mps_magnetization/benchmark/docs/development/notes/peps_simple_update.md#2026-09-30--t6-defaults-and-direct-sampling-review).

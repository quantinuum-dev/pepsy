# 2026-09-30 — Further committed TreeSampler correctness review

- Scope: user requested a careful review after the shared-density commit.
- Branch / baseline: `develop`, `7773d2e`; sampler unchanged from `623ccc3`.
- Status: review only. Added evidence/handoff documents; no library or test
  changes, staging, commit, or publication in this turn.

Reproduced five correctness issues: complex64 conditional underflow biases
long-tree sampling; native Symmray single-site flip ratios instead flip all
sites together; detached normalization gives incorrect Torch scoring
gradients for general trainable inputs; zero/non-finite/extremely scaled dense
inputs can return invalid samples; configuration coercion/indexing silently
changes invalid values, especially on CuPy. All predate the shared-density
optimization, verified by unchanged function ASTs and a baseline underflow
probe. The known amplitude visitor retention is still present. Also recorded
the shallow snapshot caveat for raw in-place source mutation.

Fresh focused sampler/readout checks: **152 passed, 1 skipped**, two expected
deprecated `dmrg1` warnings. Skip requires two GPUs. The new small reproducers
expose gaps in that passing suite; full-package checks were not run.

See [source locations, concrete reproducers, proposed remedies, and limits](../docs/development/notes/2026-09-30-tree-sampler-correctness-review.md).
Installed versions unchanged; reused the active task's upstream audit.
Local documentation links and whitespace checked. Next implementation work
should add focused regressions for these failures and account for numerical
scaling in the proposed vector/factor/prefix algorithms. This recommendation
does not claim that any of those fixes or prototypes are integrated.

# 2026-10-06 — PEPS automatic batching and review fixes

- Scope: user's requested automatic gate absorption, integer batch sizes,
  explicit NLopt defaults, and fixes from the preceding optimizer review.
- Branch / baseline commit: `develop` / `8f7c896`.
- Commit status: uncommitted working-tree changes; nothing staged, committed,
  pushed, or published. No sibling-repository files changed by this task.

Implemented `k_2q_batch="auto"` as the default, preserving gate order,
disjoint two-site support and a `2*chi` multi-gate target threshold. Exact
singleton targets may exceed that threshold; integers retain count-based
batching. Added explicit NLopt settings, norm recomputation and bounded
diagnostic retries, exact-target alias protection, physical-index dispatch,
sweep diagnostics retention and temporary mode overrides. Updated the API
guide, changelog, focused tests and
[implementation evidence](../docs/development/notes/2026-10-06-peps-auto-batching.md).

Fresh final validation: **402 passed**, 15 warnings, no skips across PEPS
batching/optimizer, global optimizer, boundary preparation, public API and
package-layout suites. Ruff, diff and local documentation-link checks passed.
Native Torch CPU U1 batching agrees with separate updates; dense-reference
tests cover gate order and the previously failing identity update.
No full suite, GPU or broad PEPO/cyclic validation or performance benchmark.

Prior review findings remain recorded in their dated report; this handoff
records their implementation follow-up. Existing untracked histories and
concurrent gradient-solver edits were preserved. See the evidence note for
ownership, retry costs and the exact-singleton/temporary-memory limitations.

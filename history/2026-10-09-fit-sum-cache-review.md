# FIT sum cache review, 2026-10-09

## Scope and status

User requested a performance/caching review of the newly added sum-target
FIT. Continued the local uncommitted implementation on `develop`, baseline
`21f883b`. No staging, commits, pushes, or sibling-repository changes.
Unrelated existing edits were preserved.

## Changes

- Existing directional environment reuse was verified through 3→2→1 sweep
  transitions. Six alternating sweeps build the fixed side once and reuse it
  five times; a new invocation rebuilds state-dependent environments.
- Sum targets now cache ordered component references per visited block.
  No precontracted target block is stored; this avoids increasing layered
  target memory or retaining a stale numerical contraction graph.
- Build one fitted bra per environment-site update and share it across all
  terms. Do not cache it across updates to the fitted state.
- Optional verbose target norms use the Hermitian triangle once per run.
  All cross terms are retained. The cache resets on each public run to keep
  target data changes and fresh autodiff graphs correct.

Regression tests count actual target selections, environment builds, bra
conjugations, and diagnostic contractions, as well as numerical results.
For three terms and three verbose sweeps, closed diagnostic contractions
drop from 39 to 18 (6 target pairs once plus 4 per sweep).

## Validation

Using `~/envs/py312`:

- Sum tests: **55 passed**, including dense/Torch, native Z2 and U1U1 MPS/MPO,
  cancellation, active windows, cache lifecycle, and ownership.
- Sum + hotpaths + schedules + MPS FIT kernels + MPO optimizer + public API +
  package layout: **373 passed**, eight existing/intentional warnings, 36.09 s.
- `python -m ruff check src tests`: passed.
- `git diff --check`: passed.

No full suite, CUDA, or compilation claim. CPU timings and the unchanged
dependency audit are recorded in the [evidence note](../docs/development/notes/2026-10-09-fit-sum-cache-review.md).

# 2026-11-20 — Review DMT region protection and share partial contractions

- Scope: review inherited regions, shape planning/cache safety, and protected
  rank requirements; implement a measured, low-risk improvement.
- Branch / baseline: `develop`, `eaedf78942b7931c07a9086e5bb1bea5ef24f65f`.
- Status: working-tree changes only; no commit or push requested this turn.

The shared ordinary/Pauli `_region_vectors` helper now reuses normalized
partial contractions between strings within a call, retaining the same
constraints, ordering, cutoff and verification policies. Numerical maps are
not retained across tensor writes or canonical gauge moves. Existing static
crossing plans and changed-range trace environments remain in place.

Expanded the physical-region API guide with subset inheritance, uncovered
straight triples/joint regions, cache ownership and conservative rank budgets.
Added 13 focused passing regressions, including lossy joint six-spin marginal
preservation in both bases and independent Torch complex128 finite differences.

The [review record](../docs/development/notes/2026-11-20-dmt-region-review.md)
contains dependency audit, helper timings, complete-operation limits, and
remaining opportunities. Helper gains are 1.14–1.39× for sampled plaquettes
and 1.65–1.95× for separated triples. Complete-compression speedup is not
established. No scientific guarantees were expanded.

Validation: 441 passed, one skipped, eight existing warnings in the focused
region/lattice/window/constraint/refinement/optimization, ordinary-MPO/replay,
Pauli/cutoff, check-cost and preservation-policy selection. This includes the
13 new cases. Ruff `src tests`, local Markdown links and whitespace checks
pass. No full-suite or GPU validation is claimed.

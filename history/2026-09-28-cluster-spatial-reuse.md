# 2026-09-28 — MPO/PEPO Hamiltonian-aware spatial reuse

- Scope: user requested translation/rotation-aware cluster construction in
  both Pepsy cluster representations.
- Branch / baseline: `develop`, `4e398e4`.
- Status: uncommitted, unstaged, unpublished; earlier and unrelated edits retained.

## Changes

Added conservative labeled-graph local-target reuse, compiled structural plans,
fresh per-evaluation numerical caches and backend-native site transport.
`spatial_reuse=True` defaults on, with False for independent comparisons.
Independent parameters, factor order, periodic bond occurrences and residual
assembly are preserved. Both builders retain all physical placements.
Sparse PEPO slot support keeps symmetry compilation inexpensive.

## Validation and limits

175 focused MPO/PEPO/API/layout tests pass, including 15 new symmetry checks;
Ruff and whitespace checks pass. The full CPU-only suite passed: **5199 passed,
105 skipped, 777 warnings**, 26:29. The first GPU-enabled broader attempt was
interrupted after a JAX GPU allocation/cuSolver failure; that case passed on
CPU. Neither test runner remains active. The upstream audit, resource
limitation and scoped benchmark are recorded in the
[dated evidence](../docs/development/notes/2026-09-28-cluster-spatial-reuse.md).
The 5x6 open uniform X + ZZ example at p=4 reduces 492 placed local targets
to six evaluated representatives. This is not an end-to-end optimizer speedup.
MPO collection-budget fallback and compression approximations remain.
Unsupported opaque MPO term forms fall back to independent evaluations.
No Gaugy changes, commits, publication, or running-job changes.

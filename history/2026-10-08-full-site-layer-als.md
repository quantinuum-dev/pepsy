# 2026-10-08 — Fixed-layer full-site PEPS refinement

- Scope: implement broader FU refinement incrementally, benchmark cost, and
  remove new environment gauge conditioning at the user's explicit request.
- Branch / baseline: `develop`, `21f883b`.
- Commit status: working-tree changes; nothing staged, committed, or published
  by this task. Existing MPS/JAX changes and their documentation are preserved.

## Changes

- Bounded exact gate-window targets, FU warm starts, and optional full-site
  row/column ALS cycles with independent whole-cycle acceptance and rollback.
- Native Hermitian positive-support pseudoinverse, directional strip cursors,
  separate norm/overlap caps, and target-only norm reuse.
- A local matrix-size guard skips refinement before dense allocation; layer
  mode also skips unnecessary target construction. Optional scalar balancing
  preserves the PEPS state through exponent compensation.
- Pair environment gauges now default to off. Existing explicit pair
  `gauge=True` remains compatible; the new refinement has no gauge option.
- API docs, module ownership map, changelog and
  [numerical evidence](../docs/development/notes/2026-10-08-full-site-layer-als.md)
  describe behavior, defaults and performance limits.

## Validation

- Broad PEPS/shared reduced ALS/native CuPy plus public API/layout selection:
  **488 passed**, 100 warnings, 75.03 seconds.
- Final focused FU/strip/layer checks after extra cache/scaling assertions:
  **93 passed**, one warning, 24.53 seconds (overlapping selection).
- Ruff `src tests` and `git diff --check` passed. No fresh full-repository run.
- Exact 3x3 reference measurements show improvements at D2/3; D4 strip and
  layer both reach roundoff in this tiny case. Interior dense kernels were
  measured through D6; end-to-end examples through D4. See the evidence note
  for boundary-estimate versus exact-fidelity distinctions.

## Decisions and limits

Keep full-layer refinement opt-in: it can improve accuracy but can cost more
than strip refinement without a useful gain. Default matrix-size limit 1024
excludes interior D6+; dense eigensolver scaling remains D12. This is not a
matrix-free solver or an optimal multi-site PEPS method. Large-D GPU timing,
long-time evolution, rank enrichment and overlapping patches remain unverified
or deferred; they are not implemented silently under this option.

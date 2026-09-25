# 2026-09-24 — batch-exact mode alias

- Scope: make MpsOptimizer(mode="batch-exact") behave as exact-batch.
- Branch / baseline commit: develop, 9a56fc8.
- Commit status: committed locally in this change; not pushed.

## What changed

- Normalize batch-exact to exact-batch in MpsOptimizer before mode validation. Construction, set_mode, and run(mode=...) use the same exact-batch replay, metadata, backend, and transition path.
- Extend the existing Gibbs and layout replay restrictions to the alias before those entry points construct an optimizer.
- Document the alias in the [MPS API guide](../docs/api/optimizers/mps.md), [implementation map](../docs/development/modules/optimizers.md), [MPS skill](../.github/skills/mps-optimizer/SKILL.md), changelog, and [exact-batch note](../docs/development/notes/mps_exact_batch.md#batch-exact-mode-spelling-2026-09-24).

## Validation

- Focused exact-batch suite: 29 passed, including alias replay, mode override, transitions, Gibbs, persistent-layout, and layout replay rejection.
- Public API, package layout, and MPS layout-upgrade suites: 73 passed.
- Five targeted MPS mode-normalization checks passed. Ruff, skill quick validator, catalog validator, and git whitespace checks passed.
- An optional broad `test_optimize_mps.py -k mode` sweep was stopped during a sustained CPU-heavy case. It emitted failure markers around the interruption but no completed failure report, so it establishes no broad-suite result.
- No numerical kernel or dependency was changed. The earlier same-task upstream audit remains applicable.

## Unrelated work

- The pre-existing tree-layout changes remain unstaged and uncommitted. No remote push was requested.

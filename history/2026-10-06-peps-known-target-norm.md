# 2026-10-06 — Skip the unitary target norm measurement

- Scope: user requested avoiding target norm contractions after normalized
  retained states undergo unitary gates.
- Branch / baseline: `develop`, `8f7c896`.
- Commit status: working-tree edits and new files; nothing committed or pushed.

## Changes

Unitary `run()` checks now supply target norm one, avoiding its norm
contraction and forwarding the known norm to variational cleanup. Explicit
`infidelity_kwargs={"norm_target": None}` restores measurement. Nonunitary
runs, disabled final normalization, and standalone metric calls default to
measurement. Output normalization and `maxeval=50` remain unchanged.
Updated [API guide](../docs/api/optimizers/peps.md), changelog and tests.

See the [evidence note](../docs/development/notes/2026-10-06-peps-known-target-norm.md)
for the measured cost/accuracy tradeoff. A low-cap 4x4 identity case now raises
on a negative estimate under assumed unit norm; accurate caps or explicit
target-norm measurement resolve it. No silent target-contraction fallback.
This supersedes target measurement in the
[preceding handoff](2026-10-06-peps-unitary-targets.md).

## Validation

Fresh selection: **418 passed**, 19 warnings, no skips, 23.45 seconds across
`test_peps_optimizer_batching.py`, `test_optimize_peps.py`,
`test_optimize_global.py`, `test_prepare_boundary_inputs.py`,
`test_public_api.py`, and `test_package_layout.py`.
Ruff, local documentation link checks and `git diff --check` passed.
No full suite or GPU testing. Concurrent solver edits were preserved.

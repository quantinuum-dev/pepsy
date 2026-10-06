# 2026-10-06 — PEPS unitary targets and retained-state normalization

- Scope: user clarification of RZZ 2D versus general-gate 4D target growth,
  `maxeval=50`, and normalization of retained PEPS using `normalize_chi`
  without rescaling unitary targets.
- Branch / baseline: `develop`, `8f7c896`.
- Commit status: working-tree edits and new files; nothing committed or pushed.

## Changes and evidence

Implemented gate-dependent automatic budgets, exact nearest-neighbor diagonal
rank ceilings, the 50-evaluation local default, and unitary target/output
normalization policy in
[optimizer.py](../src/pepsy/optimizers/peps/optimizer.py).
Updated [API documentation](../docs/api/optimizers/peps.md), changelog, and
focused tests. See the [evidence note](../docs/development/notes/2026-10-06-peps-unitary-targets.md)
for precise native/traced/routed limitations and rationale.
This follows and supersedes the defaults recorded in the earlier
[batching handoff](2026-10-06-peps-auto-batching.md).

## Validation and limitations

Fresh selection: **410 passed**, 15 warnings, no skips across batching,
PEPS/global optimizer, boundary input, public API and package-layout suites.
Ruff, documentation link checks and `git diff --check` passed.
Dense and Torch exact-target reconstruction and retained-output norm checks
pass; native fermionic regression remains passing. No full suite or GPU run.
Finite-cap normalization still needs convergence checks with `normalize_chi`.
Concurrent solver edits were preserved and are outside this task's scope.

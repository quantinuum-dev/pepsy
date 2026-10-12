# 2026-10-06 — Implement two PEPS sweep safeguards

- Scope: user authorized invalid-initial-loss handling and validation of
  local solver output before tensor writeback.
- Branch / baseline: `develop`, `8f7c896`.
- Commit status: working-tree changes only; prior PEPS changes and concurrent
  solver work preserved. Nothing staged, committed, or pushed.

Sweep now distinguishes an invalid initial boundary estimate from convergence.
The PEPS driver retains its normalized warm start and records failed cleanup.
Local nonfinite parameters or invalid losses are rejected before writeback,
with explicit diagnostics and native block handling. Caps, FIT budgets,
NLopt maxeval=50, and known unitary target-norm policy remain unchanged.

Added 23 safeguard regressions and updated API docs/changelog. Fresh focused
validation: 431 domain/API/layout cases plus four native U1/U1U1 safeguard
cases passed (435 distinct cases). Ruff, local review links, and diff checks
passed. No full suite or GPU validation.

See the [implementation evidence](../docs/development/notes/2026-10-06-peps-sweep-safeguards.md)
for exact scope, real failure reproduction, test fixture correction, and
remaining unrelated findings. This entry supersedes the preceding review's
statement that these two safeguards were only proposed.

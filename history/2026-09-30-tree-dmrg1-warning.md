# 2026-09-30 — Warn when selecting legacy Tree DMRG1

- Scope: user requested a warning directing `dmrg1` callers to `dmrg`.
- Branch / baseline: `develop` / `f01f596`; working-tree edits only.
- No staging, commit, publication, dependency or numerical-kernel changes.
  Existing unrelated changes were preserved.

TreeOptimizer now emits a visible `FutureWarning` at construction (including
the legacy `two_site_mode` selector) and explicit `run(mode="dmrg1")` selection.
The message recommends `dmrg` for one-site refinement and explicitly states
that `dmrg1` retains its two-node growth warm-up. Normalization remains pure;
copies and replay without an explicit mode override do not repeat the warning.
Updated the API guide and changelog. The preceding
[one-site default](2026-09-30-tree-dmrg-one-site-default.md) remains unchanged.

Validation: Tree API consistency, MPS-parity and replay-composition suites:
**184 passed** in 6.68 s, including seven new warning cases. Checks cover the
user-facing warning location/count, silent current names, exact state replay
and the unchanged legacy `(2,2,1,1)` schedule. Ruff `src tests` and
`git diff --check` pass. Log: `/tmp/pepsy-tree-dmrg1-warning.log`.
CPU-only validation; the broad Tree/API suite was not repeated for this
warning-only follow-up. Its earlier results and metadata failure are recorded
in the linked handoff.

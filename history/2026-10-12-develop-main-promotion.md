# 2026-10-12 — Promote develop to main

- Scope: user-authorized merge, commit and push, ending on `develop`.
- Baselines: `main` / `origin/main` at `22bff20`; `develop` / `origin/develop`
  at `6093f5ad`. Both working trees were clean.
- This handoff accompanies the promotion merge. Remote publication follows
  the merge commit; final push results and branch equality are checked afterward.

## Merge and validation

`main` was an ancestor of `develop`, 108 commits behind with no unique commits.
The non-fast-forward merge had no conflicts. Before adding this handoff,
the merged index and working tree matched `develop` exactly.

Reused today's unchanged-source smoke check: **94 passed**, two expected
compatibility-alias deprecation warnings, using the activated `genpy`
interpreter. Skill catalog and handoff link/whitespace checks passed.
The full numerical, GPU and MPI suites were not rerun for this promotion.

Publish `main`, then return to `develop`, fast-forward it to the promoted tip
and push it, preserving one shared history and all published commits.
The device-local override remains excluded. No numerical source was edited.

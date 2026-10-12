# 2026-09-30 — Align Tree DMRG aliases and runner defaults

- Scope: user requested no notebook CPU cap, identical Tree `dmrg`/`dmrg1`
  one-site defaults with a migration warning, and default Tree settings in
  roughening.
- Baselines: Pepsy `develop` / `50d319e`; examples `main` / `d9c0b6f`.
- Status: working-tree edits, not committed or pushed. Existing sampler,
  PEPS reference, documentation, notebook backend and output edits preserved.

Implemented the shared generic-DMRG policy for `dmrg1`, removing its special
growth warm-up and iteration restriction. Kept explicit settings identical
across aliases. Roughening now resolves defaults for both tree names and no
longer injects the old `dmrg1` schedule. Notebook `threads=None` preserves
ambient CPU settings. Public API docs, changelog and notebook explanation
reflect the new behavior.

This supersedes the retained-legacy-schedule decision in the earlier
[warning handoff](2026-09-30-tree-dmrg1-warning.md).
See [implementation and validation evidence](../docs/development/notes/2026-09-30-tree-dmrg1-alias.md).
New checks: 1533 tree/API passes with 101 skips and the known metadata failure;
522 benchmark passes with 6 skips and 4 known bubble failures; 16 final
runner-default/override/sweep passes. Package/changed-file Ruff and notebook
validation pass. No production launches or GPU claims.

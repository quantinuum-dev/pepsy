# 2026-10-06 — Restore the roughness illustration dependency

- Scope: fix `ModuleNotFoundError: dw_cell_complex` in the rough-exact MPS
  notebook's `plot_longest_path_roughness_examples()` cell.
- Baselines: Pepsy `develop` at `8f7c896`; examples `main` at `501ba11`.
- Commit status: working-tree edits only; nothing staged, committed or published.
- Follow-up to [transferred result plotting](2026-10-06-roughening-transferred-plots.md).

The dependency is present on eiHt-compress's local `ising-dw-postprocessing`
branch at `b9cd03c`, but absent from its active `main` checkout. Created a
separate detached worktree, `eiHt-compress-ising-dw`, beside the existing
checkout. Both dependency worktrees remain clean; the active checkout stays
on `main`. No dependency source or shared Python environment was changed.

Updated the examples plotting helper to discover either sibling checkout,
respect explicit `EIHT_COMPRESS_ROOT` overrides, load the selected file without
changing `sys.path`, and provide an actionable missing-dependency error.
Documented the separate-checkout command. Added regressions for discovery,
override precedence, cached-module replacement, and rendering all four examples.
Existing unrelated edits and source-notebook outputs were preserved.

New validation: plotting suite **90 passed, no skips**, including the four
previously skipped wall tests. One existing empty-legend warning remains.
Changed-file Ruff and `git diff --check` pass. The actual notebook setup and
failing cell executed successfully into
`/tmp/pepsy_examples_executed/roughness_import_fix.ipynb`, producing four PNG
outputs. Full notebook/data-analysis execution was not run for this import fix.
The earlier benchmark-wide Ruff result still contains six unrelated issues.

An already-running kernel must rerun its setup cell to reload the helper.

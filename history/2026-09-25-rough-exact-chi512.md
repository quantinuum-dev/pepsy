# 2026-09-25 — Add 5×6 χ=512 to the rough-exact notebook

- Scope: include the completed 5×6 χ=512 MPS sweep in `rough_exact.ipynb`.
- Branch / baseline: Pepsy `develop`, `80f451a`; examples `main`, `301159a`.
- Commit status: working-tree edits only; nothing staged, committed, or published.
- Follow-up to the [saved-data review](2026-09-25-roughening-sweep-review.md).

## Changes

- Added χ=512 to the examples plotting helper's default run registry, styling,
  and MPS norm-survival plot. The existing observable, time-trace, wall, and
  saved-shot paths now include the series automatically.
- Updated the notebook's two χ-list descriptions and the examples READMEs.
  Corrected the root README's stale 9×10 comparison description to 5×6.
- Extended the existing loading/legend and fidelity plot checks to χ=512.
- Preserved all prior working-tree edits and source-notebook outputs. Source
  plots must be rerun to refresh displayed results. No simulation was rerun.

## Validation

- Plot-helper suite: 39 passed; one empty-legend warning in an existing
  conditional-wall test.
- Ruff on the two changed Python files and `git diff --check`: passed.
- Compared this turn's changes with a temporary pre-edit baseline; only the
  intended five examples files changed, and all notebook outputs were preserved.
- Full notebook execution into `/tmp/pepsy_examples_executed/rough_exact_chi512.ipynb`
  passed without cell errors. All 12 χ=512 angles and all 12 retained wall
  snapshots loaded at t=5; the executed notebook also passes schema validation.
- No numerical package implementation changed; package/full suites were not run.

## Remaining scope

The newer t=10 small-lattice sweeps and dt=0.4 are still outside the notebook's
default selection. This follow-up only adds the requested 5×6 χ=512 series.

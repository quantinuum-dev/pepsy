# 2026-09-25 — Entropy versus time in rough_exact

- Scope: add a new notebook plot comparing entropy against time across system
  sizes and Trotter steps.
- Branch / baseline: Pepsy `develop`, `80f451a`; examples `main`, `301159a`.
- Commit status: working-tree edits only; nothing staged, committed, or published.

## Changes

- Added `exact-entropy-time-note` and `exact-entropy-time` cells after the
  dense time traces in the examples `rough_exact.ipynb` notebook.
- The helper reads scalar checkpoint entries from the t=10, 8192-shot
  exact-batch sweeps: 4×4, 4×5, 5×5, each at dt=0.05, 0.25, 0.4.
  Completed and running cases use their actual saved time grids.
- `THETA_INDEX` chooses the common angle. Colors identify size; solid,
  dashed, and dotted lines identify timestep. Partial curves are labeled
  and stop at the latest checkpoint. Missing angles are reported.
- The plotted quantity is middle snake-chain entropy in bits, with cuts
  8|8, 10|10, and 12|13 sites. No shot error bars apply. The 5×6 runs did not
  record entropy and are not included.
- Added the rendered plot only to the new cell, preserving every existing
  cell and output. Re-executing the new cell refreshes its checkpoints.
  `FIGURES["entropy_time"]` exposes the figure for export.
- Updated the examples benchmark README. No simulation or numerical package
  code was changed, and the three 5×5 background sweeps remain active.

## Validation

- Existing plot-helper suite: 39 passed, one existing empty-legend warning.
- Ruff on the changed helper and `git diff --check`: passed.
- Executed notebook setup plus the new entropy cell in a temporary notebook;
  confirmed nine curves with monotonic saved times, finite values, and three
  line styles. At validation, theta_01 of 5×5 dt=0.05 reached t=3.6; its other
  two timestep cases reached t=10. This does not imply full sweep completion.
- Notebook schema validation passed. Compared every original notebook cell
  against the pre-edit snapshot: all were unchanged.
- Visually checked `/tmp/rough_exact_entropy_time.png`; PDF also exported.
  Temporary cell execution is in
  `/tmp/pepsy_examples_executed/rough_exact_entropy_validation.ipynb`.
- Did not rerun the full notebook or numerical package suites; the change is
  a saved-data plot and does not evolve states.

# 2026-09-25 — Separate entropy time and fixed-time angle cells

- Scope: keep entropy versus time across sizes/timesteps in its own cell and
  add entropy versus Δθ at fixed t=5 in a separate cell.
- Branch / baseline: Pepsy `develop`, `80f451a`; examples `main`, `301159a`.
- Commit status: working-tree edits only; nothing staged, committed, or published.
- Follows the [entropy-time cell](2026-09-25-rough-exact-entropy-time.md).

## Changes and behavior

- Added `exact-entropy-delta-theta-note` and `exact-entropy-delta-theta`
  immediately after the entropy-time cells in `rough_exact.ipynb`.
- Both plotting methods share the saved middle-cut entropy reader and the
  exact-batch sweep definitions. No evolution or simulation settings changed.
- The new plot selects t=5 with absolute time tolerance 1e-8 and no
  interpolation. dt=0.4 has no t=5 step; the plot and text report it unavailable.
  Pending angles are NaN gaps, and partial legends report available counts.
- Saved the new rendered angle plot in its cell; all pre-existing notebook
  cells and outputs were preserved. The new cell refreshes independently of
  `THETA_INDEX` and `TARGET_TIME`. Export via `FIGURES["entropy_delta_theta"]`.
- Updated the examples benchmark README.

## Validation

- Plot-helper suite: 39 passed, one existing empty-legend warning. Ruff on the
  helper and `git diff --check` passed.
- Executed setup and both entropy cells in a temporary notebook. Verified
  nine time curves, all angle-plot points against exact t=5 checkpoint values,
  missing-angle gaps, the dt=0.4 exclusion, and independence from a separate
  target-time setting of t=4.
- At validation, 4×4/4×5 had 12/12 angles at t=5 for dt=0.05 and 0.25;
  5×5 had 1/12 and 2/12 respectively. These are snapshots of running jobs.
- Notebook schema and preservation checks passed; visually inspected the new
  `/tmp/rough_exact_entropy_delta_theta.png` (PDF also exported).
- Execution evidence: `/tmp/pepsy_examples_executed/rough_exact_entropy_both_validation.ipynb`.
  Full notebook/numerical package suites were not rerun; no package algorithm changed.

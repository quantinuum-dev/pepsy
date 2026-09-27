# 2026-09-25 — continuous 1D qMERA circuit schematic

- Scope: improve the 1D clean qMERA drawing and its TFIM notebook figure after
  user feedback that the stacked schematic was hard to read.
- Branch / baseline commit: `develop` / `80f451a`.
- Commit status: uncommitted and unstaged. Pre-existing Tree edits in Pepsy and
  other sibling-example edits were preserved.

## What changed

- The 1D clean view now draws one left-to-right circuit across selected RG
  scales. Stage headings sit above the wires; gate markers follow the schedule
  order. Disjoint pair gates share a time column only when their drawn spans
  stay separate. Curved, haloed links identify long pairs without filling the
  intervening wires.
- Dashed green windows mark isometry covering blocks. Green and gray input
  markers distinguish retained core and product wires. Explicit-mode wire
  labels show `site:mode`.
- The TFIM notebook now embeds the new figure as a PNG. Explicit image
  embedding is required for its `Agg` notebook execution; displaying the
  Matplotlib figure object produced only `text/plain` under that backend.
- Updated the API page and changelog. The 2D clean view and register view keep
  their existing drawing paths.

## Validation

- `tests/test_optimize_qmera.py`: 103 passed, with two non-failing loky worker
  warnings in the Torch full-graph check. The final schematic-only rerun passed
  6 tests after the explicit-mode label adjustment.
- `python -m ruff check src tests` and `git diff --check`: passed.
- Rendered and inspected four-site TFIM, odd periodic seven-site, and open
  six-site layouts; a three-site explicit-mode smoke render showed distinct
  labels for every mode.
- The TFIM notebook executed into
  `/tmp/pepsy_examples_executed/qmera_tfim_energy_schematic_final.ipynb`.
  Every code cell completed, the figure was embedded as `image/png`, DMRG2
  converged in three sweeps, and JAX/Torch energy and gradient checks passed.
  The inspected image is `/tmp/qmera_tfim_schematic_final.png`.

## Scope limit

- The full Pepsy repository suite was not run; this changed drawing code and
  the complete owning qMERA suite passed.

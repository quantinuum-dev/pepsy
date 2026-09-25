# 2026-09-25 — qMERA schematic clarity

- Scope: make qMERA isometry blocks, boundary disentanglers, executable
  gate rounds, and retained wires visible from the Pepsy schedule.
- Branch / baseline commit: `develop` / `4f62cb0`.
- Commit status: uncommitted and unstaged; earlier qMERA and unrelated Tree
  edits remain in the working tree. The sibling `pepsy_examples/qmera/` folder
  remains untracked.

## Changes

- Auditing `schedule.preparation_order` exposed a direction error in the
  initial clean drawing: retained circuits execute in reverse layer order,
  isometry before boundary disentangler. The clean view now follows actual
  stage order and layer order: coarse → W rounds → D rounds → fine. Site
  schedules retain fine → D → W → coarse.
- The 1D view shows one row per scheduled gate round, green covering W-block
  brackets, green retained coarse wires, gray product wires, and an arc for a
  long periodic seam pair. `style="register"` remains available.
- The 2D view groups retained physical wires by virtual parent register
  (`R0`, `R1`, …), avoiding a misleading flat coarse-site row. Arrows follow
  the schedule direction. Dark links now show each scheduled pair gate inside
  its colored covering window; block labels sit below the window. Curved
  links keep periodic 2D seam gates clear of intervening sites.
- Added the corrected drawing to the TFIM notebook, plus concise API
  documentation and periodic-odd-1D/odd-2D drawing regressions.
- Used Quimb's documented manual
  [schematic.Drawing](https://quimb.readthedocs.io/en/latest/examples/schematic-demo.html)
  primitives; no new package dependency or tensor-network path was introduced.

## Validation

- Focused schematic tests: 6 passed, including exact 2D pair-support and
  periodic-seam checks.
- Full qMERA module: 103 passed (one non-failing loky worker warning).
- Public API/layout: 57 passed, one known installed-version assertion
  deselected (`0.4.1` metadata vs `0.5.0` checkout).
- Ruff (`src tests`) and `git diff --check`: passed.
- Rendered and visually inspected TFIM, odd periodic 1D, odd retained 2D,
  and site-retention 2D diagrams. The corrected previews are
  `/tmp/qmera_tfim_schematic_review_cpu.png` and
  `/tmp/qmera_odd2d_pair_support.png` and
  `/tmp/qmera_periodic_2d_pair_support.png`.
- A GPU-enabled notebook attempt logged CUDA memory exhaustion and was
  interrupted. The documented CPU run completed into
  `/tmp/pepsy_examples_executed/qmera_tfim_energy_schematic_review_cpu.ipynb`
  with an embedded PNG; 25 JAX Adam steps ended at `-5.1873339645`, local
  terms summed correctly, and Torch AOT energy/gradients agreed. The original
  source notebook cell outputs were preserved.

# 2026-09-25 — 2D retained-register qMERA and odd grids

- Scope: implement a Pepsy 2D MERA hierarchy with 2x3, 3x3, or 4x4
  blocks, boundary disentanglers, and reshaped odd-size edge blocks.
- Branch / baseline commit: `develop` / `4f62cb0`.
- Commit status: uncommitted, unstaged, and not pushed. Pre-existing Tree
  edits in this working tree were preserved.

## What changed

- `src/pepsy/optimizers/qmera/schedules.py` now offers opt-in
  `hierarchy="retained"` for unmoded 2D spin systems. It tracks virtual
  coarse-grid cells separately from physical wire indices, retains chosen
  qubits per block, orders x/y boundary rounds, and absorbs singleton axis
  tails. The existing 2D site-retention tiler also absorbs odd tails.
- `builders.py` exposes the hierarchy choice and Pepsy pair ansatz for the
  new spin path. Public schedule/layer metadata exposes grid shapes, cell
  blocks, and retained wires. The API page, changelog, and focused tests were
  updated. The dated design/audit record is
  `docs/development/notes/qmera_2d_retained_hierarchy_2026_09.md`.

## Validation

- `python -m pytest -q -o addopts='' tests/test_optimize_qmera.py`:
  88 passed. New regressions cover odd rectangular/square blocks, periodic
  seams, explicit retention, direct versus local/compiled energies, Torch
  gradients, and JAX JIT gradients.
- Public API/layout suites: 57 passed, 1 deselected. The excluded version
  assertion fails independently because installed distribution metadata does
  not match `pyproject.toml` (checkout 0.5.0).
- Torch `aot_eager` compile matched eager energy and gradients in a separate
  1x5 odd-grid probe, with upstream graph-break warnings.
- `python -m ruff check src tests` and `git diff --check`: passed.
  Full repository suite not run.

## Decisions and limits

- Default 2D behavior remains site retention to preserve current fermion
  and layout consumers. Users opt into the spin hierarchy explicitly.
- The new path uses local two-qubit unitary-completion circuits, brickwall
  isometry structure, and boundary-face disentanglers of width 2 on both axes.
  True rectangular isometry tensors and native graded retained 2D registers
  are outside this change. Large 4x4-block contraction cost remains unmeasured.
- The `apply_patch` helper failed under this device's mountinfo sandbox;
  edits used exact-match Python replacements. No environment packages changed.

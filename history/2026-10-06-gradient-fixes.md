# 2026-10-06 — Recheck and fix gradient optimizers

- Scope: user requested careful confirmation and correction of the reported
  gradient optimizer and integration bugs.
- Branch / baseline: `develop`, `2abb1e8`.
- Commit status: working-tree edits only; nothing staged, committed, or pushed.

## Changes

Implemented the eight confirmed corrections and device-local native Torch
best-state snapshots. See [detailed evidence and limits](../docs/development/notes/2026-10-06-gradient-optimizer-fixes.md)
and the [earlier review](2026-10-06-gradient-review.md).
Updated gradient/FD/qMERA API docs and the changelog. Added deterministic
solver and real PEPS/qMERA integration regressions. Preserved all pre-existing
edits, notebooks, and simulation data.

## Validation

- Solver/regression selection: 173 passed after fixes.
- Final combined domain/API selection: **376 passed, 2 deprecation warnings**
  in 86.12 seconds on CPU. Command (after activating the development environment,
  selecting CPU JAX and limiting BLAS/OpenMP threads):
  `python -m pytest -q -o addopts='' tests/test_gradient_solver_regressions.py tests/test_gradient_solver.py tests/test_gradient_solver_jax.py tests/test_peps_sweep_safeguards.py tests/test_peps_sweep_performance.py tests/test_optimize_qmera.py tests/test_public_api.py tests/test_package_layout.py`.
  Includes dense/compiled Torch qMERA, native Symmray gradients and fermionic
  energy oracles, and native PEPS sweep safeguards.
- `python -m ruff check src tests`: passed.
- `git diff --check`: passed.
- Local links in the changed API/evidence/handoff documents: passed.
- Full-suite attempt stopped after 24 failures, 125 passes, 2 skips.
  An isolated HEAD archive reproduces the exact same 24 BP failure node IDs
  in `test_bp_compression.py` and `test_bp_open_series.py` (42 other passes).
  No claim of full-suite success. No BP changes made in this task.

## Remaining limits

GPU behavior/performance was not measured. Native JAX constraints now fail
explicitly; constrained Optax updates are not implemented. Explicit Torch
bounds still use host clipping. Stored-data recovery is outside this fix.

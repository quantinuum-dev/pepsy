# 2026-09-29 — Preserve unitary stabilization around Kraus branches

- Scope: fix the trajectory option-routing bug identified in the
  [MPS audit](2026-09-29-mps-five-point-audit.md).
- Branch / baseline: `develop`, `b343ced`.
- Commit status: working-tree changes only; nothing staged or committed.
  Existing tree repairs and concurrent cluster edits were preserved.

## Changes

- [Trajectory replay](../src/pepsy/optimizers/noise.py) disables
  `stabilize_unitary` in the copied MPS options for a nonunitary Kraus segment.
  It removes the deprecated `fit_stabilize_unitary` alias from that copy so
  alias resolution cannot re-enable restoration. Caller options and ordinary
  unitary segments retain their requested policy.
- Existing branch probability, norm accounting, normalization, and exponent
  handling remain in their original numerical paths. No tensor algebra,
  backend conversion, dependencies, or upstream dispatch changed.
- [Regression coverage](../tests/test_trajectory_noise.py) uses lossy unitary
  gates before and after amplitude damping at chi=1. It checks final states,
  both Born probabilities, physical-branch norms, local/cumulative compression
  fidelity, cleared exponents, and unchanged caller options. The 16 cases span
  direct/DMRG2, independent/coalesced, optimizer/standalone runner entrypoints,
  and canonical/deprecated stabilization options.
- Updated the [MPS API guide](../docs/api/optimizers/mps.md) and changelog.

## Validation

- Before the fix, all **16 new cases failed** with the conflicting
  `stabilize_unitary=True` / `non_unitary=True` error.
- After the fix, all **16 passed** (3.56 s).
- CPU domain selection: **371 passed**, 3 expected warnings (46.86 s):
  `tests/test_trajectory_noise.py`, `tests/test_mps_normalization.py`,
  `tests/test_mps_fit_performance.py`, `tests/test_mps_dynamic_controls.py`,
  `tests/test_mps_audit_fixes.py`, and `tests/test_optimize_mps.py`.
  This includes the ordinary nonunitary-run rejection test, so the public
  incompatibility guard remains intact outside branch orchestration.
- Repository-wide `python -m ruff check src tests` and `git diff --check` passed.
- Used the existing Python 3.12 environment with CUDA hidden and single-threaded
  BLAS/OpenMP. No fresh full-package or GPU run for this option-only repair.
  Earlier numerical/backend audits are recorded in the linked MPS audit and
  [tree repair handoff](2026-09-29-tree-norm-backend-fixes.md); they are not new
  validation of the concurrent cluster changes.
- Temporary logs: `/tmp/pepsy-kraus-stabilization-before.log`,
  `/tmp/pepsy-kraus-stabilization-focused.log`, and
  `/tmp/pepsy-kraus-stabilization-domain.log`.

## Remaining work

- No remaining issue found in the authorized Kraus-stabilization repair.
  No unrelated audit follow-ups were implemented.

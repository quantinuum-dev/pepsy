# 2026-09-29 — Tree norm and backend fixes

- Scope: fix the two tree failures and stale MPS guidance in the
  [review](2026-09-29-optimizer-commit-review.md), preserving Autoray dispatch
  for NumPy, Torch, JAX, and CuPy as requested.
- Branch / baseline: `develop` / `b343ced`.
- Commit status: working-tree changes only; nothing staged, committed, or
  published. The pre-existing uncommitted review note is unchanged.
  An unrelated `2026-09-29-stop-5x6-dmrg.md` handoff appeared during validation
  and was left untouched.
  Concurrent cluster operator, test, documentation, and changelog edits also
  appeared during the full run. Those edits are outside this repair and were
  preserved; the full-run result does not validate that later workspace state.

## Changes

- Fixed scalar clipping in tree DMRG target norms.
- Kept stabilization and compression ratios separate from represented norm
  readouts, including extreme stored scales and operator exponent changes.
- Fixed CuPy scalar unwrapping in TreeFIT, found by actual GPU validation.
- Added regression coverage and corrected the two internal MPS references to
  the explicit `fit_single_pair_fast_path=True` policy.
- Updated the public replay guide and changelog. The
  [implementation/audit note](../docs/development/notes/2026-09-29-tree-norm-backends.md)
  records backend boundaries and the existing CUDA test-setup correction.

## Validation

Used the existing `py312` environment with one numerical CPU thread in test
subprocesses. CPU runs hide CUDA; separate device checks use the available GPU.
No environment or dependency changes.

- Before the implementation fix, the new initial regressions produced
  **43 failed, 15 passed, 38 skipped**. The original review reproducer also
  reproduced both reported bugs on the baseline.
- Focused CPU validation: **138 passed, 64 skipped** (unavailable devices in
  the CPU-only subprocess). The original review reproducer now passes.
- Tree/trajectory domain selection, MPS FIT budgets, and API/layout checks:
  **1522 passed, 79 skipped** in 177.24 s. Command selected
  `tests/test_optimize_tree.py tests/test_tree_*.py tests/test_trajectory_noise.py
  tests/test_mps_fit_window_budget.py tests/test_public_api.py tests/test_package_layout.py`
  with default smoke filtering disabled and `not slow and not benchmark`.
- Final backend module: **148 passed** in 63.90 s, with Torch CPU/CUDA,
  CuPy CUDA, and JAX CPU (`JAX_PLATFORMS=cpu`, CUDA visible).
- JAX GPU backend module: **32 passed** in 48.04 s with
  `JAX_DEFAULT_MATMUL_PRECISION=highest`. Its default lower GPU matmul
  precision exceeded the tight range-test tolerances; no production precision
  default or tolerance was changed.
- Final stabilization suite, including non-finite exponent rejection:
  **60 passed** in 6.70 s.
- Full-package CPU validation with `python -m pytest -q -ra -o addopts=''`:
  **5579 passed, 149 skipped**, 684 warnings in 1957.82 s (32:37).
  This run includes slow and benchmark tests. It started before the final
  non-finite-exponent guard and its six added regressions; the final
  60-test stabilization run above validates that last change.
  Skips include hidden CUDA devices, unavailable Metal/MPI or optional
  dependencies, and special device-configuration requirements. GPU coverage
  is the separate backend-module validation above, not a full GPU suite.
- Ruff, both changed-skill validators, the skill catalog, changed Markdown
  local links, and `git diff --check` pass. Repository-wide Ruff passed before
  the concurrent cluster changes; final Ruff is scoped to this repair's five
  Python files.

Logs: `/tmp/pepsy-tree-scale-domain.log`,
`/tmp/pepsy-tree-scale-all-backends.log`, `/tmp/pepsy-tree-scale-jax-gpu.log`,
`/tmp/pepsy-tree-scale-stability-final.log`, and `/tmp/pepsy-tree-scale-full.log`.

## Limits

The exterior shortcut's algebra and schedule were not modified. The numerical
trajectory sensitivity reported in the earlier review was not investigated
further in this repair.

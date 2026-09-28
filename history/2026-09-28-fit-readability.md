# 2026-09-28 — Remaining-code audit and FIT readability

- Scope: continue the user-approved readability, comments, and public API
  review; rank remaining problems and improve one domain with existing tests.
- Branch / baseline: `develop` / `f410fa0`, the preceding published batch.
- Commit status: working-tree changes; this pass has not been committed/pushed.

## Changes

- Refreshed the structural inventory of all 212 Python modules and manually
  reviewed the next candidates. Ranked findings and alias decisions are in
  the [review](../docs/development/notes/readability_api_2026_09.md).
- Split FIT public orchestration into named full-chain execution, active-window
  scheduling, and final-polish methods. Reused one block-kernel dispatcher.
- Corrected stopping-policy documentation, added ownership/backend/return
  contracts and an executable example, and documented chain/tree differences.
- Removed dead norm comments and explained target-scale preservation.
- No public signatures, defaults, numerical kernels, dependencies, or tests
  were added/removed. No benchmark directory or documentation builder added.

## Validation

- Existing focused suites: `test_fit_gate_schedules`, `test_fit_hotpaths`,
  `test_mps_fit_kernels`, `test_mps_fit_performance`: **145 passed**, three
  expected finite-check warnings, in **4.63s**.
- All existing FIT signatures and decorators match the baseline. Executable
  ASTs of every original method except the two refactored entry points match.
- Before consolidating dispatch, extracted full-chain branches, gate schedule,
  polish, and fermionic compatibility execution match their original ASTs.
  Shared dispatch preserves each native kernel's arguments and live hooks.
- FIT class example: **6 doctest statements passed**.
- Twelve direct comparisons with the baseline implementation cover MPS/MPO
  reference, cached one-site, block growth, tolerance stopping, gate schedules,
  and final polish: dense outputs and norm traces are exactly equal; return
  values and convergence fields match.
- Default smoke suite: **89 passed**, two compatibility-alias warnings, in
  **19.65s**. The smoke selection and test configuration are unchanged.
- Ruff (`src tests`), the two CI mypy targets, and 12 local Markdown links
  passed. Whitespace check passed.
- Full suite after this refactor:
  `MPLBACKEND=Agg python -m pytest -q -ra -o addopts=''` →
  **5,168 passed, 129 skipped, 788 warnings in 519.83s (8m39s)**, exit 0.
  Skips require unavailable CuPy/accelerator hardware, multiple MPI ranks, or
  two configured XLA host devices. Those configurations remain unvalidated.
- Final diff/whitespace review passed. This pass remains uncommitted, and no
  hosted CI result is claimed.

## Limits

The inventory is package-wide; manual code review is selective. Numerical
speedup is not claimed. Remaining domains are ranked follow-up work, with
their own numerical and ownership invariants, rather than completed rewrites.

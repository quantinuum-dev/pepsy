# 2026-09-30 — MPS DMRG1 aliases DMRG

- Scope: user requested `MpsOptimizer` DMRG1 and DMRG to share one-site FIT
  from the initialized guess by default; synchronize the discussed MPS tutorial.
- Branch / baseline: Pepsy `develop` / `f01f596`; examples `main` / `3b61b7d`.
- Commit status: working-tree edits only; nothing staged, committed or published.
  Existing gates/operator/docs changes, example deletions and prior notebook
  edits were preserved.

## Implemented

- DMRG1 canonicalizes to generic DMRG through construction, `set_mode` and
  `run(mode=...)`. Both accept identical explicit block-size and guess options.
  Default sweeps are one-site from the default SRC guess, up to eight sweeps
  with existing convergence controls. DMRG2/3 schedules are unchanged.
- Removed the old DMRG1 two-site growth reservation, minimum-three-sweep rule,
  rank latch and product-fermion-only two-site exception. Retained the public
  `dmrg1_one_site_locked` result key as `False` for compatibility.
- Updated numerical regressions, API docs, changelog and maintained MPS/FIT
  skills/catalog. The MPO API cross-reference now makes clear that its own
  DMRG1 schedule is separate; other optimizer implementations are unchanged.
- Updated `pepsy_examples/tutorials/dmrg/mps/dmrg_mps.ipynb` explanations and
  schematic labels for the alias. The preceding task removed its study's
  ten-sweep override so it inherits eight. Existing outputs are preserved
  and were not recomputed; they do not measure the new DMRG1 behavior.

See the [audit and implementation evidence](../docs/development/notes/2026-09-30-mps-dmrg1-alias.md)
and [public MPS guide](../docs/api/optimizers/mps.md).

## Validation performed this session

Shared Python 3.12 environment, local Pepsy source first, one CPU numerical
thread, CUDA hidden. No dependency or shared-environment changes.

- The MPS domain (`tests/test_optimize_mps.py tests/test_mps_*.py`), public API,
  package layout and sampler selection comprised 1187 cases. It was completed
  in two batches with two explicit exclusions: **1141 passed, 43 skipped,
  1 existing failure, 2 unfinished/excluded slow references**.
  The first batch passed 581 and skipped 6 before being interrupted during
  `test_mps_optimizer_3x4_pbc_hubbard_long_range_modes_native[exact-U1U1]`.
  That case and `[exact-Z2]` were excluded from the continuation, which
  passed 560, skipped 37 and reported the metadata failure below.
  All DMRG parameters of that slow native Hubbard test completed successfully.
  Logs: `/tmp/pepsy-dmrg1-domain.log`, `/tmp/pepsy-dmrg1-remaining.log`.
- Sole failure: `test_package_version_matches_installed_distribution`, installed
  metadata `0.4.0` versus project `0.5.0`, already recorded in the
  [September 29 handoff](2026-09-29-mps-fit-budget-inheritance.md).
  No environment reinstall or test weakening.
- After retaining the compatibility diagnostic key, reran the final alias
  regression: **12 passed**. These compare numerical states, canonical metadata,
  diagnostics and sweep sequences through constructor/mode-switch/run/copy
  entry points, with default and explicit two-/three-site updates.
  Log: `/tmp/pepsy-dmrg1-final-alias.log`.
- Dense NumPy, CPU Torch/JAX and native Symmray/fermion paths passed in the
  selected suites. Skips cover CUDA/CuPy and unavailable Metal; no accelerator
  validation is claimed.
- `python -m ruff check src tests`: passed.
- Both changed skills passed `quick_validate.py`; the catalog validator passed
  for all 12 skills. No catalog entries or upload files were added/removed.
- Notebook JSON/code AST validation and unchanged-output checks passed.
  Both repositories passed `git diff --check`.

The whole package suite was not run. No performance claim is made, and the
notebook simulation was not rerun.

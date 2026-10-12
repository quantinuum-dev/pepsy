# 2026-10-08 — Full-suite recheck after PEPS review corrections

- Scope: another full check and report after the
  [three fixes](2026-10-08-full-update-review-corrections.md).
- Branch / baseline: `develop`, `ad8ed05`.
- Commit status: PEPS corrections and this report remain uncommitted.
  Nothing staged, committed, or published by this task.
- No implementation changes during this recheck. Concurrent MPS/JAX edits
  were preserved; the PEPS patch is unchanged from the start of validation.

## New validation and findings

- All **8,269 collected cases** were covered across two batches:
  **8,129 passed, 133 failed, 7 skipped**, plus two MPI module collection
  skips because mpi4py is unavailable.
- The first batch was interrupted only after its assigned range finished.
  All 166 duplicate case outcomes agreed; no collected case was missing.
  This is not an uninterrupted passing full-suite run.
- All 133 failures were rerun in fresh processes after GPU contention ended,
  with JAX preallocation disabled: **30 passed, 103 still failed**.
- **All 103 persistent failures reproduce on unchanged ad8ed05**.
  Groups include BP convergence, Torch/Autoray compilation, host conversion,
  native fermionic CTMRG, and JAX numerical comparisons.
- Fresh isolated affected suite: **486 passed**, 98 warnings, no skips.
  Both initially failing CuPy full-update cases also passed fresh reruns.
- A higher-matmul-precision JAX diagnostic passed three representative
  failures. This does not establish a fix for every JAX comparison.
- Ruff, whitespace, skill catalog (12 skills), and report link checks pass.

No additional regression attributable to the PEPS corrections was found.
The broader suite remains failing in this environment. No precision defaults
or tests were weakened; no unrelated fixes were attempted.

The [detailed evidence](../docs/development/notes/2026-10-08-full-update-full-recheck.md)
retains failure groups, all 103 persistent case IDs, skip reasons, batching
limitations, and temporary artifact locations.

# 2026-11-16 — 1D/2D-region DMT API and performance

- Scope: clarify the 1D and 2D DMT APIs, investigate bottlenecks and extend
  performance/correctness validation.
- Branch / baseline commit: `develop`, `b2df2dba`.
- Commit status: working-tree changes only; nothing staged, committed or pushed.
  Preserved the [earlier gradient fix](2026-11-16-dmt-review.md) and unrelated
  sampling/test-harness/changelog/history edits.

## Changes

- Added the [workflow guide](../docs/api/optimizers/pauli_dmt.md) with API
  selection, mutation/return semantics, runnable chain and mapped-lattice
  examples, and accuracy/performance controls. Linked the nearest API,
  optimizer, module and example indexes; retained existing low-level anchors.
- Clarified that 2D region protection uses an open Pauli MPO and does not
  change the adjacent-chain evolution schedule. Documented exact nonadjacent
  gate embedding followed by protected compression.
- Replaced general gauge-move einsums with native reshape/matrix products.
  Regional checks now use maximum-entry intermediate scaling, retaining
  Frobenius-normalized final panels and every before/after preservation check.
- Added [CPU profiling examples](../examples/pauli_dmt_benchmark.py), with
  independent compression marginal checks outside timing. Added 36 numerical
  regressions for extreme virtual/global scales across NumPy/Torch/JAX and
  all four supported dtypes.
- [Measured evidence](../docs/development/notes/2026-11-16-dmt-performance.md):
  alternating same-process comparisons gave 1.53–1.63× compression speedups,
  essentially unchanged small 1D evolution, and 1.08–1.09× lattice-region
  evolution. Main remaining costs are protected splits/SVDs and region panels.

## Validation

- Complete five-file DMT selection, including slow tests, CPU BLAS limited to
  one thread: **225 passed, one existing expected failure**, seven existing
  rank-deficient Torch QR warnings, 90.12 seconds. Includes larger chain and
  5×5/6×6/8×8 lattice probes. The expected failure is unresolved float32
  nonzero-trace classification with `trace_guard="auto"`.
- New 36-case scaling selection passed separately; both workflow-guide
  examples executed successfully.
- All four benchmark cases ran with float64 and complex128, five repetitions
  each; independent compression region errors stayed below `2.4e-14` and
  all trace errors below `6.1e-14`. Additional baseline comparisons used
  seven alternating repetitions per version.
- Public API/package layout: **54 passed, one known metadata failure**,
  `test_package_version_matches_installed_distribution`: installed version
  `0.5.0` vs project `0.6.0`. The environment was not modified.
- Ruff (`src`, `tests`, benchmark example) and `git diff --check` passed.
- Local Markdown links and anchors were checked for the edited guides,
  indexes and notes.

## Limits and remaining bottlenecks

- CPU results only; full repository suite and GPU checks were not run.
- Contraction summation order can change subsequent nonlinear finite-bond
  trajectories. Two-step benchmark marginal differences between versions
  reached `2.38e-6`, although all per-gate constraints pass. Separate dense
  circuit probes showed comparable global compression errors; see the note.
- Region verification retains its `4**k` output and MPO-span cost. No check
  was disabled and no approximate SVD or protected-rank policy was adopted.
- Automatic 2D nonadjacent-gate scheduling and native Symmray DMT were not
  implemented. Existing finite-precision and conservative-rank limits remain.

# 2026-09-24 — Costed two-value ZZ phase replay

- Scope: improve `MpsOptimizer(mode="exact-batch")` on CPU and GPU without
  increasing full-state memory use.
- Branch / baseline commit: `develop`, `6d3b6dc`.
- Commit status: committed in this local change; not pushed.

## What changed

- Long consecutive ZZ-like diagonal runs with two distinct exact value pairs
  can use one grouped population-count phase pass. The CPU and CuPy kernels
  produce one output array; other gates and unsupported layouts retain the
  existing bounded blocks.
- A state-size and avoided-pass check keeps small or short GPU/CPU workloads
  on the faster existing route. The [exact-batch audit](../docs/development/notes/mps_exact_batch.md#two-value-zz-layers-2026-09-24)
  records the algorithm, thresholds, local timings, and upstream review.
- Updated the [MPS API guide](../docs/api/optimizers/mps.md), implementation
  map, changelog, and MPS skill. Preserved the separate uncommitted tree-layout
  edits outside this commit.

## Validation

- Focused exact-batch suite: 24 passed, including NumPy/CuPy comparison with
  reference exact replay, nonunitary scale, reversed endpoints, input ownership,
  small-state fallback, and three-class fallback.
- Adjacent MPS/control/Quimb/API/package selection: 130 passed, one inherited
  JAX `complex64` direct-mode Kraus-probability precision failure. That failure
  was reproduced on the remote baseline in the preceding handoff.
- A separate CuPy `complex128` probe agreed with four ordinary diagonal
  blocks to maximum absolute error `9.43e-16` on 22 qubits.
- Ruff, skill quick validator, skill catalog validator, and whitespace checks
  passed. The full suite and 30-qubit GPU benchmark were not run.

## Remaining opportunities

Costed commuting RXX/RYY basis rotations, immutable-stream plan reuse,
copy-on-write in-place application, and cuStateVec integration still require
separate performance and ownership evidence. The prior
[handoff](2026-09-24-mps-exact-batch-rebase.md) records baseline JAX failures.

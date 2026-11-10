# 2026-11-09 — PauliMPO review and DAOE filter

- Scope: review `PauliMPO`, fix relevant defects and start DAOE implementation.
- Branch / baseline commit: `develop`, `ad523b6`.
- Commit status: working-tree edits only; nothing staged, committed or pushed.
  Existing sampling edits and prior untracked journals were preserved.

## What changed

- Added exact `PauliMPO.apply_daoe(max_weight=..., gamma=..., inplace=False)`
  for sparse strings and native dense counting cores. Compression stays explicit.
- Fixed native subtraction/commutators, real-core dtype conversion,
  dtype/device-aware constants, dense readout gradients, channel in-place
  materialization, stale gate/channel reports and noninteger gate supports.
- Corrected partial-SVD loss reports to include the uncomputed spectrum.
- Added [API usage](../docs/api/operators/pauli_mpo.md#daoe-weight-filter),
  [review/upstream evidence](../docs/development/notes/2026-11-09-pauli-mpo-daoe.md),
  numerical regressions, operator ownership navigation and changelog notes.
  The real-core conversion failure observed in the earlier read-only DMT
  probe is fixed here. Lattice-DMT implementation remains separate work.

## Validation

- Focused Pauli algebra/backend/DAOE tests: **78 passed**; includes explicit
  coefficients, 64-site no-expansion check, Torch/JAX gradients and a dense
  Heisenberg/filter/compress reference.
- Public API/package tests: **54 passed, 1 failed** from pre-existing installed
  Pepsy 0.5.0 metadata versus the checkout's 0.6.0 declaration.
- Broader MPO/automaton checks: **133 passed, 17 failed, 1 skipped**, with the
  identical 17 failures reproduced using baseline Pauli implementation loaded
  in memory. Failure names are retained in the linked review note.
- Ruff, `git diff --check` and local Markdown file-target checks pass.
  Documented ten-period example executes. Full suite, GPU/CuPy and native
  Symmray DAOE checks were not run.

## Remaining work

This is the exact artificial filter, not a complete DAOE integrator or a
transport reproduction. A local Heisenberg gate-split implementation and
explicit time/filter/compression schedule would avoid current global-product
bond inflation. Lattice-DMT protected compression remains separate work.
Do not infer physical accuracy from a zero SVD loss report; finite `gamma`
already modifies the dynamics.

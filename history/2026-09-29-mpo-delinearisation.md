# 2026-09-29 — Delinearisation and frontier notebook comparison

- Scope: implement MPO delinearisation without SVD, explain its optimality
  limits, and add frontier comparison to both 1D notebooks.
- Baselines: Pepsy develop c1ff0f8; Gaugy Examples main dbabafa.
- Status: uncommitted working-tree edits; nothing staged or published.
  Pre-existing unrelated changes in all repositories were preserved.

## Changes

Added public `delinearize_mpo` and `MPODelinearizationReport` under
`pepsy.operators`, with focused reconstruction/guard tests. Added short
commented frontier and delinearisation examples, full-matrix comparisons,
and four-method bond plots in both `cluster_1d` and `cluster_1d_joint`.
The notebooks retain the NN + NNN model, one boundary per run, cluster
orders, independent references, and single/joint factor conventions.

Updated owning API docs, API index, module map, changelog, current cluster
status ledger and examples README. See
[implementation, measured dimensions, failure correction and checks](../docs/development/notes/2026-09-29-mpo-delinearisation.md).

## Validation

- New routine + public API/layout: 81 passed.
- Existing channel and MPO compression selection: 81 passed, 6 CUDA cases skipped.
- Gaugy channel/binding/materialization selection: 27 passed.
- All 16 independent six-site OBC/PBC single/joint p=2..5 probes passed after
  fixing column-selection stability; no acceptance tolerance was relaxed.
- Both notebooks executed completely under OBC and separately under PBC,
  including all matrix, scaling, rank, single-call and Pauli checks. Maximum
  delinearisation error across these comparisons: 6.50e-14. Saved fresh OBC
  outputs in 25 single and 26 joint code cells; PBC runs remain in /tmp.
- Ruff on source/tests/notebooks, notebook schemas, strict KaTeX (62 single /
  56 joint expressions), scoped source review, local links and whitespace
  passed. Visually inspected the four-method bond and accuracy plots.
- No full repository suite was run. Existing channel regressions include
  native/Torch/JAX paths; delinearisation itself explicitly rejects those
  arrays and has separate guard tests. Six CUDA tests were skipped in the
  CPU-only run.

## Limitations

Delinearisation is numerical QR-based dependency removal for dense NumPy
open MPOs, not symbolic parameter-family minimization or an autodiff path.
It does not guarantee SVD-optimal or minimum MPO ranks. Frontier avoids the
original expanded fixed MPO, but its own tensors and preparation maps can
still be large. No full repository suite or larger-system validation.

# 2026-09-29 — Native spin cluster autodiff and assembly

- Scope: finish the remaining Pauli/spin workflow; user explicitly excluded
  native fermionic cluster construction.
- Branch / baseline: `develop`, `c1ff0f8`.
- Commit status: local working-tree edits, not staged, committed or pushed.
  Earlier cluster work and unrelated journals are preserved.

## What changed

Pepsy owns structural native charge labels, exact Torch/JAX materialization,
sector-preserving recursive/streaming assembly and public NumPy/Torch native
adaptive compression. Truncated Torch derivatives use existing paired-factor
projectors; public fixed-rank compression now rejects an unspecified sector
allocation. See the [evidence and limits](../docs/development/notes/2026-09-29-native-cluster-autodiff.md)
and [API](../docs/api/operators/mpo_cluster.md#native-block-sparse-cluster-mpos).
Gaugy runtime code is unchanged and continues to use public Pepsy APIs.

## Validation

The broad affected Pepsy selection passed **555 tests**. Gaugy's existing
joint/materialization/binding/package selection passed **137 tests**. These
are selected suites, not full-repository results. Final public compression
guards and custom-index preservation were added after the broad selection.
The final recheck of `test_native_cluster_autodiff.py`,
`test_native_cluster_mpo.py`, `test_mpo_cluster_compression.py` and
`test_mpo_cluster_recursive.py` passed **92 tests** in 111.08 seconds, including
all new native regressions. Pepsy source/test Ruff and whitespace checks pass.

## Boundaries

Exact fixed construction supports zero-coefficient derivatives and JAX JIT.
Numerical native compression supports NumPy/Torch, with discrete sector/rank
selection and first-order local-chart derivatives. Native JAX compression,
rank-change derivatives and large-lattice GPU measurements are not supplied.
Fermionic histories are outside scope. No publishing action was requested
for Pepsy. Gaugy's documentation records the dependency's local-only status.

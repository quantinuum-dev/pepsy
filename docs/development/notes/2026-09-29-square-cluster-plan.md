# Shared square-plan adapter: compatibility evidence, 2026-09-29

The shared-plan factory now selects the existing square builder when its
geometry and term contracts match. See the [API guide](../../api/operators/interaction_clusters.md)
and [validation record](../../../history/2026-09-29-square-cluster-plan.md).

The upstream review from the same active cluster task is recorded in the
[shared-planner journal](../../../history/2026-09-29-interaction-cluster-plan.md).
Its official-source audit is reused; installed versions were rechecked and
are unchanged: Quimb `1.15.1.dev66+ge927f06e1`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.4.1.dev7+g83fb22865`, Torch `2.6.0+cu124`, and JAX `0.10.2`.
The earlier unavailable Symmray documentation page is not treated as read.

Installed public signatures checked for this adapter:

- `quimb.tensor.PEPO(arrays, *, shape='urdlbk', ..., cyclic=None, **tn_opts)`
- `PEPO.to_dense(self, *inds_seq, to_qarray=False, **contract_opts)`

**Adopt:** existing Pepsy active square blocks and fixed-channel construction,
with Quimb's public PEPO constructor and dense contraction. Keep square
virtual directions, physical ordering, and explicit boundary metadata.
No new contraction driver, dependency upgrade, or vendored implementation.

**Compatibility shim:** Gaugy checks for Pepsy's public square-adapter export.
Earlier shared-plan Pepsy installations retain graph output for auto/graph
calls; explicit square options require the new capability. A downstream
regression exercises this branch.

**Defer:** square routing of diagonal/NNN edges, general local matrices,
fermionic/string/charge metadata, and native Symmray square conversion.
These inputs are never silently interpreted as supported Pauli NN terms.
No native Symmray validation or broad performance improvement is claimed.

NumPy dense/reference checks, Torch repeated materialization gradients and
JAX JIT materialization gradients are validated separately in
`tests/test_square_cluster_plan.py`. On the executable 2x2 cutoff-three
example, square and graph matrices differ by `1.44e-15`; the fixed square
PEPO has maximum compact bond dimension 13. This is one correctness example,
not a cold/warm timing, memory, or bond-dimension benchmark.

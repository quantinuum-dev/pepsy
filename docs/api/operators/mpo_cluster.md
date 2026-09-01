# MPO cluster expansion and ordered products

This page documents `pepsy.operators.mpo_product`, the connected spatial MPO
family. It is separate from the SciPost higher-order/history construction in
[`higher_order_mpo.md`](higher_order_mpo.md): `cluster_size` counts local
connected support, while higher-order `order` controls virtual history/Taylor
construction.

## Term-centric facade

`exp_mpo_cluster(...)` is the one-shot facade for the cluster family. It uses
the same term spellings, `shape`/`mapper` handling, coefficient parameters,
`dt` compatibility keyword, backend conversion, semantic return, reports, and
final `chi` compression controls as `exp_mpo`, while adding explicit
`cluster_size` and `graph` arguments:

```python
from pepsy.operators import exp_mpo_cluster

U = exp_mpo_cluster(
    [{"operator": "ZZ", "location": ((0, 0), (1, 0)), "coefficient": 0.7}],
    -1j * 0.01,
    shape=(4, 4),
    cluster_size=3,
    graph="square",
    cyclic=True,
)
```

Without `graph`, the facade selects connected chain intervals. Use
`graph="chain"` for an explicit chain graph, or `graph="square"` with a
two-dimensional `shape` for the common square geometry. `cyclic=True` makes a
chain a ring or a square periodic in both directions; for a square,
`cyclic=(True, False)` selects only the first direction. Explicit
`ClusterLattice` objects remain available for arbitrary graphs and already
encode their own periodic edges. In all graph cases, the graph is mapped
through the lattice-to-chain ordering and `cluster_size` counts graph sites.
`return_semantic=True` returns the cluster-produced `FirstDegreeMPO`; the
default returns a Quimb MPO. `max_bond` caps each analytical residual
factorization, while `chi` is an optional final numerical MPO compression.
`to_backend` is applied to local operators, the step, resolved coefficients,
intermediate tensor contractions, and the final Quimb tensor boundary.

The history-only keywords `order`, `mode`, `history_storage`, and
`extension_budget` are deliberately not accepted here. They have no safe
translation to spatial cluster size and remain part of `exp_mpo` only.

## Single-factor cluster expansion

```python
import numpy as np

from pepsy.operators import MPOClusterProductExpansion

z = np.diag([1.0, -1.0])
expansion = MPOClusterProductExpansion.from_local_terms(
    32,
    [((site, site + 1), (z, z)) for site in range(31)],
    cluster_size=4,
)
U = expansion.exp(-1j * 0.1)
```

Each connected interval is exponentiated locally, lower connected residuals
are subtracted, and the residuals are assembled as disjoint MPO paths.
`cluster_size=L` is exact for a finite chain; smaller cutoffs give the
size-extensive approximation described in
[arXiv:1912.10512](https://arxiv.org/abs/1912.10512).

Use `MPOGraphClusterProductExpansion` or
`MPOBasis.compile_graph_cluster_expansion(...)` when cluster selection should
follow a graph rather than a snake-ordered interval. A graph cluster can be a
genuine two-site cluster even when its MPO span crosses many chain positions.
The report records `cluster_mode="graph"`, graph counts, loop counts, and
local residual ranks.

## Joint ordered products

For factors `A`, `B`, and `C`, use
`MPOClusterProductExpansion.from_factors(...)`:

```python
from pepsy.operators import MPOClusterFactor, MPOClusterProductExpansion

A = MPOClusterFactor([((0, 1), (z, z))], coefficient=0.2)
B = MPOClusterFactor([((1, 2), (z, z))], coefficient=-0.3)
C = MPOClusterFactor([((2, 3), (z, z))], coefficient=0.4)

product = MPOClusterProductExpansion.from_factors(
    8,
    (A, B, C),
    cluster_size=4,
)
U = product.compile_exp().exp(0.01)
```

The factor order is algebraic order. On each connected local support `S`, the
construction forms the joint target
`exp(A_S) @ exp(B_S) @ exp(C_S)`, subtracts lower connected partitions, and
inserts the residual into one MPO topology. It does not build three
independently truncated full-lattice MPOs and multiply them. The report's
`factor_count` records how many ordered factors participated in that joint
expansion.

This is different from `exp(A + B + C)`, except in cases where the operators
commute in the relevant algebra. It is also different from multiplying
already-materialized MPO layers: `compose`/Quimb multiplication is an
execution-level operation, not the cluster-residual algorithm.

## Repeated evaluations

`compile_exp()` caches interval/factor topology and rebuilds backend tensors on
each evaluation, so Torch/JAX autodiff graphs stay current. `max_bond` is an
explicit local Schmidt-rank cap; it is separate from the history guard used by
the higher-order MPO family.

`MPOClusterBasisExpansion` and `CompiledMPOClusterExp` remain compatibility
aliases for `MPOClusterProductExpansion` and `CompiledMPOClusterProduct`.

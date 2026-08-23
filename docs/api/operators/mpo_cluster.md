# MPO cluster expansion

This page documents `pepsy.operators.mpo_cluster`, the connected spatial MPO
family. It is separate from the SciPost higher-order/history construction in
[`higher_order_mpo.md`](higher_order_mpo.md): `cluster_size` counts local
connected support, while higher-order `order` controls virtual history/Taylor
construction.

## Single-factor cluster expansion

```python
import numpy as np

from pepsy.operators import MPOClusterBasisExpansion

z = np.diag([1.0, -1.0])
expansion = MPOClusterBasisExpansion.from_local_terms(
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

Use `MPOGraphClusterBasisExpansion` or
`MPOBasis.compile_graph_cluster_expansion(...)` when cluster selection should
follow a graph rather than a snake-ordered interval. A graph cluster can be a
genuine two-site cluster even when its MPO span crosses many chain positions.
The report records `cluster_mode="graph"`, graph counts, loop counts, and
local residual ranks.

## Joint ordered products

For factors `A`, `B`, and `C`, use `MPOClusterBasisExpansion.from_factors(...)`:

```python
from pepsy.operators import MPOClusterBasisExpansion, MPOClusterFactor

A = MPOClusterFactor([((0, 1), (z, z))], coefficient=0.2)
B = MPOClusterFactor([((1, 2), (z, z))], coefficient=-0.3)
C = MPOClusterFactor([((2, 3), (z, z))], coefficient=0.4)

product = MPOClusterBasisExpansion.from_factors(
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

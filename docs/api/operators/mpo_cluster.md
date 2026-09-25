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
    graph="auto",
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

The term parser recognizes the location convention from the term itself:

```python
# 1D chain terms: integer sites are already chain positions.
chain_terms = [(("ZZ", J), (i, i + 1)) for i in range(L - 1)]
chain_terms += [(("X", h), i) for i in range(L)]
U_chain = exp_mpo_cluster(chain_terms, -1j * dt, shape=L)

# 2D lattice terms: coordinates are mapped internally. ``mapper`` is
# optional; ``map_mode`` selects the default OneDMap traversal.
lattice_terms = [(("ZZ", J), edge) for edge in edges]
lattice_terms += [(("X", h), site) for site in sites]
U_lattice = exp_mpo_cluster(
    lattice_terms,
    -1j * dt,
    shape=(Lx, Ly),
    graph="square",
    map_mode="snake",
    cyclic=True,
)
```

Pass `mapper=OneDMap(Lx, Ly, mode=...)` when a custom ordering is required;
it is not needed for the default mapping. A single 1D site is written as a
bare integer, while `(x, y)` denotes one 2D coordinate when the term has one
local operator. One call should use one convention consistently rather than
mixing chain indices and coordinates.

For the common geometries, `graph="auto"` selects a chain graph for integer
locations and a square graph for 2D coordinate locations. It preserves the
meaning of `cyclic`: a chain becomes a ring, while a square lattice becomes
periodic in both directions. Use an explicit graph for custom or 3D topology.

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

### Graph assembly safety

Crossing or nested graph clusters require products of disjoint residual paths
when they are represented as an MPO. The default
`graph_assembly="auto"` first counts compatible collections with a
cutwidth-aware chain-frontier dynamic program. Only plans whose count is within
`collection_budget=128` are explicitly materialized. If the planner detects a
larger plan (or exceeds its bounded frontier-state work budget), the call
emits a `RuntimeWarning` and uses the bounded one-cluster approximation, which
retains each individual graph residual but omits products of multiple graph
residuals.

Use the explicit controls when the tradeoff matters:

```python
# Fast, controlled graph-MPO approximation.
U = exp_mpo_cluster(
    terms,
    -1j * 0.01,
    shape=(4, 4),
    graph="square",
    cluster_size=2,
    graph_assembly="bounded",
    max_collection_order=1,
)

# Require the full collection expansion, but fail before unsafe allocation.
U = exp_mpo_cluster(
    terms,
    -1j * 0.01,
    shape=(2, 2),
    graph="square",
    cluster_size=2,
    graph_assembly="exact",
    collection_budget=1000,
)
```

`max_collection_order` counts non-single graph residuals in one collection;
it is separate from `cluster_size`. `graph_assembly="exact"` preserves the
previous full result, but raises if `collection_budget` is exceeded. Set
`collection_budget=None` only for an intentionally unbounded small-graph
calculation. The report exposes the selected assembly mode, collection count,
frontier width in the MPO ordering, and whether the collection expansion was
truncated.

### Streaming graph-path assembly

The default direct materialization keeps every selected graph residual path in
one semantic MPO and is therefore unsuitable for wide two-dimensional MPO
orderings at larger cluster sizes. Use the opt-in streaming boundary to add
paths in bounded batches and apply a fixed-rank semantic TT-SVD after each
batch:

```python
U = exp_mpo_cluster(
    terms,
    -1j * 0.01,
    shape=(4, 4),
    graph="square",
    cyclic=True,
    cluster_size=4,
    assembly="streaming",
    assembly_chi=64,
    assembly_batch_size="auto",
)
```

`assembly_chi` is a working MPO bond cap and is independent of the optional
final `chi` compression. `assembly_batch_size="auto"` selects up to 32 paths
per batch and reduces that size for very wide graph frontiers;
`assembly_batch_size=1` gives path-at-a-time accumulation. Larger batches
reduce the number of SVD sweeps while keeping the intermediate direct sum
bounded. Streaming builds local path cores and
inserts the batch directly into the accumulator's virtual direct sum; it does
not construct a temporary full-chain MPO for each path or batch. It then
applies a semantic fixed-rank TT-SVD after each batch. Streaming therefore
introduces numerical truncation after each batch, so its result can depend on
the path order. The streaming accumulator contains the singleton rail plus an
additive sum of the selected paths. It therefore does not create products of
separate paths when the graph plan is the direct or bounded one-cluster plan;
those products are a different graph-collection approximation axis. If the
planner selected an exact or higher-order bounded collection plan, streaming
batches those collection paths as well. The returned report records the
planner, frontier-state work, number of streaming compressions, peak
pre-compression bond dimensions, the requested and resolved batch sizes, and
final working bond dimensions.

For a cutoff-aware working boundary, add `assembly_cutoff`:

```python
U = exp_mpo_cluster(
    terms,
    -1j * 0.01,
    shape=(4, 4),
    graph="square",
    cyclic=True,
    cluster_size=4,
    assembly="streaming",
    assembly_chi=64,
    assembly_cutoff="auto",
    assembly_cutoff_mode="rsum2",
    assembly_form="left",
)
```

`assembly_cutoff=None` is the backend-differentiable fixed-rank policy.
Supplying a cutoff selects singular-value-dependent ranks. Tensor arithmetic
stays on the requested backend, while the discrete rank decision is not
suitable for a compiled/JIT trace. The report records the sweep direction
and discarded singular-value weights. `assembly_form="right"` is available
for a right-to-left semantic TT-SVD.

### Native block-sparse cluster MPOs

Direct cluster assembly can retain virtual operator-valued blocks and compile
them to native Symmray sectors:

```python
U = exp_mpo_cluster(
    terms,
    -1j * 0.01,
    shape=(4, 4),
    graph="square",
    cyclic=True,
    cluster_size=2,
    symmetry="U1",
    physical_charges=(0, 1),
)
```

The supported bosonic symmetries are `U1`, `Z2`, `U1U1`, and `Z2Z2`. Use
`MPOPhysicalSpace` when the physical metadata is already bundled. Native
sector compilation currently requires NumPy local blocks, and streaming
intermediate compression is intentionally rejected for symmetric clusters
until its SVD is sector-aware. Fermionic graded cluster histories remain a
separate unsupported path.

For a two-dimensional lattice at scale, prefer the graph-native PEPO active
representation. An MPO must pay for the lattice-to-chain cutwidth, while a
PEPO keeps one virtual bond per graph edge.

## Joint ordered products

For a one-shot ordered product, use the product-named facade
`exp_mpo_cluster_product(factors, step, ...)`. It has the same graph,
periodic-boundary, streaming, backend, report, and final-compression controls
as `exp_mpo_cluster`, while making the required factor list explicit:

```python
from pepsy.operators import exp_mpo_cluster_product

U, report = exp_mpo_cluster_product(
    (terms_A, terms_B, terms_C),
    -1j * dt,
    shape=(Lx, Ly),
    graph="square",
    cyclic=True,
    cluster_size=3,
    assembly="streaming",
    assembly_chi=64,
    return_report=True,
)
```

Each factor can be a term iterable, an `MPOClusterFactor`, an `MPOBasis`, or
a mapping with `terms` and an optional factor `coefficient`. For repeated
evaluations, use `MPOClusterProductExpansion.from_factors(...)`:

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

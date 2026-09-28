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

## Exact construction for autodiff without SVD

Set `factorization="fixed"` to construct cluster residual MPOs with fixed index
routing instead of a numerical operator-Schmidt basis. Both interval and graph
constructors, `MPOBasis` cluster/compile helpers, and the exponential/product
facades accept this policy. The existing `factorization="auto"` default keeps
the numerical policy, including SVD and cutoff/rank controls.

```python
compiled = basis.compile_graph_cluster_expansion(
    graph="square", cluster_size=4,
    spatial_reuse=True,
    factorization="fixed", cutoff=0.0, max_bond=None,
    graph_assembly="exact", assembly="recursive",
    assembly_chi=None, assembly_state_budget=40_000,
)
# Parameter values and the step can be backend tensors with gradients.
U, report = compiled(step, parameters=parameters, return_report=True)
```

The exact factorization uses `M = I @ M` or `M = M @ I` according to the
static matrix dimensions, preserving gradients even when M is zero. Channel
shapes do not depend on coefficients or the time step. `cutoff=0.0` (or None)
and `max_bond=None` are required; assembly rank/cutoff controls, a final facade
`chi`, and native charge-sector compilation are rejected in fixed mode so
construction cannot silently reintroduce decompositions. Compress the returned
operator separately if a numerical approximation is wanted. Fermionic cluster
construction remains unsupported.

`report.factorization` and `compiled.cache_info["factorization"]` identify the
policy. Geometry, supports, structural symmetry matches, coefficient bindings,
and graph collection schedules are reusable; matrices and gradients are fresh
on each call. Keep static operators on NumPy and pass trainable coefficient
values through `MPOParameter` bindings to enable conservative symmetry reuse.
Fixed mode aligns evaluated local targets to their tensor backend before
residual subtraction. Python/NumPy steps work with named or positional tensor
parameters, tensor-valued `MPOParameter` defaults, and callable coefficients,
including empty singleton supports in bond-only models. This alignment adds
no calls to user coefficient functions. Parameter defaults are used when
`parameters=None`; required parameters still raise when absent. Tensor inputs
must use one backend, or an explicit `to_backend` converter. One-shot fixed facades
also avoid SVD while parsing general dense local operators. A pre-existing
`MPOBasis` may already have used SVD during its Hamiltonian initialization;
its fixed cluster evaluations use no SVD.

### Explicit bond compression after fixed construction

Keep the compiled fixed-channel operator for autodiff. When a smaller numerical
MPO is acceptable, compress the completed result as a separate operation:

```python
exact_cluster = compiled(step, parameters=parameters)
compressed, compression_report = exact_cluster.compress_numerical(
    max_bond=16, cutoff=0.0, estimate_error=True, return_report=True,
)
relative_error = compression_report.operator_frobenius_relative_error
```

`relative_error` compares the compressed MPO with the **uncompressed operator
at the same cluster order p**. It does not measure the spatial cluster-order
error against the global exponential. For dense MPOs the optional estimate
canonicalizes the difference before contracting its Frobenius norm; it does
not form a global dense matrix. Error estimation can cost more than the
compression itself. The numerical compression uses Quimb's rank reduction,
including SVD, and may alter gradients. Keep `exact_cluster` for a gradient
calculation that requires SVD-free construction. Post-compression starts only
after the full fixed-channel MPO exists, so it cannot cap peak assembly memory.
For large graphs that require a working bond cap during assembly, use the
separate numerical `assembly_chi` policy and check convergence in that cap.

Run `python examples/cluster_mpo_bond_compression.py` from the repository root
for a reproducible 2x3 square-lattice chi/error sweep. It reports construction
time, initial/final MPO bonds, compression time, and the relative error against
the selected p-cluster reference. Its timings exclude global dense operators.

For qubits, a generic exact four-site residual has local MPO ranks at most
`(4, 16, 4)`. This does not bound the assembled lattice MPO: sums of many
clusters can still have large bonds. SVD-free construction removes the
singular-vector derivative and rank-selection issues; it does not provide
small global bonds for arbitrary exact operators. Inspect collection/state
budgets and memory before constructing wide-lattice exact MPOs.

A scalar array-valued loss wrapping complete fixed MPO construction has
passed JAX `jit(value_and_grad)` on a three-site order-three chain, a
2×2 square at order four, and a two-site complex-time case, including zero
parameters and zero time. Return
an array or scalar from the JIT function: the Python MPO/report objects are
not JAX outputs. This is finite-case correctness evidence, not a general
machine-code speedup or arbitrary-geometry guarantee.

Torch `torch.compile(backend="aot_eager", fullgraph=True)` captures the local
matrix exponential, identity construction, fixed split and backward pass,
including zero parameters and zero time. Eager Torch autograd also covers the
complete MPO. Full-graph capture of the complete Python MPO builder and Quimb
materialization still fails in the installed Autoray/Torch stack; compile the
numeric kernels rather than assuming `compiled(...)` itself is a Torch graph.
The `aot_eager` backend checks capture and autodiff, not optimized runtime.

### Declared spatial symmetries

In addition to automatic term-aware reuse, `spatial_symmetries` accepts an
iterable of finite-lattice site permutations. For MPOs, sites are **chain
indices after `OneDMap`**. Each permutation is a mapping from every source
index to its target index, or the target-index sequence in source order:

```python
# Reflection of a three-site chain; every site/term must obey this symmetry.
compiled = basis.compile_cluster_expansion(
    cluster_size=3, factorization="fixed", cutoff=0.0,
    spatial_symmetries=[(2, 1, 0)],
)
```

Declarations are normalized into the cache key and checked against graph edge
multiplicity, ordered Hamiltonian factors, operator labels and coefficient
identities. Invalid declarations or opaque bindings that cannot be proved
raise. Automatic local matching remains the numerical reuse authority: a
supplied permutation never bypasses its checks or requests a physical spin
rotation. `declared_spatial_symmetry_count` reports verified declarations.
`spatial_reuse=False` cannot be combined with declarations.

Finite global symmetries differ from local cluster equivalence. An open chain
has no wraparound translation symmetry, while translated interior clusters
can still share local targets automatically. Likewise, rectangular boundaries
can break a global quarter-turn while many rotated local clusters remain
identical. No global symmetry declaration is necessary for that local reuse.

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

## Reusing lattice and Hamiltonian symmetries

Cluster MPO constructors, the `MPOBasis` cluster/compile helpers, and both
one-shot cluster facades accept `spatial_reuse=True` (default). Compilation
compares local ordered Hamiltonians under site permutations. Identical
translated clusters, allowed rotations/reflections, and other graph
relabelings share their local exponential product. This also works after a
2D lattice is mapped through `OneDMap`: the supplied graph determines the
clusters, while the snake determines their MPO ordering.

Reuse requires agreement of graph edges, operator labels/matrices, coefficient
references and factor order. Spatial relabeling does not rotate spin labels
(`X` stays `X`). Distinct named parameters remain independent even if their
current values agree. The current MPO matcher supports NumPy product-term
operators with immutable scalar or `MPOParameter` coefficients; opaque
coefficients, backend-connected operators, string operators and unfactorized
dense local terms use independent evaluation. No numerical tolerance decides
whether two Hamiltonians are equivalent.

```python
compiled = basis.compile_graph_cluster_expansion(
    graph="square", cluster_size=4, spatial_reuse=True,
)
print(compiled.cache_info["spatial_plan"])
# Keys: targets, representatives, reused, search_fallbacks.
# For uniform X + ZZ on a 5x6 open square: 492, 6, 486, 0.
```

The plan persists across calls; local matrices and their gradients are fresh
for every evaluation. Set `spatial_reuse=False` to compare with independent
local evaluations. The permutation search is bounded at 4096 candidates per
cluster; larger searches fall back to identity-only matching, which can still
reuse translations. `search_fallbacks` reports that conservative fallback.

These counts concern local targets. Every finite cluster placement remains
in residual subtraction and assembly. Geometric shape inventories, cluster
cutoff, graph collection budgets and compression controls retain their existing
meaning. In particular, symmetry reuse does not remove the graph assembly
limitation described below.

### Direct graph assembly safety

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

### Complete recursive assembly: add MPO branches and compress

Use `assembly="recursive"` to keep every compatible product of disjoint
cluster residuals without listing all collections. It builds shared MPO
subproblems, adds their contributions in batches, and compresses each sum:

```python
compiled = basis.compile_graph_cluster_expansion(
    graph="square",
    cluster_size=4,
    graph_assembly="exact",
    assembly="recursive",
    assembly_chi=64,
    assembly_batch_size=1,
    assembly_state_budget=40_000,
)
U, report = compiled(-1j * dt, parameters=parameters, return_report=True)
```

For the remaining sites $S$, choose the first site $i$ in MPO order and use

$$
F(S) = E_i F(S\setminus\{i\})
     + \sum_{C\ni i,\ C\subseteq S,\ 2\leq |C|\leq p}
       R_C F(S\setminus C),\qquad F(\varnothing)=I.
$$

Here $E_i$ is the ordered singleton target and $R_C$ the connected residual.
The sum runs over connected clusters of the physical graph, including those
with gaps in snake order. Each disjoint collection appears exactly once.
Repeated remaining-site sets share one subproblem; no background inverse is
used. Compilation caches this structural graph; each evaluation builds fresh
backend tensors and releases subproblem MPOs after their last use. Ordered
joint factors use the same recurrence after forming each connected local
target as `exp(A_C) @ exp(B_C) @ ...`. A separate 2×2 set-partition check
compares joint MPO and PEPO outputs at orders 2–4 with spatial reuse on/off.

| Control | Meaning in recursive assembly |
| --- | --- |
| `cluster_size=p` | Largest connected spatial cluster |
| `assembly_state_budget=4096` | Hard limit on distinct subproblems, including the empty base; raises at compilation |
| `assembly_chi` | Bond cap after each addition batch; `None` skips intermediate compression |
| `assembly_batch_size` | Additions per subproblem before compression; `"auto"` or `None` means 1 |
| `max_bond`, `cutoff` | Local residual factorization controls |
| `chi` | Optional final one-shot facade compression |

`collection_budget` does not limit this mode. `graph_assembly="auto"` and
`"exact"` both retain complete collections; `"bounded"` and
`max_collection_order` are rejected. Exceeding the state budget raises and
never selects a reduced collection expansion. `assembly_state_budget=None`
removes that guard explicitly. Small exact checks can use
`assembly_chi=None, max_bond=None, cutoff=0.0` and omit final `chi`.

Before each truncation the assembler prepares an orthonormal environment
with an exact reverse sweep. `assembly_cutoff=None` then uses fixed ranks;
a cutoff requests adaptive ranks. `assembly_form` sets the output sweep
direction. Fixed-rank derivatives retain the backend SVD's locally smooth,
spectrally separated contract; adaptive rank selection is discrete. Dense
NumPy and Torch paths are tested. If NumPy thin SVD does not converge on
a finite two-dimensional compression matrix, an installed SciPy provides a
`gesvd` retry; without SciPy the original failure propagates. The retry
keeps the requested rank policy and is not a convergence guarantee. Native
charge/fermionic recursive assembly is currently unsupported.

This reduces collection enumeration, but does not guarantee low cost on wide
graphs: subproblem counts, residual span ranks, and tensor multiplication
still grow. The state budget is not a byte limit. Intermediate product/sum
bonds can exceed `assembly_chi` before compression, and autodiff may retain
buffers for backward. Intermediate truncation can accumulate and depend on
branch order; check convergence in `assembly_chi` separately from $p$.

The [5×6 order-four numerical record](../../development/notes/2026-09-28-graph-mpo-5x6-numerical.md)
measures actual construction, peak memory and selected matrix-element errors
against a separate complete collection reference. It illustrates why a
compiled complete plan alone does not establish numerical accuracy.

Reports expose `graph_planner="subset_dp"`, the exact nonempty
`graph_collection_count`, `graph_planner_state_count`, the state budget,
`assembly_peak_cached_states`, compression count, and peak bond dimensions.
`graph_collection_truncated` stays false. `assembly_truncated` conservatively
records bond reductions (including redundant zero channels), while adaptive
`assembly_discarded_weights` are individual sweep diagnostics, not a global
accumulated error bound. The stable `api_info.truncated` also reflects graph
collection omission and assembly bond reduction.

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
prepares an orthonormal environment and applies a semantic fixed-rank TT-SVD
after each batch. Streaming therefore
introduces numerical truncation after each batch, so its result can depend on
the path order. The streaming accumulator contains the singleton rail plus an
additive sum of the selected paths. A bounded one-cluster plan omits products
of separate residual paths. If the planner selected an exact or higher-order
bounded collection plan, streaming batches those collection paths as well.
For graphs needing no explicit collection plan (for example separated chain
intervals), streaming delegates to the complete subproblem recurrence so
products of disjoint residuals are retained. That path uses
`assembly_state_budget` and reports `graph_planner="subset_dp"`. The returned report records the
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
and recursive intermediate compression are rejected for symmetric clusters
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

For a fixed 2D geometry, create the term basis and graph once. Snake ordering
and connected-cluster selection are separate inputs:

```python
from pepsy.operators import ClusterLattice, MPOBasis
from pepsy.tensors import OneDMap

graph = ClusterLattice.square(2, 2)  # open boundaries; explicit graph owns them
mapper = OneDMap(2, 2, mode="snake")
terms = [(("ZZ", 1.0), edge) for edge in graph.edges]
terms += [(("X", 0.5), site) for site in graph.sites]
basis = MPOBasis.from_terms(terms, shape=(2, 2), mapper=mapper)
compiled = basis.compile_graph_cluster_expansion(
    graph=graph,
    cluster_size=3,
    graph_assembly="exact",
    collection_budget=128,
)
semantic, report = compiled.exp(-1j * 0.02, return_report=True)
U = semantic.to_mpo()
print(report.cluster_mode, report.graph_assembly)
print(report.graph_collection_truncated)
```

`compile_exp()` on the same expansion reuses its callable; the `MPOBasis`
compile helpers reuse that callable for matching settings too. Each evaluation
still returns a fresh semantic MPO. `compiled.last_report` is `None` before the
first evaluation and exposes the latest report afterward. The optional
`return_report=True` returns `(semantic_mpo, report)` from `exp`, `evaluate` or
the callable itself. With the default `False`, the return type is unchanged.

For a full trace only, call `compiled.trace_exp(step, parameters=...)` or
`expansion.trace_exp(...)`. It returns the unnormalized trace by default;
`normalized=True` divides by `phys_dim**L`. This computes local ordered
target traces and connected scalar residuals, then sums all compatible
cluster placements through a cached subset recursion. It does not construct
or compress MPO bonds. `state_budget=100000` caps the number of subset
states and raises if exceeded; it is separate from MPO assembly budgets.
The trace is of the **complete chosen-order expansion**, so it may differ
from a returned MPO when `graph_assembly` omits collections or any local
or assembly bond truncation is active. Torch gradients and small JAX JIT
calls remain backend-native. Large graph recursions may still cost memory
and compilation time; local matrices scale as `phys_dim**p`. Full
`torch.compile` capture of this call remains unsupported by the local
Autoray target path; eager Torch gradients are validated.

`compiled.cache_info` describes requested settings, whereas the returned report
describes the selected assembly. In particular, `cache_info["graph_assembly"]`
can remain `"auto"` while `report.graph_assembly` is `"bounded"` and
`report.graph_collection_truncated` is true. Inspect the report when comparing
cutoffs or MPO and PEPO results. `graph_assembly="exact"` means complete
compatible collections at the chosen spatial cutoff; it does not mean the
full-system exponential is exact for a truncated cutoff.


`compile_exp()` caches interval/factor topology and rebuilds backend tensors on
each evaluation, so Torch/JAX autodiff graphs stay current. `max_bond` is an
explicit local Schmidt-rank cap; it is separate from the history guard used by
the higher-order MPO family.

`MPOClusterBasisExpansion` and `CompiledMPOClusterExp` remain compatibility
aliases for `MPOClusterProductExpansion` and `CompiledMPOClusterProduct`.

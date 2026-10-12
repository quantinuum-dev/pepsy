# Shared interaction-graph cluster plans

`pepsy.operators.ClusterPlan` derives finite connectivity from located term
supports and shares structural bookkeeping across MPO, graph-PEPO and Gaugy
Pauli expansions. It supports irregular graphs, triangular lattices,
long-range edges, higher-body terms and disconnected components.

This planner and the MPO/PEPO implementations belong entirely to Pepsy and
work without Gaugy. Gaugy's optional downstream use calls public Pepsy APIs;
its sparse Pauli expansion and optimization code stays in Gaugy. Reusing a
Pepsy plan does not introduce a shared source package or reverse dependency.

```python
from pepsy.operators import (
    ClusterPlan, MPOClusterFactor, MPOClusterProductExpansion,
    PEPOClusterProductExpansion,
)

# A four-site open chain with nearest and next-nearest interactions.
terms = [
    {"sites": (i, i + distance), "paulis": "ZZ", "coefficient": coupling}
    for distance, coupling in ((1, 0.3), (2, 0.1))
    for i in range(4 - distance)
]
terms += [{"sites": (i,), "paulis": "X", "coefficient": 0.2}
          for i in range(4)]
plan = ClusterPlan.from_terms(terms, sites=4, cluster_size=3, cyclic=False)
factors = [MPOClusterFactor(terms)]
mpo = MPOClusterProductExpansion.from_plan(plan, factors, cutoff=0)
pepo = PEPOClusterProductExpansion.from_plan(plan, factors)

assert mpo.cluster_plan is pepo.cluster_plan is plan
print(plan.counts)                  # {1: 4, 2: 5, 3: 4}
print(plan.collection_count())      # 12, including the singleton background
a = mpo.trace_exp(-0.05j, normalized=True)
b = pepo.trace_exp(-0.05j, normalized=True)
```

With matching Gaugy installed, the same dictionaries and plan work directly:

```python
from gaugy.cluster import PauliClusterBasis
pauli = PauliClusterBasis.from_plan(plan, terms)
c = pauli.compile_exp().exp(-0.05j).trace(normalized=True)
```

For ordered products, use `PauliClusterProductExpansion.from_plan(plan,
factors)` with `PauliExponential` factors or sequences of located Pauli terms.
All three `from_plan` product frontends use algebraic order: `[A, B]` means
`exp(step*A) @ exp(step*B)`. Legacy `ExactClusterBasis.from_plan` and
`TaylorClusterBasis.from_plan` retain Gaugy's application order. Build the
plan from the **union of all factors' supports**.

## Geometry and boundaries

- `from_terms` accepts located objects with `.sites` or mappings with `sites`.
  Square homogeneous direction templates must first be expanded into located
  terms. `from_supports(sites, supports, ...)` needs only site supports.
- Supply all sites to retain isolated sites. Without `sites`, `from_terms`
  uses first-occurrence order. MPO and Gaugy terms use integer positions in
  `plan.sites`; plan queries themselves use the original hashable labels.
- Higher-body supports induce all pairwise connectivity edges. Duplicate
  supports and their original ordering remain in `plan.supports`; only the
  undirected topology edges are deduplicated. Numerical factors retain every
  physical term occurrence, including repeated bonds on small periodic tori.
- Zero-valued coefficients do not remove edges. Parameter values can change
  without rebuilding geometry.
- `cyclic` is explicit metadata, either a bool or one bool per dimension.
  It does **not** add wraparound interactions. Include those terms yourself.
  The planner cannot distinguish all physical embeddings or boundary choices
  from an abstract graph. Complete rectangular coordinate labels provide a
  shape; `shape=(Lx, Ly)` gives row-major coordinates for other labels.

For example, a triangular axial lattice uses displacements `(1, 0)`, `(0, 1)`
and `(1, -1)`. Add the desired further-neighbor displacements to the term
list; no square-lattice assumption enters connected-cluster enumeration.
The existing MPO convenience API also accepts `graph="interactions"` to
infer connectivity from every factor. Existing interval and `graph="auto"`
defaults are unchanged. With `from_plan`, every term support must be a
connected subset of the supplied graph within `cluster_size`; use the union
of term supports when constructing the plan. Legacy explicit-graph MPO calls
retain their existing support rules, including terms connected through
intermediate sites, also when binding runtime coefficient vectors.

## Counts, symmetry and caches

`cluster_size` counts connected **sites**, not bonds or Taylor degree.
Inventory enumeration is lazy. `plan.clusters` returns finite placements in
stable size/site order, and `plan.counts` counts those placements by size.

`plan.partition_count(cluster)` counts proper connected-block partitions of
one cluster; the full cluster as a single block is excluded. Set
`proper=False` to include it. `plan.partitions(cluster)` materializes those
partitions subject to `max_partitions`. `plan.collection_count()` counts
complete disjoint collections of retained clusters across the entire graph,
including the all-singleton background. Use `include_background=False` to
match existing MPO collection-report conventions. Counts use exact Python
integers and a subset recurrence, without enumerating every collection.

`plan.spatial_symmetries()` checks finite coordinate candidates against the
actual edges and support multiplicities. Candidates include 1D inversion,
translations along periodic axes, square rotations/reflections, and axial
triangular rotations/reflections. Returned permutations include translation
generators, not necessarily every composed group element. An OBC translation
is not a finite automorphism, even when interior local clusters are reusable.
No exhaustive arbitrary-graph automorphism search is promised.

Geometry alone does not establish Hamiltonian symmetry. For a numerical
certificate, provide ordered factors of `(binding_key, ((site, operator_key),
...))` records to `spatial_symmetries` or `reuse`. Keys must identify immutable
operator definitions and parameter bindings, **not current parameter values**.
Independent equal parameters must have different keys. `plan.reuse(())`
groups geometry only; `plan.reuse()` conservatively disables numerical reuse.
The returned `ClusterSymmetryPlan` contains representative/axis mappings,
finite multiplicities, and search-fallback counts. Above the local permutation
budget it safely uses identity matching, potentially finding less reuse.

MPO and graph-PEPO automatically use verified local operator/binding matches.
The graph local-product engine and square Pauli PEPO builder also match each
ordered factor separately. Thus a shared `A` can reuse its local exponential
even when independent parameters in `B` prevent reuse of the complete
`exp(A_C) @ exp(B_C)` target. Physical-axis permutations are applied before
ordered multiplication. Equal-size square exponentials remain batched.
`spatial_reuse=False` disables both levels of reuse. Explicit coefficient
vectors keep separate slots, and opaque callback bindings do not gain
factor-level sharing. Numerical factor caches live for one evaluation only;
the reusable plans contain no coefficients, exponentials or autodiff graphs.

Gaugy `ExactClusterBasis` supports `symmetry="auto"` for connected logs and
`compile_trace`. It validates local Pauli words and parameter identities,
including independent `PauliParameter` slots. Opaque bindings disable reuse.
Existing named symmetry catalogs remain available and do not enumerate the
entire finite inventory just to construct a basis.

Bounded process-local caches store graph inventories, local partition
templates and complete collection recurrences. Equivalent graphs share the
inventory even with different site labels or edge orientation; operator
orientation remains in backend term data. `ClusterPlan.cache_info()` reports
cache hits and sizes. Coefficients, exponentials and autodiff graphs are fresh
for each evaluation. `max_clusters`, `max_partitions` and `state_budget` raise
on exhaustion; they never return silently truncated counts. Planning remains
combinatorial at large cutoff or graph width.

## Representation boundaries

| Entry point | Located input | Result of `exp(step)` |
| --- | --- | --- |
| `MPOClusterProductExpansion.from_plan` | `MPOClusterFactor`/term sequences | MPO operator wrapper |
| `PEPOClusterProductExpansion.from_plan` | Same factors as MPO | Active square or graph PEPO blocks |
| `PauliPEPOBasis.from_plan` | One sequence of located terms | Active square or graph PEPO blocks |
| `GraphClusterExpansionPlan.from_plan` | Uniform dense edge/onsite matrices | Active graph PEPO blocks |
| Gaugy `PauliClusterBasis.from_plan` | One sequence of located Pauli terms | Pauli cluster operator |
| Gaugy `PauliClusterProductExpansion.from_plan` | Algebraically ordered Pauli factors | Pauli cluster operator |

`PEPOClusterProductExpansion.from_plan` and `PauliPEPOBasis.from_plan` accept
`layout="auto"` (default), `"square"`, or `"graph"`. Automatic selection returns
`SquarePEPOClusterProductExpansion` for a complete square nearest-neighbor
graph with a two-dimensional shape and fixed one-/two-site Pauli product
terms. Other supported plans retain `GraphPEPOClusterProductExpansion`.
`layout="square"` also supports other dense bosonic graph interactions by
explicitly routing their virtual bonds onto a rectangular square lattice;
`layout="graph"` explicitly preserves the previous generic representation.
Direct `GraphPEPOClusterProductExpansion.from_plan` always selects the graph
builder.

The graph builder supports nonuniform couplings and higher-body terms using
exact local products and spanning-tree residual factorizations.
`materialize=True` produces a generic Quimb `TensorNetwork`. Ordered graph
PEPO materialization, active-block `to_dense()`, PEPO `trace_exp` and
`residuals` preserve Torch/JAX values and gradients. This applies to the
`GraphPEPOClusterProductExpansion` route; the legacy fixed-dense
`GraphClusterExpansionPlan` keeps its existing numerical construction.

Graph products accept `factorization="auto"` (default) or `"fixed"`, and
`spatial_reuse=True|False`. Fixed mode uses exact matrix-unit splits with
shape-determined ranks, requires `max_tree_rank=None`, and preserves zero
coefficient derivatives. Auto preserves the existing numerical SVD route for
NumPy; uncapped tensor backends use fixed splits, while capped tensor backends
use a static-rank backend SVD. Differentiation through a truncated or degenerate
SVD has the usual rank/gap limitations; fixed mode avoids that decomposition.
Tensor-valued zero blocks are retained during graph compaction. Fixed channels
can require substantially larger bonds than numerical SVD compression.

```python
builder = PEPOClusterProductExpansion.from_plan(
    plan, factors, layout="graph", factorization="fixed",
)
active = builder.exp(step, parameters=params)  # Torch/JAX inputs remain live.
matrix = active.to_dense()                    # Explicit small-system check.
# Or builder.exp(step, parameters=params, materialize=True) for the network.
```

`max_tree_rank` affects materialization and `trace_exp`, while the explicit
`partition_trace_exp` shortcut ignores it. Reports
include the requested factorization policy. NumPy SVD reconstruction errors
are local diagnostics, not a global error certificate; backend/fixed
factorizations report `None` for uncomputed reconstruction error and norm.
No backend value is copied to the host to produce those diagnostics.

## Shared square plans and 2D PEPO construction

```python
from pepsy.operators import ClusterPlan, MPOParameter, PauliPEPOBasis

edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
terms = [((i,), "X", MPOParameter("h")) for i in range(4)]
terms += [(edge, "ZZ", MPOParameter("J")) for edge in edges]
plan = ClusterPlan.from_supports(
    4, [term[0] for term in terms], shape=(2, 2), cluster_size=3,
)
builder = PauliPEPOBasis.from_plan(plan, terms)
assert builder.cache_info["representation"] == "square-pepo"
active = builder.exp(-0.05j, {"h": 0.3, "J": 0.7})
pepo = active.to_pepo()  # Quimb PEPO with Lx=Ly=2 and square virtual legs.
# Equivalent: builder.exp(..., materialize=True)
```

The square adapter retains one coefficient slot per input term and the
original parameter references and factor scales. Repeated terms remain
separate slots. Existing square local-target and lower-support symmetry reuse
therefore recognizes shared bindings; runtime override vectors remain
independent. Local cluster records use the supplied plan's cached inventory.
The square tensor network can use the existing 2D PEPO contraction workflows.
`ActivePEPOBlocks.to_dense()` explicitly materializes and contracts it.

Provide rectangular coordinates in row-major order, or supply `shape=(Lx, Ly)`
for row-major integer/arbitrary labels. A one-dimensional inferred shape
keeps graph output unless you explicitly provide a two-dimensional shape.
Automatic conversion uses the complete nearest-neighbor graph and fixed
NumPy I/X/Y/Z terms; scalar amplitudes belong in term coefficients. Its
specialized cutoff limit remains nine. Missing bonds, diagonal/NNN edges,
higher-body terms and non-Pauli matrices retain graph output in automatic
mode. Explicit `layout="square"` routes these through the graph residual
engine, with `cache_info["representation"] == "routed-square-pepo"`.
Permuted coordinate labels are rejected; arbitrary labels use row-major
`shape`. Native charge, fermion and string routing is not supported.

OBC, cylinders and fully periodic squares are supported. Wraparound terms
must still be supplied explicitly. For a length-two periodic axis, each
supplied term is assigned one deterministic virtual route, even though two
routes share its endpoints. Terms are never implicitly doubled or dropped.
Routing uses deterministic shortest Manhattan paths, tensor-products wires
that share square bonds, and transmits wires independently of physical
operators at intermediate sites. It preserves every disjoint cluster product
even when its wires cross or pass through another occupied site. Routing
does not add sites or edges to the interaction plan, change cluster counts,
or reorder exponential factors. Bond dimensions can grow multiplicatively,
which is why this broader routing is explicit. Existing graph blocks also
offer `active.to_square((Lx, Ly), cyclic=(False, False))`.

The default square result is `ActivePEPOBlocks`; `materialize=True` returns a
Quimb `PEPO`. `compile_exp()` caches square structure. The options
`spatial_reuse` and `factorization` also apply to graph output; `factorization="fixed"` supports
Torch/JAX differentiable materialization without numerical rank selection
and requires `max_tree_rank=None`. Its larger fixed channels are distinct
from rank-capped numerical construction. `return_report=True` reports the
layout, cutoff, cluster counts, factorization policy and rank cap. PEPO
`trace_exp` builds and traces the selected operator. `residuals()` lazily uses the shared local
MPO engine and keeps integer indices in plan order.

Gaugy's shared-plan Pauli factories use the same layout selection;
`bound.to_pepo()` returns a 2D PEPO for compatible square inputs. The
connected-log and residual-partition scalar endpoints remain distinct.

## Explicit differentiable compression

Ordered square and graph builders offer reference-based compression:

```python
# Prepare outside JIT/grad, preferably at representative nonzero parameters.
compression = builder.prepare_compression(
    step, parameters=params, max_tree_rank=2,
)
# theta may now contain fresh Torch/JAX autodiff parameters.
active, report = builder.exp(
    step, parameters=theta, compression=compression, return_report=True,
    materialize=False,
)
```

The returned `ClusterCompressionPlan` stores immutable NumPy tree subspaces
chosen by a reference SVD. Preparation explicitly copies reference residuals
to the host. Replay projects fresh, exact local residuals into those fixed
subspaces using backend matrix products, before virtual routing and dense
materialization. Geometry remains cached; reference tensors and their
autodiff graphs are not retained. Use `max_tree_rank=None` on the builder;
the compression plan supplies its own ranks. This is separate from Quimb
`compress=True` and the two options cannot be combined.

This differentiates the **projected operator**, including at zero parameters,
with fixed ranks and subspaces. It does not differentiate reference SVDs,
adapt ranks inside an objective, or guarantee an accurate approximation far
from the reference. Refresh the plan explicitly when needed. A zero reference
can choose uninformative subspaces. `trace_exp(..., compression=plan)` builds
and traces the projected PEPO. To measure an existing result, use
`pepsy.operators.trace_pepo(active)` or `active.trace()`. The explicitly named
`partition_trace_exp()` remains the uncompressed scalar expansion.

`report["compression"]` records the method, ranks, and reference local
Frobenius errors/norms. These are neither current-parameter error estimates
nor a global error bound. Native symmetry compression and optimal global
PEPO fitting are outside this dense local-projection API.

## Materialization reports

`exp(..., return_report=True)` returns `(result, report)` for ordered square
and graph builders. `report["materialization"]` counts actual work in that
evaluation: requested/evaluated/reused local products and factor exponentials,
actual exponential batch sizes, lower-support contractions and their reuse,
and graph partition products. Graph partition products and square tensor
contractions are distinct operations; a zero count is meaningful. Uniform
square counts refer to source-shape requests, not every translated placement.
Counters are evaluation-local and do not inspect tensor values or synchronize
backend arrays. Reports also give active-block counts and estimated dense site
storage, excluding contraction/autodiff workspace and later Quimb compression.

With matching Gaugy, `bound.to_pepo(return_report=True)` forwards the report.
`bound.materialization_info()` performs a fresh active-block evaluation without
allocating dense site tensors. It is distinct from structural connected-log
`symmetry_info()`. Use `bound.prepare_compression(max_tree_rank=...)`, followed
by later `bound.to_pepo(compression=plan)`, for the same compression workflow.

The uniform `GraphClusterExpansionPlan` applies one `twosite_op` to each
unique oriented graph edge. It does not infer distinct couplings or multiply
by `plan.supports` occurrence counts; use the product frontend for those.

PEPO `trace_exp` constructs and traces the PEPO, unnormalized by default;
Gaugy's bound `trace()` follows that contract. Active blocks are a sparse
PEPO representation: their physical traces and virtual bonds are contracted
without allocating dense site tensors. The default `state_budget=100000`
caps sparse site entries and pairwise join work; exceeding it raises rather
than changing the approximation. Backend-valued zero blocks remain live for
autodiff. Large PEPO contractions can still be expensive. Sparse
`state_budget` and materialized `contract_opts` are mutually exclusive; a
custom sparse budget is rejected instead of ignored, and `contract_opts` must
be a mapping. For a materialized network,
`trace_pepo(pepo, **contract_opts)` uses Quimb's public contraction.

`partition_trace_exp` is the separate scalar partition shortcut (Gaugy bound
`partition_trace`). Gaugy's `connected_trace_exp`/`log_trace_density` use a
different connected-log approximation. They need not agree at a partial spatial cutoff. Full-size
connected components recover the exact ordered product before rank
truncation, up to numerical precision.

The shared planner belongs to **`pepsy.operators.ClusterPlan`**. Gaugy's
older `gaugy.cluster.ClusterPlan` name continues to mean its Pauli execution
plan and has not been replaced.

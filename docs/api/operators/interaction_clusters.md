# Shared interaction-graph cluster plans

`pepsy.operators.ClusterPlan` derives finite connectivity from located term
supports and shares structural bookkeeping across MPO, graph-PEPO and Gaugy
Pauli expansions. It supports irregular graphs, triangular lattices,
long-range edges, higher-body terms and disconnected components.

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
| `PEPOClusterProductExpansion.from_plan` | Same factors as MPO | Active graph PEPO blocks |
| `PauliPEPOBasis.from_plan` | One sequence of located terms | Active graph PEPO blocks |
| `GraphClusterExpansionPlan.from_plan` | Uniform dense edge/onsite matrices | Active graph PEPO blocks |
| Gaugy `PauliClusterBasis.from_plan` | One sequence of located Pauli terms | Pauli cluster operator |
| Gaugy `PauliClusterProductExpansion.from_plan` | Algebraically ordered Pauli factors | Pauli cluster operator |

Graph PEPO factories return `GraphPEPOClusterProductExpansion`. They support
nonuniform couplings and higher-body terms using exact local products and
spanning-tree residual factorizations. `materialize=True` produces a generic
Quimb `TensorNetwork`, because Quimb's square `PEPO` container has fixed
lattice legs. Graph PEPO materialization currently requires NumPy values;
it rejects differentiable backend values instead of detaching them.
`trace_exp` and `residuals` retain Torch/JAX gradients. `max_tree_rank` affects
materialization, not scalar trace closure; factorization errors in a report
are local diagnostics, not a global error certificate.

The uniform `GraphClusterExpansionPlan` applies one `twosite_op` to each
unique oriented graph edge. It does not infer distinct couplings or multiply
by `plan.supports` occurrence counts; use the product frontend for those.

`trace_exp` is the residual-partition trace, unnormalized by default. Gaugy's
`connected_trace_exp`/`log_trace_density` use a different connected-log
approximation. They need not agree at a partial spatial cutoff. Full-size
connected components recover the exact ordered product before rank
truncation, up to numerical precision.

The shared planner belongs to **`pepsy.operators.ClusterPlan`**. Gaugy's
older `gaugy.cluster.ClusterPlan` name continues to mean its Pauli execution
plan and has not been replaced.

# PEPO cluster expansion and joint ordered products

For arbitrary interaction graphs and a common MPO/PEPO/Pauli plan, see the
[shared interaction-cluster guide](interaction_clusters.md).

For the canonical MPO/PEPO entry-point map, step convention, and return-type
summary, see the [unified exponential API](exponentials.md). This page gives
the detailed `pepsy.operators.pepo_cluster` cluster and fixed-channel PEPO
reference. The higher-order MPO and MPO cluster families are separate APIs.

`cluster_size` is the common spatial cutoff keyword across PEPO, MPO and
Gaugy Pauli clusters. PEPO's existing `order` keyword is an equivalent alias;
conflicting values are rejected. Both reusable dense plans and the Pauli
basis expose `.cluster_size`. Defaults remain three for dense plans and four
for the Pauli basis. Ordered products inherit the cutoff of their bases.
See the [shared call table](exponentials.md#shared-connected-cluster-calls).
For the dense square dataclass, retain `order` when copying configuration with
`dataclasses.replace(plan, order=p)`; `cluster_size` aliases this field rather
than storing a second cutoff.

For a fixed dense generator, `plan.compile_exp().exp(step)` now gives the
same sign convention as Pauli PEPO and MPO exponentials. It returns active
blocks by default; use `materialize=True` for a Quimb operator. This is
`plan.build(-step, materialize=False)` with reusable geometry, not backend
JIT. Existing `build(beta)` defaults and numerical behavior are unchanged.
Use `PauliPEPOBasis` for runtime coefficient vectors, autodiff and trace-only
calls.

`build_itf_cluster_expansion_pepo` constructs a dense square-lattice PEPO
approximation to `exp(-beta * H)` for Pepsy's transverse-field Ising
convention, `H = J * sum Z_i Z_j + field * sum X_i`:

```python
import numpy as np

from pepsy.operators import build_itf_cluster_expansion_pepo

pepo = build_itf_cluster_expansion_pepo(
    4,
    4,
    0.05,
    J=1.0,
    field=0.5,
    order=3,
    cyclic=True,
)
```

Order 1 keeps only the local exponential. Order 2 factorizes the connected
two-site residual into operator-Schmidt channels. Order 3 adds the residual
for every three-site straight path and corner, represented by local PEPO
entries with two active virtual directions. Order 4 adds four-site tree
clusters: degree-three stars and non-loop paths, as well as every present
four-site plaquette loop. The PEPO is built directly on the square lattice, so
`cyclic=True` does not first create a snake MPO and then embed it. The
plaquette correction is carried by a fixed-rank tensor ring with one paired
operator-history space on each of its four virtual bonds.

For repeated evaluations, cache the geometry and C4 orbit structure in a
plan. Set `materialize=False` to keep only active virtual-sector blocks until
the PEPO is needed:

```python
from pepsy.operators import ClusterExpansionPlan

plan = ClusterExpansionPlan(
    4,
    4,
    np.kron(np.diag([1.0, -1.0]), np.diag([1.0, -1.0])),
    0.5 * np.array([[0.0, 1.0], [1.0, 0.0]]),
    order=3,
    cyclic=True,
    symmetry="C4",
)
active = plan.build(0.05, materialize=False)
pepo = active.to_pepo()
```

`build(beta)` uses the `exp(-beta * H)` convention with real or complex
`beta`. It rebuilds numerical blocks using NumPy and the combined dtype of
the plan and `beta`, reusing the planned geometry. Input operators are not
modified. The default result is a Quimb `PEPO`; `materialize=False` returns
`ActivePEPOBlocks`. Every successful block assembly updates `plan.last_report`
before optional dense materialization.
The C4 path requires a real/Hermitian reshuffled edge residual; complex
evolution that violates this condition needs a plan with `symmetry=None`.

### Shapes, tensor ordering, and ownership

For local dimension `d`, `onesite_op` has shape `(d, d)` and `twosite_op`
has shape `(d**2, d**2)`. Both are matrices with output rows and input
columns; two-site product order follows the edge's source then target.
They must use the same local physical basis.

`active.blocks[site][sector_tuple]` holds an `(output, input)` matrix of
shape `(d, d)`. The sector tuple follows `active.site_directions[site]`.
`active.to_pepo()` transposes each physical block into Quimb's `urdlbk`
layout: present virtual directions up/right/down/left, then physical input
`b` and output `k`. Missing open-boundary virtual legs are omitted. Use the
returned tensor's named indices when manipulating arrays.

Building updates the plan's report, and materialization leaves the active
blocks unchanged. Inputs are not mutated, but input arrays can be retained
by the plan; keep them fixed during reuse. `compact_bonds=True` remaps sector
labels only in the returned PEPO. This structural compaction preserves the
operator exactly; `max_tree_rank` and cutoff settings instead control
approximations during construction.

Use `return_report=True` to receive local residual and storage diagnostics
alongside the result. For a four-site path, `max_tree_rank` optionally caps
the internal path SVD rank:

```python
active, report = build_itf_cluster_expansion_pepo(
    1,
    4,
    0.03,
    order=4,
    materialize=False,
    return_report=True,
    max_tree_rank=16,
)
assert report.residual_norms["four_site_path"] < 1e-10
```

For ITF, `build_itf_cluster_expansion_pepo` enables this C4 reduction by
default. It solves one straight and one corner representative and rotates
their active blocks to the other orientations. The generic P=5/P=6 stage
also transports tree factorizations across C4-related shapes when the finite
residual matches the rotated representative; otherwise it falls back to a
direct dense solve.

This dense implementation supports orders 1–9 with ordinary dense matrices.
Orders five through nine use a recursive generic support contraction and
spanning-tree SVD; orders above 9 raise `NotImplementedError` explicitly.
For the broader
cluster-expansion design, the reference implementation is the Julia
[`ClusterExpansions`](https://github.com/sanderdemeyer/ClusterExpansions)
package.

## Hamiltonian-aware reuse in Pauli PEPOs

`PauliPEPOBasis.compile(..., spatial_reuse=True)` enables automatic local
Hamiltonian equivalence checks by default, for both homogeneous and located
Pauli terms. Ordered `PEPOClusterProductExpansion` factors must use the same
`spatial_reuse` setting. The check preserves the complete ordered factor
list, Pauli labels, coefficients and directed/parallel bond occurrences.
Allowed translations, rotations, reflections and other graph relabelings can
share an exact local target. Located builds also reuse contractions of the
frozen lower-order PEPO under the same verified site permutation. Each
placement still receives its own residual subtraction and tensor insertion;
backend values are cached only for that evaluation and cluster order. No
extra truncation is introduced. A [5×6 stage and memory profile](../../development/notes/2026-09-28-pepo-stage-profile.md)
measures target evaluation, lower contraction, Pauli expansion, insertion and
process peak RSS separately.

```python
from pepsy.operators import PauliPEPOBasis

basis = PauliPEPOBasis.compile(
    5, 6, [("onsite", "X", 0.2), ("edge", "ZZ", 0.7)],
    order=4, spatial_reuse=True,
)
compiled = basis.compile_exp()
active = compiled.exp(-0.01j)
print(compiled.cache_info["spatial_plans"])
print(compiled.cache_info["last_local_targets_evaluated"])
```

Compilation prepares structural plans for the usual default and coefficient
vector bindings; mixed binding modes in ordered products are added on demand.
A `coefficients=` override preserves each slot's independence, even if values
are equal. Repeated `MPOParameter` references can establish equivalence for
located terms; distinct parameter names or differing defaults cannot. Opaque
PEPO coefficients retain their own slot identity. Numerical targets are held
only during the current call, preserving repeated Torch evaluation/gradients.

Each `spatial_plans` entry reports `targets`, `representatives`, `reused` and
`search_fallbacks`. Entries can describe different binding modes, so do not
sum them as one build's work. `last_local_targets_evaluated` counts local
ordered targets evaluated in the latest call (zero before a build; `None`
for the unreduced homogeneous route); it does not count lower-support
contractions. Set `spatial_reuse=False` for a reference comparison. Search is limited to 4096 permutations per cluster; beyond that,
identity-only matching retains safe translation reuse.

This automatic local-target policy is separate from the existing
`symmetry="C4"` policy that transports PEPO blocks. Keep `symmetry=None` for
models without that stronger symmetry. The legacy dense
`ClusterExpansionPlan` retains its existing TI/C4 handling; it does not accept
`spatial_reuse`. Physical spin rotations and fermionic site permutations are
not inferred by the new matcher.

For uniform onsite plus symmetric nearest-neighbor interactions, different
square-lattice embeddings may have identical local Hamiltonian graphs: the
19 oriented four-site shapes have three such graph types (path, star, loop).
This can reduce local exponential work beyond geometric C4 grouping, while
all 19 shapes and their finite placements are still assembled.

## SVD-free differentiable construction

`PauliPEPOBasis(..., factorization="fixed")` requests exact fixed-index
construction throughout, including the generic homogeneous tree route at
orders five through nine. The existing `factorization="auto"` default retains
its fixed Pauli channels at low orders and numerical tree factorization where
applicable. `max_tree_rank=None` is required in fixed mode.

```python
from pepsy.operators import MPOParameter, PauliPEPOBasis, PauliPEPOTerm

basis = PauliPEPOBasis.compile(
    5, 6,
    [PauliPEPOTerm("onsite", "X", MPOParameter("h")),
     PauliPEPOTerm("edge", "ZZ", MPOParameter("J"))],
    order=4, factorization="fixed", spatial_reuse=True,
)
compiled = basis.compile_exp()
active = compiled(step, parameters={"h": h, "J": J})
```

The lattice, local embeddings, verified symmetry matches, Pauli channels and
tree topology are prepared independently of coefficient values. Each call
builds fresh backend arrays and an autodiff graph with fixed structural channel
shapes, including at zero coefficients/residuals. Generic tree topology has a
bounded immutable cache and is warmed by compilation. Python scalar constants
are converted at the precision of trainable coefficient slots.

Ordered `PEPOClusterProductExpansion` factors must agree on the factorization
policy. Fixed products reject `compress=True`; users may compress the resulting
operator separately. `cache_info["factorization"]` identifies the policy.
This policy concerns the Pauli/fixed-channel constructors, not the separate
legacy dense numerical `ClusterExpansionPlan` solver.

Fixed mode does not guarantee small PEPO tensors or cheap contraction. Keep
`ActivePEPOBlocks` when possible, use physical-trace pruning only where its
structural certificate applies, and inspect `active_nbytes`/`dense_nbytes`
before materialization. Native charge conversion and boundary contraction are
separate stages with their own decomposition and differentiation contracts.
A scalar array-valued loss wrapping complete fixed PEPO construction has
passed JAX `jit(value_and_grad)` on a three-site order-three chain, a
2×2 square at order four, and a two-site complex-time case, including zero
parameters and zero time. Return
an array or scalar from the JIT function: `ActivePEPOBlocks` and Quimb PEPO
objects are Python containers, not JAX outputs. This finite-case check does
not establish arbitrary-geometry JIT support or a speedup.

Torch `torch.compile(backend="aot_eager", fullgraph=True)` captures the local
matrix exponential and fixed PEPO tree factorization with correct values
and gradients. Complete PEPO construction remains an eager Torch autograd
path: full-graph capture currently stops in Autoray coefficient/map preparation
before materialization. Structural
`compile_exp()` reuse is separate from Torch graph capture.

### Automatic and supplied spatial symmetries

`spatial_reuse=True` proves local equivalence under translations, rotations,
reflections and graph relabelings, preserving directed terms and coefficient
bindings. Spin labels do not rotate automatically. Independent parameters
are not merged because their current numbers agree.

Optional `spatial_symmetries` declares full finite-lattice site permutations.
Each entry is a coordinate-to-coordinate mapping or a target-coordinate
sequence in lexicographic source-site order. For example, on a 2x2 lattice:

```python
rotation = {(i, j): (j, 1 - i) for i in range(2) for j in range(2)}
reflection = {(i, j): (i, 1 - j) for i in range(2) for j in range(2)}
basis = PauliPEPOBasis(
    2, 2, [("onsite", "X"), ("edge", "ZZ")], order=4,
    factorization="fixed", spatial_symmetries=[rotation, reflection],
)
```

Declarations must preserve lattice edges, including multiplicities, and term
bindings; invalid or unprovable declarations raise. They are checked against
the configured term bindings, while independent `coefficients=` overrides
still get their own safe local reuse plan. Declarations never force equality
of overridden values. Cache diagnostics expose `declared_spatial_symmetry_count`.
A periodic translation can be declared as a permutation; an open-boundary
wraparound translation is rejected. Automatic reuse still recognizes equal
translated local clusters on open lattices.

The older `symmetry="C4"` option transports entire PEPO blocks. In fixed mode,
its homogeneous edge slots must each be invariant under endpoint reversal,
which keeps transport valid under independent coefficient overrides. For
other oriented interactions use `symmetry=None` and term-aware automatic reuse
or validated declarations. Geometric inventories still distinguish the 19
oriented four-site shapes; reusing their local values does not remove placements.

## Cluster shapes and geometry reuse

Both `ClusterExpansionPlan` and `PauliPEPOBasis` expose `cluster_inventory`,
a fresh dictionary keyed by exact site count, from one through `order`:

```python
plan = ClusterExpansionPlan(5, 6, np.kron(np.diag([1., -1.]), np.diag([1., -1.])),
                            np.array([[0., 0.5], [0.5, 0.]]), order=4)
print(plan.cluster_inventory[4])
# {'shapes': 19, 'trees': 18, 'loops': 1,
#  'c4_shapes': 7, 'c4_trees': 6, 'c4_loops': 1}
```

These counts describe shapes on the infinite nearest-neighbor square lattice.
Unprefixed counts identify translations only; `c4_*` additionally identifies
rotations, keeping reflections distinct. Both views are reported regardless
of the plan's numerical symmetry policy. `loops` counts shapes containing a
cycle, not the number of independent cycles. The singleton counts as a tree.

| Exactly four sites | Oriented shapes | Topology |
| --- | ---: | --- |
| Straight line | 2 | Tree |
| L | 8 | Tree |
| Zigzag (S/Z) | 4 | Tree |
| T | 4 | Tree |
| Plaquette | 1 | Loop |

These are not placement counts: on a 5-by-6 open lattice the four-site shapes
have 275 tree placements and 20 plaquette placements. Nor are they counts of
factorization solves or virtual channels. On a periodic lattice, wrapping can
change the induced graph and different oriented embeddings can share a site
set; do not interpret summed shape-embedding counts as unique periodic site
clusters. Inspect the finite topology and the builder's report separately.

Geometry caches are bounded and contain immutable Python data only. Shape
levels are reused when increasing the cutoff, and finite translated embeddings
are reused for identical shapes, lattice dimensions and boundary flags. Cached
embeddings preserve orientation and multiplicity required by the builder.
No coefficients, exponentials, residual tensors or autodiff graphs are retained
by these caches. Reuse a plan (or `basis.compile_exp()`) across parameter values;
numerical targets and residuals are still evaluated for the current parameters.
This reduces geometry setup work; it does not accelerate matrix exponentials
or PEPO contraction directly.

## Finite model adapters

`ClusterModelAdapter` separates standard dense spin-model definitions from
the cluster solver:

```python
from pepsy.operators import (
    ClusterModelAdapter,
    build_model_cluster_expansion_pepo,
)

model = ClusterModelAdapter.heisenberg(J=1.0, field=0.2)
pepo = build_model_cluster_expansion_pepo(
    4,
    4,
    0.02,
    model,
    order=5,
)
```

Factories are provided for transverse-field Ising, spin-1/2 Heisenberg, and
spin-1/2 XXZ models. Custom adapters can be made from dense `twosite_op` and
`onesite_op` matrices, or recovered from a mapping/object exposing those
terms. These adapters are finite and dense; fermionic parity, native
Symmray charge blocks and graded fermionic histories are intentionally outside
this layer.

## Generic cluster geometry

The first higher-order planning surface is independent of PEPO tensor values:

```python
from pepsy.operators import generate_connected_cluster_shapes

shapes = generate_connected_cluster_shapes(5)
counts = [sum(s.nsites == n for s in shapes) for n in range(1, 6)]
assert counts == [1, 2, 6, 19, 63]
```

Each `ConnectedClusterShape` contains translation-canonical sites, nearest-
neighbour edges, diagonal-neighbour metadata, and its graph loop number. Use
`quotient_rotations=True` for a C4 planning inventory. The dense P=5–9
builders recursively subtract the actual lower-order PEPO on each support and
then perform a spanning-tree SVD factorization. Set `max_tree_rank` to
truncate those generic tree bonds. The fixed-channel Pauli builder now uses
the same connected-shape hierarchy through order 9 while keeping coefficient
and step values backend-native. On a periodic square lattice, translated
copies reuse one local residual and C4-related shapes reuse one
factorization. A loop's full graph is used for its local residual, while the
final PEPO channel uses an exact spanning-tree representation of that local
tensor. This avoids a `bond_dim**4` dense allocation at PBC vertices, but
p≥5 remains exponentially more expensive in local cluster size.
For memory-controlled p≥5 runs, pass `max_tree_rank` to
`PauliPEPOBasis.compile`; this is a fixed-rank differentiable truncation of
generic tree channels, while `None` keeps the local factorization exact.

## Arbitrary finite graph lattices

The square `PEPO` builder keeps Quimb's four-leg `PEPO` contract. For a
triangular, honeycomb, Kagome, or irregular finite graph, use
`ClusterLattice` and the graph-native builder instead:

```python
from pepsy.operators import ClusterLattice, build_graph_cluster_expansion_pepo

z = np.diag([1.0, -1.0])
x = np.array([[0.0, 1.0], [1.0, 0.0]])
y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
lattice = ClusterLattice.from_edges(
    (0, 1, 2),
    ((0, 1), (1, 2), (2, 0)),
    name="triangle",
)
active = build_graph_cluster_expansion_pepo(
    lattice,
    0.03,
    np.kron(z, z),
    0.2 * x,
    order=3,
    materialize=False,
)
operator = active.to_tensor_network()
```

`GraphActivePEPOBlocks` has one virtual leg per graph edge and contracts to a
generic Quimb `TensorNetwork`; `to_dense()` is intended for small validation
graphs. The graph solver enumerates connected induced subgraphs, subtracts the
complete lower-order graph operator, and factors every residual over a
spanning tree. This is exact at the local residual level when the tree ranks
are not capped, including for induced graph loops. It is deliberately a
separate return type because Quimb's `PEPO` wrapper is square-lattice-only.

## Internal symmetries

Geometric `symmetry="C4"` and internal charge symmetry are separate options.
`ClusterInternalSymmetry` currently validates neutral `U1`, `Z2`, `U1U1`, and
`Z2Z2` Hamiltonian terms. Sector-ordered dense bases can use
`physical_sectors`; arbitrary dense bases can use local `generators`:

```python
from pepsy.operators import ClusterInternalSymmetry

u1 = ClusterInternalSymmetry("U1", physical_sectors={0: 1, 1: 1})
active = build_graph_cluster_expansion_pepo(
    lattice,
    0.03,
    np.kron(x, x) + np.kron(y, y),
    z,
    order=3,
    internal_symmetry=u1,
    materialize=False,
)
assert active.charge_symmetry == "U1"
```

The validator prevents charge-changing local terms and records the compatible
physical-sector metadata on the active blocks. Dense SVD factors are not
automatically relabeled as native block-sparse factors: Symmray conversion
still requires explicit `virtual_charges` whenever endpoint factors carry
nonzero charges. Fermionic graded factorization remains a separate native
Symmray subsystem.

## Coefficient-dependent real-time exponentials

For numerical, coefficient-dependent evolution, use
`build_real_time_cluster_expansion_pepo`. It accepts local terms as dense
matrices or `(coefficient, operator)` pairs and assembles each local
Hamiltonian before evaluating `exp(-1j * time * H)`. This is the exponential
of the summed Hamiltonian, not a product of independently exponentiated
terms:

```python
from pepsy.operators import build_real_time_cluster_expansion_pepo

pepo, report = build_real_time_cluster_expansion_pepo(
    4,
    4,
    0.01,
    twosite_terms=[(1.0, zz), (0.25, xx)],
    onesite_terms=[(0.5, x)],
    order=5,
    max_tree_rank=32,
    max_loop_rank=16,
    return_report=True,
)
```

For generic orders five through nine, `fit_method="quimb"` selects Quimb's
tree fit for loop-free cluster shapes and complex ALS for cyclic shapes.
`fit_steps`, `fit_tol`, `fit_solver_maxiter`, and `fit_seed` control that
numerical fit; `report.relative_residual_norms` is the local factorization
diagnostic to inspect when a loop rank is capped. The fit is intentionally
not differentiable with respect to coefficients. Use `PauliPEPOBasis` for
the existing fixed-channel autodiff route.

For loop clusters whose rank is not known in advance, set
`adaptive_loop_rank=True`. ALS then tries ranks from `loop_rank_start` to
`max_loop_rank` in `loop_rank_step` increments, stopping when the local fit
reaches `fit_tol`; `fit_warm_start=True` carries the previous fit into the
next larger ansatz. The report exposes the number of generic loop solves as
`cluster_counts["generic_loop_solved"]` and the largest generic loop rank as
`report.generic_loop_rank`:

```python
active, report = build_itf_cluster_expansion_pepo(
    2,
    3,
    1e-4j,
    order=5,
    fit_method="quimb",
    adaptive_loop_rank=True,
    loop_rank_start=1,
    loop_rank_step=1,
    max_loop_rank=8,
    materialize=False,
    return_report=True,
)
```

This is adaptive fitting of each finite cluster residual. It is separate from
the later global/environment-aware compression of the assembled PEPO.

BP loop-cluster expansion belongs to the contraction side of the workflow.
It can correct PEPO/PEPS observables or inform environment-aware truncation,
but it does not create the connected operator terms in `exp(-beta * H)`.

## Fractional-step fourth-order composition

The latest reference workflow also suggests composing several order-three
(`P=3`) cluster-expansion PEPOs at signed fractional steps. Pepsy exposes that
composition through Quimb's native PEPO multiplication:

```python
from pepsy.operators import compose_cluster_expansion_pepo

pepo = compose_cluster_expansion_pepo(
    4,
    4,
    0.02,
    twosite_op,
    onesite_op,
)
```

The default Yoshida triple jump uses coefficients `(a, b, a)` with
`a = 1 / (2 - 2**(1/3))` and `b = -2**(1/3) * a`. The three layers are
composed with `PEPO.apply`, so no global dense matrix is formed. Uncompressed
virtual bonds grow multiplicatively; pass `compress=True` and Quimb
compression options when an intermediate truncation is appropriate. For
reusable geometry, call `ClusterExpansionPlan.build_composed(beta)` on an
order-three plan. Arbitrary already-materialized Quimb layers can be composed
with `compose_pepo_layers`.

The construction follows the cluster-expansion prescription of forming exact
local cluster exponentials and subtracting the lower-cluster contributions;
see the original PEPO construction in
the [cluster-expansion paper](https://arxiv.org/pdf/1912.10512). Its smallest
loop is the four-site plaquette, represented here as an explicit active
virtual-history sector rather than as dense tensor inflation.

For a general dense local model, use
`build_cluster_expansion_pepo(lx, ly, beta, twosite_op, onesite_op, ...)`.

## Ordered PEPO products

For a noncommuting product, compile one `PauliPEPOBasis` per factor and use
`PEPOClusterProductExpansion`. Factors are listed in algebraic order, so the
following constructs `exp(A) @ exp(B) @ exp(C)`:

```python
from pepsy.operators import PauliPEPOBasis, PEPOClusterProductExpansion

A = PauliPEPOBasis.compile(4, 4, [("onsite", "X")], order=5)
B = PauliPEPOBasis.compile(4, 4, [("edge", "ZZ")], order=5)
C = PauliPEPOBasis.compile(4, 4, [("onsite", "Z")], order=5)

product = PEPOClusterProductExpansion.from_bases(
    (A, B, C),
    coefficients=(0.2, -0.3, 0.4),
)
compiled = product.compile_exp()
U = compiled.exp(0.01, compress=True, max_bond=64)
```

For every connected cluster `S`, the local target is formed jointly as
`exp(A_S) @ exp(B_S) @ exp(C_S)`, lower connected partitions are subtracted,
and the residual channels are assembled into one PEPO. No full-lattice PEPO is
built for an individual factor. All bases must use the same cluster order;
`order=2` is one joint two-site expansion, not `PEPO(order=2) @
PEPO(order=3)`. Factor coefficients, term coefficients, and the step remain
backend-native for Torch/JAX autodiff. This product path is intentionally
separate from `exp(A + B + C)`, which is a different operator unless the
factors commute. Contract the returned PEPO with the network contraction
workflow appropriate for the observable you need.

Pass `materialize=False` to `product.exp(...)` or `compiled.exp(...)` to
receive `ActivePEPOBlocks` before allocating dense Quimb site tensors. This is
useful for checking `bond_dimensions` and `dense_nbytes`; `compress=True`
requires materialization.
Storage estimates inspect shape/dtype metadata without detaching or converting
Torch blocks, so they can guard dense allocation during autodiff. The estimate
covers dense site tensors, not the contraction or backward graph.

## Trace-only cluster evaluation

When the required observable is only the full trace, use
`compiled.trace_exp(step, normalized=False)` on a compiled
`PauliPEPOBasis` or `PEPOClusterProductExpansion`. The unnormalized trace
is the default; `normalized=True` divides by `2**(lx * ly)`.
`parameters` and `coefficients` follow the corresponding `exp` call.
For an ordered product, the same factor order is used on every local cluster.

The evaluator computes the normalized trace of each exact local ordered
target, subtracts proper connected partitions as scalar residuals, then
sums all compatible disjoint placements with a cached subset recursion.
It never creates PEPO blocks, tree factorizations, virtual bonds, or a
boundary contraction. Static geometry and Hamiltonian symmetry reuse apply
to local targets. The default `state_budget=100000` caps subset states and
raises if exceeded; no collection is silently omitted. The coefficient and
step values remain backend-native, including Torch gradients and JAX JIT.

This returns the **complete selected-order cluster trace**, independent of
`max_tree_rank`, PEPO compression, or boundary approximations. If those
approximations are part of the quantity you need, trace the constructed
representation instead. The local exponentials still require matrices of
size `2**p`; the subset recursion can also become costly for large
lattices or cluster order. This is an ordinary bosonic full trace. Full `torch.compile` capture of
`trace_exp` is currently unsupported by the local Autoray target path;
eager Torch autodiff and JAX JIT on small instances are validated.

To trace an already constructed active PEPO (including its factorization
policy), close its physical blocks first:


```python
active = compiled.exp(0.01, materialize=False)
print(active.dense_nbytes, active.trace_nbytes)
trace_network = active.to_trace_network()
trace = trace_network.contract(all, optimize="greedy")
normalized_trace = trace / 2**(active.lx * active.ly)
```

The returned `TensorNetwork2D` is an **unnormalized** trace of the same PEPO.
It uses at most `dense_nbytes / physical_dim**2` bytes. Located exact
Pauli-history builders additionally certify which sectors survive a complete
physical trace: each nonroot selector must carry identity on every site in
its subtree. Removing the other sectors is algebraic and independent of
coefficient values, backend, or gradient tracking. Allowed channels remain
even when their current coefficient is zero. The full operator and the
lower-cluster operator subtraction retain every history.

`trace_sectors` records this optional builder certificate. It is not valid
for partial traces or after arbitrary edits to physical blocks. Builders
without a certificate, including rank-capped SVD trees, retain all sectors.
There is no value-based pruning or detached coefficient graph. This is a
dense bosonic trace, not a graded fermionic trace. Boundary contraction adds
its own approximation and workspace beyond these storage estimates.

Located cluster targets are evaluated in bounded batches of equal matrix
size. Each batch keeps the complete ordered factor sequence and independent
coefficients; it does not imply translation symmetry or split noncommuting
generators. A mixed located/uniform ordered product uses the located route
for every factor. The verified joint symmetry plan shares equivalent local
targets and frozen lower-support contractions; numerical values remain local
to the evaluation. The located route only constructs its site/edge component
maps, without unused homogeneous component tensors.

An independent 2×2 set-partition reference checks noncommuting joint MPO
and PEPO products at orders 2, 3 and 4, with reuse enabled and disabled.
Complete two-site fixed-channel joint MPO/PEPO scalar losses also pass JAX
`jit(value_and_grad)` for the coefficient and step. These finite checks do
not establish arbitrary-geometry JIT support or large-system bond convergence.

The order `p` controls local dimension: for physical dimension `d`, each
`p`-site target is a `(d**p) x (d**p)` matrix, or `d**(2*p)` coefficients.
The cost is exponential in `p` but not in the total lattice size `N` at fixed
`p`; the number of translated/graph-embedded clusters scales with `N`.

## Compile once, construct repeatedly

Fix the lattice dimensions, boundary conditions, Pauli term supports, cluster
order and rank/symmetry policy when creating a basis. Compile it once, then
reuse the callable for each time step or coefficient vector:

```python
from pepsy.operators import PauliPEPOBasis, PauliPEPOTerm

basis = PauliPEPOBasis.compile(
    5, 6,
    [PauliPEPOTerm("onsite", "X", coefficient=0.5),
     PauliPEPOTerm("edge", "ZZ", coefficient=1.0)],
    order=4,
    cyclic=False,
)
compiled = basis.compile_exp()
active = compiled.exp(-1j * 0.05)  # fixed H, sparse PEPO representation
next_active = compiled.exp(-1j * 0.02)  # reuse topology at another step
# Optional: active.to_pepo() materializes the Quimb PEPO tensors.
```

`PauliPEPOBasis.compile()` establishes the term maps and lattice structure;
`compile_exp()` prepares the required cluster geometry and static local
operator maps. Located Hamiltonians also prepare deterministic tree topology.
Identical ordered local graphs share their static Pauli maps across different
placements, while global site and bond indices still select distinct
coefficients. Directed endpoint order and parallel bond occurrences are
preserved. Mixed uniform/located ordered products compile every factor for
the located evaluation route. Homogeneous orders five through nine prepare
their generic shape records and source operator maps during compilation too.

Each evaluation assembles the current cluster Hamiltonians, evaluates their
exponentials (or the ordered product), subtracts the lower-order PEPO and fills
fresh active blocks. Compilation does not freeze coefficient values, numerical
residuals, SVD factors or gradients. A complete fixed Hamiltonian can use its
stored term coefficients as above; pass `coefficients=` to override values
without changing supports. If both H and the step are fixed and no new
autodiff graph is needed, retain the constructed PEPO itself for reuse.

Rebuild the basis when changing geometry, term supports, boundary conditions,
order or rank/symmetry policy; mutating an existing compiled basis is not a
cache-invalidation API. `cache_info` exposes `prepared_exp_modes`,
`localized_embedding_plans` and `localized_tree_plans` alongside the existing
homogeneous embedding and cluster counts. Static-map preparation and memory
grow with cluster size; compilation moves this cost out of evaluation.

The compiled single-factor and ordered-product PEPO callables both expose
`cache_info` and `cluster_inventory` directly:

```python
print(compiled.cache_info)
print(compiled.cluster_inventory[4])  # 19 oriented shapes: 18 trees + one loop
```

For an ordered product, `cache_info["factor_cache_info"]` contains each basis's
preparation diagnostics. Its shape inventory is counted once at the shared
spatial cutoff, not multiplied by the number of factors. These dictionaries
are independent inspection snapshots; editing them does not change a plan.
They report topology and preparation, not measured operator error or numerical
PEPO residual norms. The dense `ClusterExpansionPlan.build(return_report=True)`
interface remains the numerical residual-report surface for that construction.

## Fixed Pauli coefficient slots

`PauliPEPOBasis` is the coefficient-oriented interface. It compiles the
square-lattice topology and fixed physical Pauli channels once, then evaluates
the real-time operator
`exp(-1j * tau * H(coefficients))` without caching backend values or their
autodiff graphs:

```python
import torch

from pepsy.operators import PauliPEPOBasis

basis = PauliPEPOBasis.compile(
    4,
    4,
    [("onsite", "X"), ("edge", "ZZ")],
    order=4,
    cyclic=True,
    symmetry="C4",
)
coefficients = torch.tensor([0.5, 1.0], dtype=torch.float64, requires_grad=True)
tau = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)

# Keep the fixed Pauli channels sparse during the order-four build.
exp = basis.compile_exp()
active = exp.exp(
    -1j * tau,
    coefficients=coefficients,
    materialize=False,
)
pepo = active.to_pepo()  # use for small lattices or explicit interop
```

`basis.exp(step, ...)` is the direct form. `evaluate` and `time_evolution`
remain compatibility aliases; new code should use `exp` and `compile_exp`.

Terms with `support="onsite"` are translation-invariant one-site slots;
`support="edge"` slots apply to the ordered positive lattice directions.
Mappings with `support`, `paulis`, and optional `coefficient` fields, tuples
such as `("edge", "ZZ", J)`, and `PauliPEPOTerm` values are accepted. Pass
`beta=1j * tau` explicitly when using the cluster convention
`exp(-beta * H)` rather than the real-time `tau` shorthand.

For finite colorings or independent coefficients, add an explicit location:

```python
from pepsy.operators import PauliPEPOBasis, PauliPEPOTerm

basis = PauliPEPOBasis.compile(
    2,
    3,
    [
        PauliPEPOTerm("onsite", "X", coefficient=h00, where=(0, 0)),
        PauliPEPOTerm(
            "edge",
            "ZZ",
            coefficient=j01,
            where=((0, 0), (0, 1)),
        ),
    ],
    order=2,
)
```

Unlocated and located slots can be mixed, including across
`PEPOClusterProductExpansion` factors. The localized builder enumerates each
connected finite-lattice site subset independently, forms its actual ordered
local product, and subtracts the completed lower-order PEPO on that support.
It supports open and periodic square lattices at orders one through nine.
Translation/C4 orbit reuse remains disabled because independently varying
coefficients do not share one residual.

On a periodic dimension of length two, two distinct PEPO bonds connect the
same endpoints. Select one unambiguously with the direction measured from the
first endpoint:

```python
wrapped = PauliPEPOTerm(
    "edge",
    "XY",
    coefficient=j_wrap,
    where=((0, 0), (1, 0)),
    direction="d",
)
```

Endpoint-only locations remain valid when exactly one lattice bond occurrence
matches. Ambiguous periodic locations raise and require `direction`; they are
never silently merged. Every bond occurrence internal to a connected site set
is included in its local Hamiltonian, including parallel length-two PBC bonds.

With `max_tree_rank=None`, each localized residual is represented by an exact,
coefficient-independent Pauli-history tree. This avoids differentiating
through an SVD gauge. Setting `max_tree_rank` below the required exact history
rank opts into the existing fixed-rank backend SVD and is a controlled PEPO
approximation. Orders four and above can still have large materialized local
legs. Inspect `active.bond_dimensions` and `active.dense_nbytes` before calling
`to_pepo()` on a larger lattice.

The Pauli basis is physical and fixed. The PEPO virtual channels are separate
active history sectors for edge, pair, star, and path clusters. Coefficient
slots are fused into the 4 onsite and 16 edge Pauli components before local
Hamiltonians are assembled; static Pauli banks and small-cluster embedding
maps are cached by `compile_exp()`. This avoids coefficient-dependent SVD
channel selection and lets Torch/JAX scalar coefficients flow through local
matrix exponentials and block assembly. The fixed basis is intentionally
returned as `ActivePEPOBlocks` by default: its bond dimension is larger than
the numerical SVD reference, while its stored blocks remain sparse.
`cache_info` reports the prepared embedding-plan count and fused slot count.
PEPO–PEPS contraction and expectation-value routines are outside this
operator-construction API.

`ActivePEPOBlocks.to_pepo()` remaps global history ids separately on each
physical bond by default, so occurrence-specific channels do not pad every
leg to the network-wide history count. Pass `compact_bonds=False` only to
inspect that global labeling. `bond_dimensions` reports the compact dimension
of every oriented leg; the two maps for opposite ends of one bond agree.

## Native symmetry blocks

`ActivePEPOBlocks` is also the sparse-to-native boundary. `compact()` removes
zero and orphaned history channels, while `to_symmray_pepo()` groups the
remaining integer histories into Symmray charge degeneracies:

```python
charge_basis = PauliPEPOBasis.compile(
    2,
    2,
    [("onsite", "Z"), ("edge", "ZZ")],
    order=4,
)
active = charge_basis.exp(-1j * 0.01, coefficients=[0.2, 1.0], materialize=False)
native = active.to_symmray_pepo(symmetry="U1")
```

The conversion preserves Torch/JAX block values. It supports a homogeneous
operator charge and validates every nonzero block; a mixed-charge operator
such as an unsplit `exp(h * X)` under Z2 must first be decomposed into charge
components. `virtual_charges={sector_id: charge}` can provide explicit history
charges, and repeated charges are packed as Symmray degeneracies.

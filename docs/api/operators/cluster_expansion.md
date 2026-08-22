# `pepsy.operators.build_itf_cluster_expansion_pepo`

For the canonical MPO/PEPO entry-point map, step convention, and return-type
summary, see the [unified exponential API](exponentials.md). This page gives
the detailed cluster and fixed-channel PEPO reference.

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

The order `p` controls local dimension: for physical dimension `d`, each
`p`-site target is a `(d**p) x (d**p)` matrix, or `d**(2*p)` coefficients.
The cost is exponential in `p` but not in the total lattice size `N` at fixed
`p`; the number of translated/graph-embedded clusters scales with `N`.

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

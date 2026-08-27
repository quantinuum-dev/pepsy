# `pepsy.optimizers.tree_peps`

`TreePeps` is the first PEPS-like state class for tree-embedded tensor
networks. Every lattice site keeps one physical tensor, while the retained
virtual bonds are a validated spanning tree of an open 2D or 3D lattice.

The state exposes both coordinate and logical identities:

```python
from pepsy.optimizers import TreePeps, TreePepsPlan

plan = TreePepsPlan.from_shape((3, 4), order="snake")
state = TreePeps.rand(plan, bond_dim=4, phys_dim=2, seed=7)

state.site_tag(1, 2)       # "I1,2"
state.logical_site_tag(7)  # "I7"
state.site_ind(1, 2)       # "k1,2"
state.site_ind_1d(7)       # "k1,2" (the same physical leg)
```

The physical index is intentionally present only once. The logical 1D
address is represented by an additional tag, not by a second physical leg.
Each tensor also carries a structural `N{q}` tag, making it straightforward
to select either lattice sites or tree regions with Quimb operations.

`TreePepsPlan.from_shape` uses a canonical lattice-adjacent snake path by
default, so the default tree has virtual degree at most two. Custom tree
edges can be supplied as logical-id pairs or coordinate pairs, for example:

```python
plan = TreePepsPlan.from_shape(
    (2, 2, 2),
    tree_edges=[
        ((0, 0, 0), (1, 0, 0)),
        ((1, 0, 0), (1, 1, 0)),
        ((1, 1, 0), (0, 1, 0)),
        ((0, 1, 0), (0, 1, 1)),
        ((0, 1, 1), (1, 1, 1)),
        ((1, 1, 1), (1, 0, 1)),
        ((1, 0, 1), (0, 0, 1)),
    ],
    max_virtual_degree=3,
)
```

The state API includes exact `norm`, `to_dense`, and local observable readout,
together with `show`, `canonicalize`, `canonize_subtree`, and `compress`.
Canonical operations track `canonical_region` and `orthogonality_center`, and
store each proven outward isometry in the tensor's Quimb-compatible
`left_inds`. Moving a known center uses only the unique tree path and skips QR
when those local proofs already establish the required edge gauge. A
multi-site canonical region can be reduced to a center before the path move,
and the center-oriented compression sweep performs the inward edge reductions
without a redundant full-tree QR. Callers that already use Quimb-style
optimizer state can pass a mutable `info_c` mapping to synchronize
`cur_orthog`, `canonical_region`, `isometry_map`, and `left_inds` snapshots.

The first operator layer is now available through `TreePepo` and
`TreeSubPepo`:

```python
from pepsy.optimizers import TreePepo, TreeSubPepo

gate = TreeSubPepo.from_operator(plan, dense_gate, support=(0, 5))
updated = gate.apply_to(state, compress=True, max_bond=8)
value = gate.expectation(state)
```

`TreePepo` is a generic tree operator with separate input/output physical
legs. `TreeSubPepo` records the physical support and its connected tree span;
applying it fuses operator bonds into the state tree before optional
canonical compression. The full design and the future
`TreePepsOptimizer`/`TreePepsStabOptimizer` interfaces are documented in the
development plan.

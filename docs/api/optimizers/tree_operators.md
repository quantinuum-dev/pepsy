# Tree operators

[Tree API overview](tree.md)

Build full-tree or compact operators and evaluate their action on a matching tree state.

## Operator representations

The operator classes have distinct representation contracts:

| Class | Stored region | Labels and physical indices |
| --- | --- | --- |
| `TreeMPO` | The whole lattice: every node of the TreePlan, including explicit identity tensors where appropriate. | Full-lattice node tags, site labels, and physical input/output indices. |
| `SubTreeMPO` | Only a connected active region, normally the gate support's Steiner subtree. No exterior operator tensors. | The same original node IDs, site labels, and configured tag/index formats; never locally renumbered. |

`SubTreeMPO.plan` still refers to the original full-lattice plan.
`active_nodes` gives the stored region; `sites` gives physical sites present
there, including any physical routing site. `operator_support` gives the
gate support. Connecting internal nodes are included without inventing
physical site legs. Identity action outside the region is implicit.

## Layout-aware native MPOs

After selecting a plan, the canonical tree-native operator is built with
`plan.build_tree_operator(...)` or `Fermion.build_tree_operator(...)`. The
native path includes `U1FermionicArray` and `U1U1FermionicArray` tensors:

```python
from pepsy.tensors import Fermion
from pepsy.optimizers.tree import TreeLayoutFinder

finder = TreeLayoutFinder(gates, n=8, max_arity=2)
plan = finder.run(order="quality")
fermion = Fermion(spinful=True, symmetry="U1U1")
hamiltonian = fermion.hamiltonian(edges, t=1.0, U=2.0, mu=0.1)

tree_operator = plan.build_tree_operator(hamiltonian)
energy = tree_operator.expectation(opt.tn)
```

For a model-facing operator build, use `Fermion.build_tree_operator(...)`.
`Fermion.to_tree_mpo(...)` and `TreePlan.to_tree_mpo(...)` remain native
compatibility aliases. `TreeMPO` stores only the native tree representation:

```python
from pepsy.tensors import Fermion

tree_operator = fermion.build_tree_operator(
    hamiltonian=hamiltonian,
    tree=plan,
    compress=True,
)
energy = tree_operator.expectation(opt.tn)      # TreePlan-native readout
```

When a chain MPO is specifically required, call the model-level
`Fermion.to_mpo(...)` or `SymHamiltonian.to_mpo(...)` explicitly. Tree builders
do not create or attach chain MPOs.

`TreeMPO.from_terms(plan, dense_terms)` provides the corresponding ordinary
dense backend. Native fermionic `TreeMPO` objects retain Symmray arrays for
U1, U1U1, and other supported symmetries; dense and native operators are not
silently mixed with incompatible tree states.

For an exact native readout that keeps the chain MPO separate, use
`expectation_mpo_exact` as shown below. The general `expectation_mpo` API remains
available for an explicitly approximate structured-MPO application: it uses a
private transformed copy of the TTN. Routing itself is lossless, but the final
subtree sweep may compress that copy when an MPO increases a bond beyond
`max_bond`; `cutoff=0.0` does not disable a finite bond cap. The default
`warn_on_truncation=True` reports this approximation, and the diagnostic form
makes it easy to check a benchmark:

```python
energy, report = opt.tn.expectation_mpo(
    mpo,
    range(plan.n),
    max_bond=64,
    cutoff=0.0,
    return_diagnostics=True,
)
assert not report["truncated"]
```

Use a larger `max_bond` when an untruncated measurement is required. Native
fermionic trees also reject ordinary dense MPOs (and dense trees reject native
Symmray MPOs) instead of silently changing the fermionic interpretation.

For an exact readout that does not move the MPO into the tree at all, use
`expectation_mpo_exact`:

```python
energy = opt.tn.expectation_mpo_exact(
    mpo,
    range(plan.n),
)
```

The same method is available directly as `opt.expectation_mpo_exact(...)`.

This keeps the bra, ket, and structured MPO as separate networks. Fresh ket
physical indices connect to the MPO input legs, the MPO output legs connect to
the bra, and Quimb contracts the complete doubled network. No state-bond
compression or `to_dense()` lowering occurs. Native Symmray MPOs retain their
graded contraction and fermionic sign rules.

The optimizer-level `opt.expectation_mpo(...)` retains the configured update
algorithm for TreeMPO input: direct/DM, SRC/SDC/SDCR, zipup, or FIT. It preserves
the parent state and sampling RNG, including when a private copy fails.
Private readout starts with fresh histories and does not copy the queued gate
stream. Public `copy()` retains independent histories and the queued stream.
Measurement warnings and returned edge records remain available with
`record_history=False`; enabling them does not enable expensive singular
spectrum probes. Multi-node FIT also warns because finite variational sweeps
can be approximate without producing edge-cut records. Its report includes
`fit_diagnostics` and `approximation_possible`; `truncated=False` alone does
not certify exact FIT. Single-node exact FIT is exempt from this warning.
Use `warn_on_truncation=False` to opt out of approximation warnings.
Explicit chain-MPO input retains the lower-level chain routing and final
subtree compression even when the optimizer is configured for DMRG.

Every TreeMPO application mode preserves `operator.exponent`, adding it to the
state's represented exponent without forming `10**exponent`. Copies,
conjugation, exact expectation, dense readout, scalar multiplication, addition,
and composition also preserve this scale. Stored sector networks retain their
relative exponents; the public operator exponent supplies their common offset.

`TreeOptimizer.to_dense()` / `TreeTensorNetwork.to_statevector()` return numeric
host vectors for native Symmray trees too. Readout contracts separate physical
legs and restores the declared `physical_sectors` mapping before densifying,
so empty charge sectors do not disappear from the output basis after gauge
moves. Within each local dimension, charges use Symmray's sorted sector order.

`TreePlan.mpo_order()` remains available as the structural leaf-position order
chosen by the plan; a physical `root_qubit`, when present, is placed first.
It is useful when a caller deliberately chooses a one-dimensional model-level
MPO ordering, but Tree builders do not construct a chain MPO from it.

`TreeMPO` is the tree-routed operator class. `tree_networks` contains the
TreePlan-labelled operator networks used by `expectation`; no second chain
representation is stored on the object. Native neutral
terms are factorized directly from their native Symmray operator tensor over
the term's TreePlan Steiner subtree, then amalgamated into one charge-aware
direct-sum TTNO. This applies to one-, two-, and higher-site native terms; it
does not create a hyperedge for the normal Hamiltonian path. The resulting
TTNO can be canonicalized and compressed with
`tree_operator.canonicalize()` and
`tree_operator.compress(cutoff=..., max_bond=...)`; no Jordan--Wigner
conversion is used. Nonzero or mixed operator charges remain separate
homogeneous native networks inside the same public `TreeMPO`, so callers do
not need `charge_sectors=True` just to construct one operator object;
`charge_sectors=True` remains available when separate objects are preferred.
Structured observables can use a smaller compact TTNO.
Pass `fermionic=False` only for dense ordinary/Jordan--Wigner-compatible terms.
Native `TreeMPO.identity()` preserves the operator's Symmray symmetry and
returns a bond-one TTNO that can be applied directly to a native
`TreeTensorNetwork`; dense identities remain ordinary dense TTNOs.
`OneDMap` is the shared source of truth for regular 2D/3D coordinate layouts;
the tree and MPS geometric layout finders consume its row/column, snake,
alternate-x/y/z, folded-snake, and generalized Hilbert traversals directly.
For a tree-native layout, the short canonical spelling is
`map_mode="coarse-alternate-x"` (or another `coarse-*` preset). It describes
the lattice coarsening/traversal used to label the leaves and is available as
`map_mode` on the resulting `TreePlan`, `TreeTensorNetwork`, and `TreeMPO`.
Pass the same plan through state, operator, and optimizer construction so all
three components share one binary-tree geometry.

`TreeMPO` subclasses Quimb's `TensorNetworkGenOperator`, in the same way that
`TreeTensorNetwork` subclasses `TensorNetworkGenVector`. It is the tree twin
of Quimb's `MatrixProductOperator`: the common operator surface includes
`sites`, `nsites`, `site_tag`, `upper_ind`, `lower_ind`, `to_dense`, `H`,
`copy`, `identity`, `from_dense`, `add_TreeMPO`, `singular_values`, and
`amplitude`. Tree-specific geometry is exposed through `plan`, `node_tensor`,
`neighbors`, and `bond`; `canonicalize`/`compress` perform the corresponding
tree-wide QR/SVD sweeps. It cannot inherit Quimb's chain-only
`MatrixProductOperator` implementation because a branched tree has no single
left/right ordering. A chain MPO for a chain workflow is constructed
separately with the model-level `to_mpo(...)`; it is not stored on `TreeMPO`.
`validate()`
checks every stored TTNO network against the
TreePlan; `validate(check_canonical=True)` also checks the tracked operator
`left_inds` directions when a canonical region is known.

Dense `TreeMPO` builder outputs run the same exact structural boundary sweep
over the rooted tree before the native tree SVD. It reduces proportional and
roundoff-safe linearly dependent edge channels in a leaf-to-root and reverse
pass, so direct-sum term assembly does not carry redundant states into every
parent tensor. Native Symmray/fermionic networks are left on their existing
charge-preserving QR/SVD route.

`ham_tn.to_tree_mpo(...)` defaults to `compress="term"`, so it adds and
compresses one term at a time. Pass `compress=True` or `compress="auto"` for
the workload-aware route, or `compress="automaton"` to force full native
assembly. Pass `progbar=True` for the same MPS-style construction bar. Term
mode advances once per added term and reports the current `chi` against the
requested cap; the progress-bar default is `False`.

`add_TreeMPO(..., compress=True)` first builds the tree direct sum and then runs
the same native tree SVD sweep; its compression options are `max_bond`,
`cutoff`, and `order`. The default `order="rank"` is a deterministic greedy
leaf-elimination policy that uses live edge dimensions to reduce small-rank
branches before they enlarge parent tensors. Use `order="depth"` for the
simple fixed depth-first sweep when reproducing older benchmarks. This is
rank-aware ordering on a fixed TreePlan, not a global search over alternate
tree geometries. Native graded Symmray TTNOs safely retain the charge-preserving
depth order and report that effective fallback, because arbitrary sibling
reordering requires an explicit graded permutation proof. Ordinary
`copy(transpose=True)` and `conj()` views preserve the
canonical gauge metadata when they keep the TreePlan index layout unchanged.
For native Symmray operators, addition uses a charge-aware TreePlan direct
sum; it does not use Quimb's dense-axis padding. A chain MPO, when needed, is
constructed separately with the model-level `to_mpo(...)`.
`add_MPO(...)` remains a compatibility alias for `add_TreeMPO(...)`.

`TreeMPO.ascii_tree()` returns a compact Quimb-inspired native drawing, and
`TreeMPO.show()` prints only this clean native tree by default, with one bond
dimension per branch. When a
`TreeLayoutFinder` with `lattice_shape=` is attached, the operator also keeps
the physical coordinate map and term supports for display:

```python
print(tree_operator.ascii_tree())
tree_operator.show(bond_dims=True)
tree_operator.show(layout="both")  # physical lattice above native tree
```

Hamiltonian builders attach this layout metadata automatically when their
lattice shape is known. For a manually constructed operator, pass the finder
explicitly:

```python
finder = TreeLayoutFinder(
    supports=terms,
    n=16,
    lattice_shape=(4, 4),
)
tree_operator = TreeMPO.from_terms(
    plan,
    terms,
    layout_finder=finder,
)
tree_operator.show(layout="lattice")
```

`layout="tree"` (the default) keeps the output to the native ASCII tree.
`layout="auto"` remains an explicit convenience option that shows both
sections when this metadata is available. `ascii_lattice()` shows
the physical site array and a compact support list; `plot_layout()` or
`show(layout="plot")` gives the Matplotlib tent view with the physical lattice,
tree hierarchy, and term connectivity.

Dense `TreeMPO` objects also provide exact `+`, `-`, scalar multiplication,
and operator composition with `@` while retaining the native TreePlan
network. Composition does not lower either operand to a chain MPO; use
`compress(...)` explicitly to truncate the resulting tree bonds. Composition
of native graded fermionic operators is intentionally guarded until a graded
fused-bond kernel is available.

`canonicalize(center=..., info_c=...)` performs lossless tree QR
canonicalization. The tree-native source of truth is `canonical_region` plus
each tensor's `left_inds`; when `info_c` is supplied, Pepsy mirrors the
single-node center as `info_c["cur_orthog"] = (center, center)` and stores the
connected region in `info_c["canonical_region"]`. It also records
`info_c["isometry_map"]` and immutable per-network `info_c["left_inds"]`
snapshots for diagnostics and optimizer synchronization.

`TreeMPO.from_terms(plan, terms)` accepts dense one-site, two-site, and
higher-order terms. A higher-order term is first factorized exactly on the
minimal Steiner subtree joining its physical sites, then combined with the
other terms by a TTNO virtual direct sum. Thus every resulting network still
has one operator tensor per `TreePlan` node; it is not a hyperedge that must be
lowered to a chain. The same Tree-native factorization is used by the dense
term path and by the native Hamiltonian builders (with graded Symmray tensors
on the fermionic path).

When the term mapping includes a higher-order term, dense factorization and
direct-sum assembly preserve the input array backend and device. All terms
must share that backend and device; mixed inputs raise `ValueError`. An
explicit `dtype=` casts on that backend. Without it, assembly retains the
first term's dtype, as before. The separate compact one-/two-site Hamiltonian
automaton remains a host construction path. Upstream SVD rank selection can
still read scalar decisions during factorization.

Ordinary device norm bookkeeping uses cached Autoray namespaces when
available, with a dispatch fallback for older Autoray. This reduces Python
dispatch overhead without changing the existing diagnostic readout boundaries.

For a local gate, use `SubTreeMPO.from_gate(plan, gate, where)`. This compact
operator stores tensors **only on the connected Steiner subtree**. It retains
the original logical site labels, node IDs, and configurable tags. Connecting
internal nodes are included; a physical routing node not acted on by the gate
has a local identity pair. There are **no exterior operator tensors or dangling
exterior operator bonds**. Exterior identity action is implicit.

```python
from pepsy import SubTreeMPO

subop = SubTreeMPO.from_gate(plan, gate, where)
assert subop.num_tensors == len(subop.active_nodes)
optimizer.apply_sub_mpotree(subop)
# Equivalent stream entry:
event = optimizer.sub_mpotree_event(subop)
```

`subop.sites` lists physical sites present in the subtree, including any
physical routing node; `operator_support` names the actual gate support.
`to_dense()` returns an operator on `subop.sites`, not on every state site.
Copies, backend conversion, canonicalization, and compression preserve the
compact region. Ordinary gate replay builds this representation directly;
it must not construct a full identity-padded operator and strip it afterward.
Dense factorization removes only machine-precision null operator-Schmidt
sectors; configured operator compression remains explicit.

`TreeMPO.from_gate` remains the full-tree constructor for callers that need
a complete operator for general operator algebra. It includes exterior
bond-one identities and is not the ordinary local gate replay builder.
`TreeMPO.from_pauli_sum(plan, weighted_terms)` provides a full-tree TTNO for a
weighted sum of product-Pauli branches. It uses one virtual branch channel per
retained term only on the union of the active Steiner subtrees, with bond-one
identity legs outside; it never constructs a full-system dense matrix or a
chain MPO. `SubTreeMPO.from_pauli_sum(plan, weighted_terms)` provides the
compact form: it omits those exterior identity tensors and makes identity
outside the union of the active Steiner subtrees implicit.

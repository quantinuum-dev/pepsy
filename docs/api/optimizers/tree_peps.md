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

The retained tree degree is capped at three virtual bonds per site. Thus a
state tensor has at most rank four: one physical leg plus at most three
virtual legs. A normal `topology="tree"` plan also requires at least one
three-virtual-bond site (a rank-four tensor), so it cannot silently degenerate
to an MPS. `state.max_virtual_degree`, `state.tensor_rank(site)`,
`state.max_rank`, `state.topology`, and `state.is_branching` expose these
diagnostics.

The physical index is intentionally present only once. The logical 1D
address is represented by an additional tag, not by a second physical leg.
Each tensor also carries a structural `N{q}` tag, making it straightforward
to select either lattice sites or tree regions with Quimb operations.

For a workload-adapted tree, score the physical lattice supports before
constructing the state. The finder returns a regular `TreePepsPlan`, so the
same result can be passed to the state, PEPO constructors, and optimizer:

```python
from pepsy.optimizers import TreePepsLayoutFinder

layout = TreePepsLayoutFinder(
    plan,
    interactions=[(dense_gate, (0, 7)), (dense_gate, (2, 5))],
    objective="hybrid",
    seed=0,
).recommend()

state = TreePeps.rand(layout, bond_dim=4, seed=7)
operator = TreePepo.from_operator(layout, dense_gate, support=(0, 7))
optimizer = TreePepsOptimizer(state, plan=layout)
```

`objective="span"` minimizes the weighted number of virtual edges in each
gate’s minimal tree span. `"load"` emphasizes peak routed edge demand, and
`"hybrid"` combines both with total edge load. The bounded refinement is
deterministic for a fixed `seed`; inspect `finder.report` for spans, edge
loads, degree, and rank diagnostics.

The finder compares the source tree with deterministic `OneDMap` seeds and
workload-weighted growth. Use `seed_modes` (or its aliases `tree_orders` and
the singular `tree_order`) to choose candidates such as `"row-major"`,
`"col-major"`, `"hilbert"`, `"inside-out"`, `"diag"`, or `"snake"`. Supplying
`root="center"` selects the geometric center for a finder constructed from a
shape. The selected seed, candidate count, and seed modes are recorded in
`finder.report`.

For 2D states, `.show()` follows Quimb’s PEPS schematic conventions with
Unicode lattice bonds and bond dimensions, while omitted lattice edges remain
visible as gaps because only the selected virtual tree is drawn. Three-
dimensional states use the same coordinate schematic layer-by-layer.

`TreePepsPlan.from_shape` uses a branching spanning tree by default. The
logical site order (`order`) and retained-tree growth priority (`tree_order`)
are independent. For a 2D shape `(Lx, Ly)`, `tree_order="row-major"`
selects a horizontal row-comb: every fixed-`y` row is a tooth and the
`x=0` column is the backbone. `tree_order="col-major"` selects the transpose:
every fixed-`x` column is a vertical tooth and the `y=0` row is the backbone.
These are spanning trees with maximum virtual degree three, matching the
horizontal/vertical layouts used by `.show()` and the gallery notebook.
`"snake"`, `"hilbert"`, and `"inside-out"` remain traversal-priority
branching growth modes rather than Hamiltonian paths; `"inside-out"` starts
at the geometric center and grows toward the boundary. Aliases include
`"center-out"` and `"outward"`.

One-dimensional and geometrically non-branching lattices must opt into their
MPS-compatible path topology explicitly:

```python
path_plan = TreePepsPlan.from_shape((1, 16), topology="path")
assert path_plan.is_mps_topology
```

```python
plan = TreePepsPlan.from_shape(
    (4, 4), order="row-major", tree_order="inside-out"
)
plan.coordinate(plan.root)  # (1, 1)
```

Custom tree edges can be supplied as logical-id pairs or coordinate pairs,
for example:

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
    topology="path",
)
```

This particular custom edge list is a Hamiltonian path, so it is marked
explicitly as `topology="path"`. A custom branching edge list can keep the
default `topology="tree"`.

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

For an explicit path-method selection, the same call can retain both layers
until Quimb compresses them:

```python
updated = gate.apply_to(
    state,
    compress=True,
    compression_mode="zipup",
    compression_layout="two_layer",
    max_bond=8,
)
```

`TreePepo` is a generic tree operator with separate input/output physical
legs. `TreeSubPepo` records the physical support and its connected tree span.
The default `compression_layout="auto"` preserves the fused operator/state
application for ordinary and branching updates, while path updates using
Quimb's multi-tensor methods can retain the separate operator and state
layers until compression. Use `compression_layout="fused"` to force the
original fused path, or `"two_layer"` to require the path-only MPO-MPS-style
path. The full design and the future
`TreePepsStabOptimizer` interface are documented in the development plan.

`TreePepsOptimizer` owns a state copy by default and supports the two update
paths:

```python
from pepsy.optimizers import TreePepsOptimizer

direct = TreePepsOptimizer(state, mode="direct", chi=16)
direct.apply_gate(dense_gate, where=(0, 5))

subtree = TreePepsOptimizer(state, mode="sub_treepepo", chi=16)
subtree.apply(subop)
```

Direct gates are factorized over the unique tree path between their sites.
`TreeSubPepo` updates fuse the complete connected span before one localized
leaf-to-center compression sweep. Both paths keep intermediate routing
lossless and use `left_inds`-aware canonical movement.

The optimizer also owns a persistent, replayable stream. Install or extend
it without executing the state, then call `run()` when ready:

```python
streamed = TreePepsOptimizer(state, run=False, chi=16)
streamed.set_gates([
    (dense_gate, (0, 5)),
    TreePepsOptimizer.sub_treepepo_event(subop),
])
streamed.add_gates([
    TreePepsOptimizer.gate_event(dense_gate, 2),
])
streamed.run()
```

The accepted event forms are `(gate, where)`, tagged
`("gate", gate, where)`, a `TreePepo`, or a `TreeSubPepo`; mapping forms with
`kind`, `gate`/`where`, or `operator` keys are also accepted. `run()` without
arguments replays the currently queued stream, while `run(gates)` preserves
the older one-shot spelling by replacing the queue first. The normalized
stream is available as `gate_stream`. Convenience methods `apply_1q`,
`apply_2q`, `apply_multi_site`, and `apply_pepo` match the corresponding
optimizer vocabulary.

Use `set_state(new_state)` (or assign `optimizer.tn`) to replace the live
state. The new state must have the same tree plan, and all queued operator
payloads must match its backend, device, and required dtype contract before
the replacement is installed. By default the replacement is copied; use
`inplace=True` at construction when state identity should be retained.
`backend_info()` reports the live state metadata.

Compression is selected independently from the operator route with
`compression_mode="direct"` (the default SVD decomposition) or
`compression_mode="dm"` (Quimb's density-matrix-equivalent `svd:eig`
decomposition of the local fused compression core). `compression_mode="zipup"`
is also available for path operator-state compression. In fused mode the
state is canonicalized around the active span first, the PEPO is fused locally
with the state, and only then are the combined tree bonds truncated. In
two-layer mode, the state and PEPO tensors are grouped by the same site tags
and passed to Quimb's 1D compressor as an MPO-MPS-like network. No global
dense lattice state is formed. For convenience, `mode="dm"` is accepted as a
shorthand for direct TreePepo routing with `compression_mode="dm"`.

`mode="sdc"`, `mode="src"`, and `mode="zipup"` are also accepted shorthands
for direct TreePepo routing with the corresponding compression mode. On an
explicit path topology, `compression_layout="auto"` uses Quimb's actual 1D
SDC/SRC/ZipUp kernels with the separate operator and state layers, then
restores the TreePeps plan, tags, exponent, and canonical metadata. On a
branching topology, `sdc` uses the tree's deterministic successive edge sweep
and `src` uses randomized SVD per edge; neither silently invokes a chain-only
environment algorithm, while `zipup` is path-only. Truncating path `sdc`/`src`
requires finite `chi`/`max_bond` (`sdc` with zero cutoff may still be used as a
lossless canonicalization), and `compression_seed=...` makes randomized
results reproducible. The paper's full projected Cholesky (CBC) tree
compressor is not represented by these aliases and remains a separate future
method.

### TreePEPS FIT / DMRG

`TreePepsOptimizer` also exposes the tree-native `TreeFIT` engine through
`mode="dmrg"` and the `dmrg1`/`dmrg2`/`dmrg3` aliases. The exact fused
operator-state target is built on a disposable copy, then fitted on the
active connected tree span with cached directed branch environments:

```python
optimizer = TreePepsOptimizer(
    state,
    mode="dmrg2",
    chi=16,
    fit_n_iter=3,
    fit_init_strategy="guess-src",
)
optimizer.apply_gate(gate, where=(0, 5))
report = optimizer.get_fit_diagnostics()
```

Generic `dmrg` uses `fit_block_size=2` and its configured adaptive warm-up.
`dmrg1` and `dmrg2` use two-node warm-up blocks, while `dmrg3` uses
three-node warm-up blocks; all named modes then refine with one-node sweeps.
The remaining controls are `fit_adaptive_sweeps`, `fit_min_iter`, `fit_rtol`,
`fit_patience`, `fit_sweep_sequence`,
`fit_init_rand_strength`, and `fit_init_seed`. Initial guesses may be
`"direct"`, `"guess-src"`, `"guess-sdc"`, `"guess-dm"`, `"random"`, or
`"random_expand"`; random policies are disposable, seeded, and active-span
only. `get_fit_diagnostics()` reports the cache hit/miss counts, block size,
convergence, and optional normalized target overlap when
`fit_overlap_diagnostics=True`.

This DMRG path currently builds the exact TreePEPO/TreeMPO target as a fused
TreeFIT target. TreeFIT also accepts a correctly tagged layered target when
each layer tensor belongs to exactly one structural node group; local layer
bonds remain inside that group and inter-group bonds must follow the tree.
The separate two-layer operator-state path remains available for the direct
`sdc`/`src`/`zipup` compressors above when that Quimb path is desired.

## TreeTensorNetwork API parity

The dense TreePeps state and optimizer expose the reusable, geometry-neutral
parts of the existing `TreeTensorNetwork`/`TreeOptimizer` surface. The state
provides `nsites`/`nqubits`, `root`, `top_arity`, `is_binary`, rooted
`parent`/`children`/`is_leaf` helpers, `tree_distance` and `subtree_span`,
`max_bond`/`bond_sizes`/`bond_report`, batched `local_expectations`,
`to_statevector`, and in-place `normalize`. The optimizer provides the `p` and
`tn` state aliases, `center`/`orthogonality_center`, center movement and
`left_inds` validation delegates, `show`, `to_dense`, `norm`/`normalize`,
`bond_report`, conservative `estimate_bonds`/`preflight`, and a
`truncation_report` over its replay history. Logical site order is fixed by
the plan, so the optimizer's `qubits`, `logical_order`, `position`, and
`remap_sample` helpers are identity mappings.

The optimizer-level `canonicalize`/`canonize`, `canonize_subtree`,
`canonize_around_qubits`, and `compress` methods delegate to the live state and
refresh `info_c` after every center or region change. `compress(sites, span=True)`
uses the same minimal-span, leaf-to-center sweep as a gate update; its history
record reports `span`, `touched_edges`, `uncompressed_bonds`, and
`compression_scope="span"`, so exterior tree bonds are not silently included.
`run` also supports `non_unitary`, `normalize_every`, `normalize_final`, and
`track_infidelity` controls. `profile=True` enables update-envelope timings,
`track_bond_diagnostics=True` records transient versus live bond pressure, and
`max_intermediate_bond` can reject a queued stream during preflight before
replay begins. `TreePepsOptimizer.find_tree_layout(...)` and
`convergence_sweep(...)` provide the corresponding layout and bond-cap
convenience entry points.

Pass `progbar=True` to `run()` for a replay bar matching the MPS optimizer's
compression readout. It reports the latest local fidelity as `F`, the
log-accumulated retained fidelity as `~F`, the live maximum bond as `bnd`, and
event counts such as `2q`; it does not display the live state norm. `F` and
`~F` are retained-norm compression proxies, not directional overlaps with a
target state. They are available after replay as
`norm_diagnostics()["local_fidelity"]` and
`norm_diagnostics()["cumulative_fidelity"]`, with matching infidelity fields.

As with the state, TreePeps truncation history records exact bond dimensions;
it does not claim a scalar discarded-weight fidelity unless a caller performs
an explicit reference comparison (the convergence sweep does so for small
enough dense states).

The intentionally absent TTN-only paths are qubit measurement/reset/capping,
`TreeMPO` expectation/application, and native Symmray/fermionic support.
TreePeps sites can have general physical dimensions and its lattice plan is
fixed during compression; these operations need dedicated physical-space and
topology contracts rather than a compatibility alias. Stabilizer replay and
structured TreePePO backends remain separate roadmap phases.

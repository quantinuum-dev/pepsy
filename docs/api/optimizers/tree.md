# `pepsy.optimizers.tree`

Use these guides for tree tensor networks, circuit simulation, and readout.

| Task | Guide |
| --- | --- |
| Choose and refine a tree plan | [Layout](tree_layout.md) |
| Build states and manage canonical form | [States](tree_state.md) |
| Build full-tree and compact operators | [Operators](tree_operators.md) |
| Apply gates and configure compression | [Circuit replay](tree_replay.md) |
| Configure TreeFIT and DMRG | [Variational fitting](tree_fit.md) |
| Measure states and inspect accuracy or timing | [Readout and diagnostics](tree_readout.md) |

`TreeOptimizer` simulates a quantum circuit by replaying a canonical bundled
gate stream `[(gate, where), ...]` on a **rooted tree tensor network**, after
*Simulating quantum circuits using tree tensor networks* (Seitz, Medina, Cruz,
Huang, Mendl; Quantum 7, 964, 2023; [arXiv:2206.01000](https://arxiv.org/abs/2206.01000)).

By default the state is stored with one leaf tensor per qubit. A plan may
instead designate one `root_qubit`, placing that physical index directly on
the top tensor while every other qubit remains a leaf. Internal nodes may have
**any arity** -- the default is a binary tree below a three-virtual-leg root,
but a fixed binary root, flatter `k`-ary trees, or gate-connectivity-driven
communities (see [tree layout](tree_layout.md#tree-structure)) all work through
the same machinery.

For example, this constructs a binary detector tree whose logical qubit is the
open top index. The root tensor has two virtual child bonds and physical index
`k4`:

```python
from pepsy.optimizers.tree import TreeLayoutFinder, TreeOptimizer

finder = TreeLayoutFinder(gates, n=5, root_qubit=4, max_arity=2)
plan = finder.run(
    order="quality",
    refine="greedy",
    refine_budget=64,
    search="nevergrad",
    search_budget=128,
    progbar=True,
)
opt = TreeOptimizer(
    gates,
    tree=plan,
    chi=64,
    cutoff="auto",
    cutoff_mode="auto",
)
assert opt.plan.node_of_qubit[4] == opt.plan.root
assert set(opt.tn.node_tensor(opt.plan.root).inds) >= {"k4"}
```

`root_qubit` is first-class rather than an unregistered outer leg:
`to_dense()` retains it in normal qubit order, `cap(root_qubit, vec)` contracts
only that physical leg, and direct gates, dense subtree operators, and
structured sub-MPOs may include it in their support. `TreeLayoutFinder` keeps
the site fixed at the root while its path, Steiner, congestion, greedy, and
Nevergrad objectives permute only the remaining leaf sites.

## Section links

Existing section links are kept below; each points to its full guide.

## Tree-edge entanglement entropy

See [Tree-edge entanglement entropy](tree_readout.md#tree-edge-entanglement-entropy).

## Layout-aware native MPOs

See [Layout-aware native MPOs](tree_operators.md#layout-aware-native-mpos).

## Native fermionic QR stability

See [Native fermionic QR stability](tree_state.md#native-fermionic-qr-stability).

## Native central-edge compression and profiling

See [Native central-edge compression and profiling](tree_readout.md#native-central-edge-compression-and-profiling).

## QR/hop and bond-growth diagnostics

See [QR/hop and bond-growth diagnostics](tree_readout.md#qrhop-and-bond-growth-diagnostics).

## Range / subtree canonicalisation

See [Range / subtree canonicalisation](tree_state.md#range--subtree-canonicalisation).

## Multi-qubit / sub-MPO application

See [Multi-qubit / sub-MPO application](tree_replay.md#multi-qubit--sub-mpo-application).

## Tree-native FIT / DMRG

See [Tree-native FIT / DMRG](tree_fit.md#tree-native-fit--dmrg).

## Performance-oriented defaults and warnings

See [Performance-oriented defaults and warnings](tree_replay.md#performance-oriented-defaults-and-warnings).

## Coefficient-backend feature boundary

See [Coefficient-backend feature boundary](tree_readout.md#coefficient-backend-feature-boundary).

## Tree state class

See [Tree state class](tree_state.md#tree-state-class).

## Tree structure

See [Tree structure](tree_layout.md#tree-structure).

## Fixed-plan refinement and Nevergrad search

See [Fixed-plan refinement and Nevergrad search](tree_layout.md#fixed-plan-refinement-and-nevergrad-search).

## Initial-state layout handoff and array backends

See [Initial-state layout handoff and array backends](tree_layout.md#initial-state-layout-handoff-and-array-backends).

## Diagnostics

See [Diagnostics](tree_readout.md#diagnostics).

## Readout

See [Readout](tree_readout.md#readout).

## Performance and stability

See [Performance and stability](tree_replay.md#performance-and-stability).

```{toctree}
:hidden:

tree_layout
tree_state
tree_operators
tree_replay
tree_fit
tree_readout
```

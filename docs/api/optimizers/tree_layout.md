# Tree layout

[Tree API overview](tree.md)

Choose a tree plan, refine its geometry, and pass it to the state and optimizer.

## Root arity

The conventional binary TTN with a three-leg top tensor is the default when
there are at least three leaves and no `root_qubit`. Pass
`max_arity=2, top_arity=3` explicitly to `TreePlan.from_order`,
`TreeLayoutFinder`, or `TreeTensorNetwork.from_order` for the same geometry.
The structural root then has three **virtual**
child bonds and no parent bond; every non-root internal tensor has two child
bonds and one parent bond. Thus the root is still in the rank-three binary
class, rather than being a genuinely wider tensor. `top_arity=3` is not
combined with `root_qubit`, because adding a physical root leg would make a
rank-four tensor. `TreePlan.is_binary()` accepts this ternary-root convention,
while `TreePlan.is_strictly_binary()` requests two children at every internal
node.

## Tree structure

The tree structure is chosen by `TreeLayoutFinder`, which builds a weighted
interaction graph from the two-qubit supports of the gate stream and applies
recursive spectral (Fiedler) partition, keeping the recursion as the rooted
tree (`structure="quality"`). This reuses the interaction-graph and spectral
machinery of `pepsy.optimizers.mps.layout`; where the MPS finder flattens the
recursion into a 1D order, the tree finder keeps the tree. Strongly coupled
qubits become nearby physical nodes, minimising the tree-path length that
two-qubit gates thread across. With `root_qubit=q`, that physical node stays
fixed at the top while the finder searches over the remaining leaf sites.
`structure="balanced"` splits the leaf-qubit order in half at each level.
`TreeLayoutFinder.score(plan)` returns the total interaction-weighted tree-path
length that the structure minimises.

For circuits with gates of different operator-Schmidt ranks, use
`TreeLayoutFinder(..., objective="congestion")` or
`TreeOptimizer(..., layout_objective="congestion")`. This evaluates interaction,
congestion-aware, and balanced candidates using the predicted log bond growth
on every edge. A gate crossing an edge contributes `log2(k)`, where `k` is its
operator-Schmidt rank across that edge; the maximum edge load therefore
predicts the worst-case multiplicative bond growth. `TreeOptimizer` uses
`layout_objective="congestion"` by default because it is a better
execution-oriented choice at finite `chi`; a bare `TreeLayoutFinder` retains
`objective="path"` as its fast, backward-compatible default. The path
objective remains the co-occurrence/path-length heuristic.
`objective="hybrid"` is useful when both replay cost and bond pressure matter:
it combines normalized path score, maximum edge load, and total edge load with
`hybrid_weights=(path, max_edge_load, total_edge_load)`. The
`weight_mode` / `layout_weight_mode` option accepts `count`, `auto`, `angle`, or
`operator_schmidt` for interaction-graph weighting.

For a genuinely multi-site layout objective, use
`objective="hypergraph"` (or `layout_objective="hypergraph"`). Each original
gate support is kept as one hyperedge, and the finder scores its actual
operator-Schmidt load on every crossed tree edge rather than selecting from a
pairwise proxy alone. This mode starts from inexpensive pairwise-derived seed
trees, then automatically performs bounded direct greedy leaf swaps and binary
NNI topology moves using the full hyperedge score. Pass
`refine=None, topology_refine=None` to inspect the unrefined direct score, or
set explicit budgets for a larger search. Dense operators wider than
`max_operator_qubits` still use the documented conservative rank bound.

For a whole-tree optimization, use `objective="full_tree"` (also accepted as
`"tree"` or `"cotengra"`). This evaluates dynamic operator-Schmidt demand,
cap overflow, working tensor width, estimated work/write volume, and route
length across every hierarchical scale, not only the root cut. Finite-`chi`
overflow and edge demand are ranked before tensor-work proxies, since avoiding
unnecessary truncation is the primary execution concern. It enables bounded
subtree reconfiguration and simulated annealing by default; override these
with `topology_refine="subtree"`, `topology_budget=`, `search="anneal"`, and
`search_budget=`. The result is still a cheap layout proxy rather than a real
TTN replay, so the state-aware pilot remains the final accuracy check. The
default `chi=None` leaves this as a static, chi-blind objective; supplying
`chi` only adds cap-aware ranking and does not change the no-tensor nature of
layout discovery.

For the 6×6 periodic square-lattice calibration stream (Hadamards followed by
periodic controlled-phase gates), predicted total overflow ranked binary,
ternary, and four-way candidates in the same order as actual capped replay
pressure and truncation counts. This validates the profile as a layout-ranking
proxy; use the state-aware pilot when the circuit has strong cancellations or
state-dependent rank loss.

Use `order="quality"` with `finder.run()` (or set it on the finder) for the
MPS-style high-quality offline search. Quality mode now means
`objective="full_tree"`: it evaluates every hierarchy scale, enables bounded
greedy leaf refinement and all-scale subtree topology refinement, and runs a
hybrid topology-annealing/Nevergrad search when Nevergrad is available. It
falls back explicitly to dependency-free simulated annealing otherwise. A
finder without `order="quality"` keeps the fast deterministic zero-argument
`run()` path.
Disable stages explicitly with `refine=None`, `topology_refine=None`, or
`search=None`, or bound them with `refine_budget=`, `topology_budget=`, and
`search_budget=`.

For a stream whose locality changes over time, pass `time_decay=` and/or
`time_window=` to `TreeLayoutFinder`. A decay in `(0, 1]` weights an event by
`time_decay ** age` (the newest event has age zero), while a window keeps only
the final events. The same factors are used for interaction paths, congestion
candidate construction, and per-edge operator-Schmidt load estimates, so the
diagnostics and selected plan use one consistent time model. The defaults are
unchanged. `TreeOptimizer` exposes these as `layout_time_decay=` and
`layout_time_window=`.

For compression-first selection, use `objective="compression"` (or
`layout_objective="compression"`). It prioritizes peak and total predicted
operator-Schmidt load, then penalizes the estimated local tensor size at the
configured `chi`, and only then uses path length. This differs from the
default fast `"path"` objective: path optimizes routing locality, while
compression also accounts for bond pressure and wider-node cost.

`weight_mode="operator_schmidt"` is a cheap **two-qubit entangling-strength
proxy** used to form the spectral qubit order; it is not itself the exact
operator-Schmidt rank. Use `objective="congestion"` when selecting a tree: its
edge-load calculation uses the actual rank across each candidate tree cut (or
an MPO bond bound), which is the quantity that predicts TTN bond growth.

Rank diagnostics are explicit. Small dense qubit operators use exact
operator-Schmidt ranks; opaque native arrays, MPO bond products, and supports
larger than `max_operator_qubits` use conservative operator-space bounds and
are counted in `rank_bounded_events` with a reason in `rank_bound_reasons`.
They are not silently assigned rank two.

The structure is **not restricted to binary trees**. Internal nodes may have
any arity, controlled by two knobs on `TreeLayoutFinder` / `TreePlan.from_order`
/ `TreeOptimizer`:

- `max_arity` caps the children per internal node. It accepts a scalar (a
  single fixed tree: `2` reproduces the strictly-binary tree exactly, larger
  values give flatter `k`-ary trees with shorter geodesics, `None` leaves the
  arity unbounded) or an iterable of candidate arities to **search**. The
  default `2` selects the fixed binary tree; pass an iterable such as
  `(2, 3, 4)` to search candidate arities explicitly.
- `structure="adaptive"` reads the gate-stream interaction graph and lets each
  level branch into as many children as it has strongly coupled communities
  (edges above `community_frac` times the level's strongest edge). A densely
  coupled block -- a near-clique with a present-strong-edge fraction of at
  least `star_frac` -- is collapsed into a single flat **star** node, so all
  its pairwise geodesics are length two instead of the up-to-`log2 m` of a
  bisection. Binary trees remain a valid special case (`max_arity=2`).

A caller may bypass the finder entirely by passing an explicit `TreePlan` via
`TreeOptimizer(..., tree=plan)`. `TreePlan` is exported from both `pepsy` and
`pepsy.optimizers.tree`. Build one with
`TreePlan.from_order(order, weights=..., structure=..., max_arity=..., top_arity=...)`, or -- for
a fully hand-specified arbitrary-arity tree -- with
`TreePlan.from_children(children, qubit_of_leaf)`, which validates that the
children map and leaf assignment describe a single rooted tree covering qubits
`0..n-1` exactly once. Set `top_arity=3` with `max_arity=2` for the
three-virtual-bond root convention described above. `TreePlan.max_arity()` and
`TreePlan.is_binary()` report the shape; `TreePlan.is_strictly_binary()` is the
strict two-child-at-every-internal-node predicate.

`TreeLayoutFinder` also provides the same regular-lattice baseline vocabulary
as `OneDMap`. Pass `lattice_shape=(Lx, Ly)` or
`lattice_shape=(Lx, Ly, Lz)` once, then use named `order` presets for exact
tree coarsenings of the corresponding leaf traversal:

```python
finder = TreeLayoutFinder(
    gates,
    n=36,
    lattice_shape=(6, 6),
    max_arity=2,
    top_arity=3,
)

tree_row = finder.run(order="row-major")
tree_snake = finder.run(order="snake")
tree_folded = finder.run(order="folded-snake")
tree_hilbert = finder.run(order="hilbert")
tree_coarse = finder.run(order="coarse-alternate-x")
tree_quality = finder.run(order="quality")

# Equivalent one-string spelling for the tree geometry:
tree_coarse = TreeLayoutFinder(
    gates,
    n=36,
    lattice_shape=(6, 6),
    map_mode="coarse-alternate-x",
).run()
```

The supported 2D geometric presets include `"row-major"`, `"snake"`,
`"alternate-x"`, `"alternate-y"`, `"folded-snake"`, and `"hilbert"`, plus
their `coarse-*` variants. In 3D, use `"row-major"`, `"col-major"`,
`"snake"`, `"snake-row-major"`, `"alternate-x"`, `"alternate-y"`, or
`"alternate-z"`, together with their supported `coarse-*` variants.
`alternate-x` and `alternate-y` snake within each xy layer and reverse the
layer direction along z; `alternate-z` makes z the alternating inner line.
The 3D folded-snake and Hilbert presets remain intentionally 2D-only because
`OneDMap` does not define a 3D version of those paths.

A coarse preset first partitions the lattice into blocks, follows the selected
base traversal on the coarse grid, and then expands each block back to its
physical qubits. The default `coarse_grain=(2, 1)` groups two neighboring x
sites in 2D and `(2, 1, 1)` does the same in 3D; pass `(1, 2)` or `(1, 2, 1)`
for y-oriented blocks, or `(1, 1, 2)` for z-oriented blocks. Edge blocks may
be smaller. Coarse modes change only the leaf traversal order and never merge
physical tensors. Preset orders use a balanced recursive tree so the leaf
sequence and every higher tree layer are preserved as contiguous intervals of
that traversal; `"quality"` remains the independent interaction-aware
TreeLayoutFinder search. The default logical label is
`x * Ly + y` in 2D and `x * Ly * Lz + y * Lz + z` in 3D; pass a
`lattice_site` callable when the gate stream uses a different convention.

For example, a 3D Tree layout can use the same handoff through
`TreeOptimizer`:

```python
finder3d = TreeLayoutFinder(
    gates,
    n=4 * 4 * 3,
    lattice_shape=(4, 4, 3),
    coarse_grain=(2, 2, 1),
)
tree3d = finder3d.run(order="coarse-alternate-z")
opt3d = TreeOptimizer(gates, tree=tree3d, chi=32)
```

The lower-level helper is useful when constructing a `TreePlan` directly:

```python
from pepsy.optimizers.tree import TreeLayoutFinder, TreePlan

zigzag = TreeLayoutFinder.lattice_order(6, 6, "coarse-alternate-x")
tree_plan = TreePlan.from_order(
    zigzag,
    structure="balanced",
    max_arity=2,
    top_arity=3,
    map_mode="coarse-alternate-x",
)

zigzag3d = TreeLayoutFinder.lattice_order(
    4, 4, 3, "coarse-alternate-z", grain=(1, 1, 2)
)
tree_plan3d = TreePlan.from_order(zigzag3d, structure="balanced")
```

Both layout finders still accept an explicit site permutation as `order` for a
custom fixed baseline. For example, this builds the same binary/ternary-root
geometry using a square-lattice snake order without refinement:

```python
zigzag = py.square_lattice_zigzag(6, 6)
tree_plan = TreeLayoutFinder(
    gates,
    n=36,
    max_arity=2,
    top_arity=3,
    lattice_shape=(6, 6),
    coarse_grain=(2, 1),
).run(order="coarse-alternate-x")
```

The explicit order must cover every site exactly once and cannot be combined
with iterable `max_arity` candidate search.

For an automatic arity search, call `finder.recommend_arities((2, 3, 4))`
explicitly. The default `TreeOptimizer(gate_stream, n=n, chi=chi)` uses the
fixed binary/ternary-root geometry; it does not allocate tensors or perform
truncations while finding the layout.
The result contains the recommended
`TreePlan` plus per-candidate path, edge-load, peak-bond-growth, and local
virtual-degree summaries. An explicit handoff looks like:

```python
finder = TreeLayoutFinder(gate_stream, n=n, objective="congestion")
choice = finder.recommend_arities((2, 3, 4))
opt = TreeOptimizer(gate_stream, tree=choice["plan"], chi=chi)
```

The `path` and `congestion` objectives are `chi`-blind cost proxies: they score
geodesic length and additive edge load, so they can favour a wider block or
arity whose widest bond induces a qubit bipartition too large to fit `chi`.
Every bond splitting `k` of the `n` qubits from the rest can carry a Schmidt
rank up to `2 ** min(k, n - k)`, so `TreePlan.max_bond_cut()` is a purely
structural accuracy ceiling: the tree can hold an arbitrary state exactly only
when `chi >= 2 ** max_bond_cut`. Pass `chi=` to `recommend_layered` or
`recommend_arities` to make the search `chi`-aware -- candidates are ranked
first by `chi_overflow` (how far the widest bond exceeds `log2(chi)`), so a
structure that stays exact at `chi` is preferred and the layout objective only
breaks ties. Each candidate then also reports `max_bond_cut`, `chi_overflow`,
and `exact_at_chi`:

```python
choice = finder.recommend_layered((2, 3, 4), chi=chi)   # prefers a chi-exact block
opt = TreeOptimizer(gate_stream, tree=choice["plan"], chi=chi)
```

For a fixed layered family, prefer an explicit recommendation to a hard-coded
block size:

```python
finder = TreeLayoutFinder(
    gates,
    n=L,
    objective="congestion",
    weight_mode="operator_schmidt",
    chi=chi,
)
choice = finder.recommend_layered(block_sizes=(2, 3, 4))
tree_plan = choice["plan"]
```

`layered(block_size=4)` remains the right API when the block size is an
intentional experimental control: it spectral-orders the qubits and builds
exactly that fixed structure, but it does not score alternatives or use `chi`.
`recommend_layered()` inherits the finder's `chi` when its own `chi` argument
is omitted; pass `chi=None` explicitly for a chi-blind comparison. Inspect its
per-candidate `max_edge_load`, `peak_bond_growth`, `max_bond_cut`, and
`chi_overflow` rather than relying only on the chosen block size.

### Fixed-plan refinement and Nevergrad search

The tree topology is fixed before `TreeOptimizer` begins replay. Moving the
canonical centre and threading a gate along a path are tensor operations, not
layout rewrites. For a stronger *pre-simulation* layout search,
`recommend_layered` and `recommend_arities` can refine each candidate through
adjacent leaf-label swaps. This preserves every parent/child edge in the plan:
only which qubit label occupies each leaf changes.

```python
finder = TreeLayoutFinder(
    gate_stream,
    n=n,
    objective="hybrid",
    hybrid_weights=(1.0, 1.0, 0.25),
    chi=chi,
)
choice = finder.recommend_layered(
    block_sizes=(2, 3, 4),
    refine="greedy",
    refine_budget=64,
)
tree_plan = choice["plan"]
```

`refine="greedy"` is deterministic and bounded; it is opt-in for the existing
fast/default objectives. The explicit `objective="hypergraph"` mode enables
greedy and NNI refinement by default because otherwise its full-support score
would only rank a few pairwise-derived seed trees. A balanced TTN turns a
well-aligned physical span `r` into a path with `O(log r)` tree hops, so the
hybrid score uses path length as a replay-cost proxy while edge loads estimate
the accuracy/bond-dimension cost.

For offline quality searches, `search="nevergrad"` starts from the
spectral/greedy plan, proposes leaf orders, and keeps its result only when it
improves the same objective. `search="anneal"` explores subtree replacements
at multiple scales without optional dependencies. For the highest-quality
`objective="full_tree"` search, use `search="hybrid"`: it splits the bounded
budget between topology annealing and Nevergrad leaf refinement. Quality mode
selects this hybrid automatically when Nevergrad is installed and falls back
to annealing otherwise. None of these stages allocates or replays a TTN.
Install the optional dependency from the checkout with
`python -m pip install ".[layout]"`:

```python
choice = finder.recommend_layered(
    block_sizes=(2, 3, 4),
    refine="greedy",
    search="nevergrad",
    search_budget=128,
    seed=0,
)
```

The same fixed-plan quality controls can be supplied directly to `run()`,
which gives finder-based frontends the same define-then-search shape as the MPS
layout API:

```python
tree_plan = finder.run(
    order="quality",
    topology_refine="nni",
    topology_budget=64,
    refine="greedy",
    refine_budget=64,
    search="hybrid",
    search_budget=128,
    seed=0,
    nevergrad_optimizer="OnePlusOne",
    progbar=True,
)
```

Omitted `run()` options inherit the finder configuration, preserving the
zero-argument API. Structure, arity candidates, objective, and event weighting
remain finder-construction options because they define the Tree search space
and scoring model rather than one refinement pass.

Nevergrad evaluates every candidate plan, so reserve it for offline circuit
studies rather than routine short simulations. Candidate records expose their
initial/final leaf order and the greedy/Nevergrad diagnostics under
`candidate["planning"]`.

When an explicit iterable of arity candidates is supplied, a `chi` biases the
search toward `chi`-exact structures. The default fixed binary/ternary-root
geometry is independent of `chi`; layout finding still does not allocate
tensors or perform truncations. A bare finder with no `chi` is likewise
static unless candidate search is explicitly requested.
Set `max_operator_qubits` to bound dense rank diagnostics and operator
allocation; wider native MPO events can still replay without dense
materialization. `TreeLayoutFinder(..., max_operator_qubits=...)` uses a
conservative rank bound above that width and reports the bounded events. The
public `score` remains the path score for compatibility; inspect
`objective_key`, `max_edge_load`, `peak_bond_growth`, and the tensor-cost
fields for compression decisions. `report(plan, include_edge_loads=False)`
skips the event-by-edge calculation for path-only diagnostics; when loads are
included, `peak_bond_growth_log2` remains finite even when the human-readable
`peak_bond_growth` would overflow floating point.

For a state-aware choice between static candidates, call:

```python
choice = opt.select_layout_for_compression(
    pilot_candidates=4,
    pilot_steps=64,
)
opt = py.TreeOptimizer(gate_stream, tree=choice["plan"], chi=chi)
```

The pilot replays candidates on independent copies with the real tree update
kernels and returns measured infidelity, final bond, truncation count, and
runtime under `choice["pilot"]`. By default one bounded `order="quality"`
candidate (greedy leaf refinement plus topology refinement; for
`full_tree`, this also includes bounded subtree/hybrid search) is
reserved a pilot slot, so it cannot be rejected before state-aware replay. Use
`include_quality=False` for the static-only candidate set. The original
optimizer is unchanged unless `install=True` is passed. Installation is
restricted to product initial states; an entangled TTN cannot generally be
relaid out exactly.

For the recommended closed-loop choice, use the high-level optimizer helper:

```python
choice = opt.optimize_layout(
    objective="full_tree",
    rounds=2,
    pilot_candidates=4,
    pilot_steps=64,
    pilot_workers=2,
    topology_budget=32,
    search_budget=64,
)
```

This keeps the finder tensor-free, pilots the selected candidates with the
actual tree replay kernels, and uses each round's per-edge truncation loss,
discarded weight, and update runtime to seed bounded NNI, subtree, and
cross-cut leaf proposals for the next round. `choice["pilot"]["rounds"]`
contains every round and `report["edge_diagnostics"]` identifies hot tree
edges. `objective="full_tree"` combines all-scale static work/bond estimates
with this short state-aware replay. `install=True` remounts the product state
on the final plan; it remains rejected for an entangled state.

Independent product-state pilots can be evaluated concurrently with
pilot_workers greater than one; the default is one for minimal overhead and
deterministic resource use. Candidate order and tie-breaking remain
deterministic.

Both helpers are also available from the package-level API:

```python
import pepsy as py

finder = py.TreeLayoutFinder(gate_stream, n=n, objective="congestion")
opt = py.TreeOptimizer(gate_stream, layout=finder, chi=chi)
```

To evolve a non-product or entangled initial state, pass it explicitly as
`state=` (or the backward-compatible `tn=`):

```python
opt = TreeOptimizer(gate_stream, layout=finder, state=initial_ttn, chi=chi)
```

`tree=` / `layout=` accept only a `TreePlan` or `TreeLayoutFinder`; passing a
`TreeTensorNetwork` there raises an error so an entangled state cannot be
silently replaced by the default `|0...0⟩` product state.

### Initial-state layout handoff and array backends

`TreeLayoutFinder` is deliberately circuit-only: it consumes gate supports and
weights to choose a plan, never an already-entangled coefficient state. Pass
the resulting finder/plan *and* a state separately to `TreeOptimizer`.
An entangled `TreeTensorNetwork` must already own that same plan. Supplying a
different `tree=` or `layout=` raises an error before any tensor is changed:
there is no generally exact, cheap relayout of an entangled TTN, and silently
compressing it would hide a fidelity loss.

Product states are the safe exception. A `TreeTensorNetwork` with
`max_bond() == 1` is rebuilt exactly on the requested plan (and emits a warning
that the requested layout replaced its old geometry). A bond-one Quimb
`MatrixProductState` is likewise accepted and mounted exactly on the selected
tree, so a caller may choose the tree layout after preparing an MPS product
state. Entangled MPS inputs are rejected rather than implicitly converted.

The live TTN has one array contract: every tensor must have the same backend,
dtype, and device. `backend_info()` reports that contract. Construct the full
initial state and every user-supplied gate/operator with the same converter.
Every user gate and every tensor in every queued sub-MPO/TreeMPO must match the
state backend and device. Non-NumPy payloads must also match dtype, while
NumPy-to-NumPy dtype promotion is compatible. A mismatch raises `TypeError` at construction,
`set_gates`, `add_gates`, or replacement-state installation, before replay or
tensor work. Prepare a payload explicitly with `opt.to_backend(payload)` (or
the same converter used to build the state). Internal Pauli/projector tensors
follow the state backend automatically.
Fixed control matrices are cached by backend/device/dtype; projectors and
Pauli-sum operators are assembled on that backend. Public
`TreeMPO.from_pauli_sum(..., like=array)` and
`SubTreeMPO.from_pauli_sum(..., like=array)` select the construction backend
and device explicitly; omitting `like` retains NumPy construction.
Dense local gate factorization also preserves its input backend. One-site
unitarity certification performs matrix work there and reads only its final
Boolean result.
Mixed-backend initial states fail immediately because there is no unambiguous
safe execution backend.

The same diagnostic is reflected by the state-derived `backend`,
`backend_dtype`, `backend_device`, and `array_backend` attributes. The
complete gate stream is checked once at its boundary, including every tensor in
each sub-MPO; replay uses the accepted payload objects without a second scan or
implicit transfer. Native Symmray states report `backend="symmray"` plus
the underlying NumPy, Torch, or CuPy `array_backend`, preserving U1/U1U1 charge
and fermionic metadata.

```python
import pepsy as py
import torch

to_backend = py.backend_torch(device="cuda", dtype=torch.complex128)
finder = py.TreeLayoutFinder(gates, n=L, weight_mode="operator_schmidt")
plan = finder.layered(block_size=4)

state = py.TreeTensorNetwork.from_plan(plan)
state.apply_to_arrays(to_backend)  # backend-only conversion preserves left_inds

# Convert user-provided gate arrays once, at their source.
native_gates = [(to_backend(gate), where) for gate, where in gates]
opt = py.TreeOptimizer(native_gates, layout=plan, state=state, chi=chi)
assert opt.backend_info()["backend"] == "torch"
```

The same rule applies to CuPy; choose `py.backend_cupy(...)` and convert every
state tensor and payload with that converter. `to_dense()` intentionally returns
a host NumPy vector for interoperability; the live state remains on its native
backend.

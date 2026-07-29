# `pepsy.optimizers.tree`

`TreeOptimizer` simulates a quantum circuit by replaying a canonical bundled
gate stream `[(gate, where), ...]` on a **rooted tree tensor network**, after
*Simulating quantum circuits using tree tensor networks* (Seitz, Medina, Cruz,
Huang, Mendl; Quantum 7, 964, 2023; [arXiv:2206.01000](https://arxiv.org/abs/2206.01000)).

By default the state is stored with one leaf tensor per qubit. A plan may
instead designate one `root_qubit`, placing that physical index directly on
the top tensor while every other qubit remains a leaf. Internal nodes may have
**any arity** -- by default the layout finder *searches* a small set of
candidate arities `(2, 3, 4)` and keeps the objective-best plan, but a fixed
binary tree, flatter `k`-ary trees, or gate-connectivity-driven communities
(see *Tree structure*) all work through the same machinery.

For example, this constructs a binary detector tree whose logical qubit is the
open top index. The root tensor has two virtual child bonds and physical index
`k4`:

```python
from pepsy.optimizers.tree import TreeLayoutFinder, TreeOptimizer

finder = TreeLayoutFinder(gates, n=5, root_qubit=4, max_arity=2)
plan = finder.run(
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
    cutoff=1e-12,
    cutoff_mode="rsum2",
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

Gates are absorbed into the tree:

- **single-qubit gates** are contracted into their site tensor with no bond
  growth; a unitary one-qubit gate preserves the tree canonical form regardless
  of where the orthogonality centre sits;
- **two-qubit gates** on sites `a` and `b` are split by SVD into two factors
  joined by a virtual bond; the factors are absorbed into the two site nodes and the
  bond is *threaded exactly* (lossless economical QR) along the tree path from
  `a` to `b`. Only once **both** factors are in place is a single canonical
  compression sweep run back along the path, truncating every touched bond to
  `chi` -- so each truncation sees the complete gate, markedly more accurate at
  finite `chi` than truncating each hop as the bond is threaded (Seitz et al.,
  Figs. 3-6).
- **operators on three or more qubits** -- a `k`-qubit gate (Toffoli, Fredkin),
  a multi-site non-unitary / Kraus operator, or a whole Trotter block -- are
  applied *in one shot* over their minimal spanning subtree by
  `apply_subtree_operator`. All open operator bonds are QR-routed to a subtree
  hub before a final canonical compression sweep truncates every touched edge
  once; `apply_gate` routes any support with `len(where) >= 3` there
  automatically (see *Multi-qubit / sub-MPO application*).
- **stream events** -- MPS-compatible `measure`, `cap`, `reset`, and
  `measure_reset` entries can be mixed into the stream. Measurements use Pauli
  eigenvalue outcomes (`+1`/`-1`) and are appended to `measurements`;
  explicit `submpo` markers use the same native QR-routing and final subtree
  compression when the payload exposes Quimb's MPO site interface, so
  `to_dense()` is not required; opaque MPO-like payloads fall back to the dense
  recursive subtree-operator path.
  `cap` contracts and removes one physical site, compacts the remaining qubit
  labels above it, and keeps the live tree canonical.

The orthogonality centre is a single node id tracked on the
`TreeTensorNetwork` itself (`orthogonality_center`), so the state -- not any one
driver -- owns the canonical form; it survives `.copy()` and is what
`TreeOptimizer.center` reads. It is moved with
`TreeTensorNetwork.shift_orthogonality_center(node)`, the tree analogue of
Quimb's MPS `shift_orthogonality_center`: the centre is walked to the target
along the unique tree geodesic with a per-edge lossless QR (Quimb
`canonize_between`), touching only the tensors on that path (an O(path length)
move, not O(N)). The move is idempotent when already centred; when the centre is
unknown it is established once with Quimb `canonize_around`. This mirrors the
`info_c["cur_orthog"]` centre tracking of `MpsOptimizer`.
`TreeTensorNetwork.is_canonical_form(center)` verifies the property directly
(every non-centre tensor is an isometry toward the centre) as a diagnostic/test
aid. `TreeOptimizer` mirrors this public surface: `TreeOptimizer.center` (with
the `orthogonality_center` name-parity alias), `shift_orthogonality_center(node)`
and `is_canonical_form(center)` delegate to the state, so the optimizer and its
`TreeTensorNetwork` speak the same canonicalisation vocabulary.

Local isometry orientation also has one owner: each live Quimb tensor carries
its proven `left_inds`, while `TreeTensorNetwork.isometry_direction(node)` and
`isometry_map()` derive read-only node-to-neighbour views from those tensors.
`can_skip_canonize(a, b)` exposes the exact dense-edge condition used to avoid
an already-proven QR, and `validate_isometry_metadata()` checks the local
orientations against the tracked canonical region. `TreeOptimizer` delegates
the same four methods without maintaining another mutable map. Native
fermionic trees retain explicit graded QR and therefore never report a
skippable edge through this API.

`TreeTensorNetwork.validate()` checks the live tensor set, physical legs, tree
edges, and bond ownership against the `TreePlan`; pass
`check_canonical=True` when the metadata alignment and more expensive numerical
isometry check are also desired.
Direct Quimb mutations such as `gate_inds_`, `canonize_between`,
`compress_between`, and `canonize_around_` invalidate the tracked canonical
region. Call `invalidate_canonical_form()` after mutating tensor data directly;
it also invalidates the native fermionic norm cache. The optimizer's
state-aware wrappers do both automatically and restore the centre only for
operations that prove canonicality is preserved.

Native fermionic trees use a separate graded edge path. Centre moves explicitly
QR-split the Symmray tensor and absorb the native carry into the next node;
edge compression explicitly forms the two-node tensor and performs its native
block SVD. Dense and nonfermionic trees continue to use Quimb's generic
`canonize_between` / `compress_between` wrappers. A graded exterior is not
assumed to be an ordinary Frobenius identity for readout: a known native
fermionic centre uses a one-tensor `TensorNetwork.H` contraction (which applies
the required outer-leg phase flips), while an unknown centre falls back to an
exact complete doubled-network contraction.

## Range / subtree canonicalisation

The single orthogonality centre generalises to a connected **canonical region**
-- the tree analogue of an MPS mixed-canonical range. `canonical_region` is a
frozenset of node ids tracked on the `TreeTensorNetwork` alongside (in fact,
underlying) `orthogonality_center`, which is simply the one-node special case:
when the region spans more than one node `orthogonality_center` honestly reads
`None`. `TreeTensorNetwork.canonize_subtree_(nodes)` gauges every tensor
*outside* a connected subtree to point inward (Quimb `canonize_around` with
`which="any"`), so the whole state norm is carried by the region tensors --
contracting just the region against its graded conjugate reproduces the squared norm,
exactly as the single centre tensor does for a one-node region. Disconnected
`nodes` raise unless `span=True` auto-expands to the minimal connected subtree
that spans them (`subtree_span`). `canonize_around_qubits_(qubits)` is the
qubit-level entry point: it canonicalises around the minimal subtree spanning
those qubits' physical nodes, so the reduced state on a set of qubits is captured by one
subtree. `is_subtree_canonical_form(nodes)` verifies the outside-is-isometric
property directly; `is_canonical_form` is its one-node case. `TreeOptimizer`
mirrors this too: `canonical_region`, `canonize_subtree(nodes, span=...)`,
`canonize_around_qubits(qubits)`, and `is_subtree_canonical_form(nodes)` all
delegate to the state.

## Multi-qubit / sub-MPO application

`apply_subtree_operator(op, where, *, max_bond=None, cutoff=None, renormalize=False)`
applies a general operator on `k >= 1` qubits as a single object, the one-shot
generalisation of the two-qubit gate: a `k`-qubit gate, a multi-site
**non-unitary / Kraus** operator, or a whole **Trotter block**. It is the tree
analogue of a sub-MPO applied over the covering range and then compressed (cf.
Quimb's `MatrixProductState.gate_with_submpo`, which exists for the 1D chain
only). The dense operator is first factorized into an exact tree-MPO on the
**minimal connected subtree** (Steiner subtree) spanning the target physical
nodes.
Application then proceeds recursively from subtree leaves to a hub: each local
state/operator message is losslessly QR-split on one edge and absorbed by its
parent, carrying every still-open operator virtual leg. No dense state tensor
for the whole Steiner subtree is formed. Each dense routed Q tensor retains its
`left_inds` isometry metadata, so canonical recovery recognizes that it already
points toward the hub instead of repeating the same QR; native fermionic trees
retain explicit graded QR recovery. Once all MPO factors have arrived, every
touched edge is SVD-compressed once. Thus every truncation sees the complete
operator in an isometric environment.

`op` acts on `len(where)` qubits: an array reshaped to `(2,) * 2k` with output
indices first, `op[o_0..o_{k-1}, i_0..i_{k-1}]` (a `(2**k, 2**k)` matrix is
accepted). It need **not** be unitary; pass `renormalize=True` to renormalise
afterwards (e.g. after a Kraus/projection operator). `max_bond` / `cutoff`
default to the optimizer's `chi` / `cutoff`. `apply_gate` dispatches `len(where)
== 1` and `== 2` to the optimised leaf-absorb / geodesic-threading paths and any
larger support to `apply_subtree_operator`; the cost scales with the operator's
spread and factor ranks, using recursive edge messages rather than one dense
state tensor for the whole spanning subtree.

An explicit MPS-style sub-MPO marker, `("submpo", mpo, where)` (or the
equivalent mapping form), is accepted in a TreeOptimizer stream. Quimb MPOs are
applied natively by carrying their virtual operator bonds through the same
lossless leaf-to-hub QR sweep followed by one subtree compression sweep; bond
estimates use MPO bond dimensions as a conservative Schmidt-rank bound.
Payloads without the required site interface fall back to `mpo.to_dense()`,
which must produce an operator on the declared support.
`TreeOptimizer.submpo_event(...)` builds the tuple form.

Both two-site implementations preserve native Symmray gates and their
block-sparse fermionic grading. Direct mode splits the rank-four gate; MPO mode
asks Quimb to make the equivalent two-tensor sub-MPO. The resulting two factors
then enter the *same* TTN kernel: attach one factor, QR-thread its shared
operator bond along the unique path, attach the other, then make one final
compression sweep. Neither path converts the gate to a dense qubit array or
splits it into base-2 legs.

For an ordinary two-site gate, choose `mode="direct"` to use the
specialised gate-SVD/QR threading implementation, or `mode="mpo"` to
use the same Quimb gate-to-sub-MPO route. `mode="auto"` is the default:
it selects direct factorization for every backend, including native Symmray
fermionic gates. Select `mode="mpo"` explicitly to inspect or benchmark
Quimb's operator-TN factorization. Direct and MPO share the update kernel and
defer truncation until the complete gate has reached the affected path, so at
an exact `chi` they differ only by the factorization gauge and numerical
roundoff.

`run(mode=...)` has the same persistent semantics as `MpsOptimizer`: it updates
the optimizer's selected two-site mode for that run, later runs, and copies.
The old `run(mode="tree")`/`"ttn"` selector is a deprecated no-op retained only
for shared frontends.

`TreeOptimizer` accepts Quimb's `cutoff_mode` conventions for every truncating
Tree-edge SVD. Its default `"rel"` preserves historical Tree behavior;
`"rsum2"` applies a relative discarded-squared-weight threshold and matches
the default used by `MpsOptimizer`.

`TreeOptimizer.apply_submpo(...)` is the public form for an explicit MPO of
arbitrary support. It losslessly QR-routes its virtual bonds, then uses its
supplied (or configured) `max_bond` / `cutoff` in one final canonical sweep over
the affected subtree.
The tree backend also exposes numerical Pauli primitives used by a future
stabilizer frontend: `apply_pauli_rotation(...)`, `apply_pauli_sum(...)`,
`expectation_pauli(...)`, `measure_pauli(...)`, and `project_pauli(...)`. These
operate on dense two-level qubit coefficient states and do not require tableau
metadata; they intentionally reject native fermionic TTNs.
`measure_pauli` returns `(outcome, probability)` and accepts an optional
`return_diagnostics=True` flag. `project_pauli` normalizes by default; pass
`renormalize=False` to retain the branch norm. Both APIs can report projection
diagnostics containing the norm ratio and support, spanning-tree, and bond
snapshots before and after the update. The records are also available through
`projection_diagnostics` and `get_projection_diagnostics()`.

## Coefficient-backend feature boundary

`TreeOptimizer` covers the state operations shared with `MpsOptimizer`:
ordinary one-/two-/multi-qubit gates, structured sub-MPOs, Pauli expectation
and projection, measurement, reset, measure-reset, cap, normalization,
copying, canonicalization, layout construction, dense readout, and truncation
diagnostics for dense two-level qubit TTNs. A cap's `absorb` argument is
accepted for stream compatibility. A leaf site absorbs into its unique parent;
a root site is contracted directly without changing the tree edges.
`cap(q, vec)` compacts labels by default; use
`stable_labels=True` (or `compact_labels=False`) to preserve caller-facing
logical IDs across the cap while the internal TTN stays compact.
`TreeOptimizer.qubits`, `logical_order`, `position`, and `logical_site` expose
that mapping. Native fermionic TTNs support native Symmray gates and MPOs, but
the qubit Pauli/measurement/reset helpers intentionally reject them; use the
fermion model's native observable/projector with
`TreeTensorNetwork.local_expectation` instead of silently treating a graded
local space as a qubit.

The shared trajectory runner supports dense-qubit `TreeOptimizer` instances as
well. Independent trajectories can sample Pauli mixtures, depolarizing
channels, and state-dependent Kraus channels; branch probabilities are
evaluated from copied TTNs and selected branches are normalized before replay
continues. Coalesced trajectory replay supports exact branching of mid-circuit
measurement, reset, and measure-reset events through `expectation_pauli`. Use
matrix-valued gate payloads in tree streams (for example `pepsy.h()`), since
textual MPS gate aliases are not normalized by the tree gate parser. Native
fermionic trajectories may use native gates/MPOs, but Pauli/control events
require a model-native observable or projector.

MPS execution modes such as `svd`, `dmrg`, `mpo`, `swap`, `perm`, `su`, and
`mix` are chain algorithms and are intentionally not copied into
`TreeOptimizer`. Tree layout is part of the TTN geometry and is selected with
`tree=`/`layout=` at construction. `TreeStabOptimizer` now delegates its first
milestone of numerical coefficient updates to this class while keeping tableau
state and stabilizer-specific bookkeeping above it. See the
[TreeStabOptimizer API](tree_stabilizer.md) for the supported fixed-basis,
basis-updating, immediate, and deferred magic-injection
Clifford/rotation/measurement paths, bounded dense matrix dispatch, and the
safe MPS naming-compatibility surface.

## Tree state class

`TreeTensorNetwork` is the tree analogue of Quimb's `MatrixProductState`: a
geometry-owning subclass of Quimb's arbitrary-geometry vector class
`quimb.tensor.TensorNetworkGenVector`. It *is* a Quimb tensor network, so all of
Quimb's arbitrary-geometry methods (`canonize_around`, `canonize_between`,
`compress_between`, `gate_inds`, `to_dense`, `copy`, ...) apply directly; the
class adds the naming and geometry glue on top of a
`TreePlan`:

- every node (leaf **and** internal) is one tensor tagged with the structural
  node tag `node_tag_id.format(nid)` (default `"N{}"`);
- leaf tensors additionally carry the Quimb site tag `site_tag_id.format(q)`
  (default `"I{}"`) and physical index `site_ind_id.format(q)` (default `"k{}"`)
  for qubit `q`; when `plan.root_qubit` is set, the root tensor carries that
  qubit's site tag and physical index as well;
- adjacent nodes share one live virtual bond. Newly constructed edges use the
  deterministic `_tb{lo}_{hi}` name, but Quimb may replace it with a UUID during
  threading or canonicalisation; `TreeTensorNetwork.bond(a, b)` resolves the
  current live index.

Because the geometry (`plan`) and naming live in `_EXTRA_PROPS`, they survive
`.copy()` and every Quimb view, exactly like `site_ind_id` does for an MPS.
Build one with `TreeTensorNetwork.from_plan(plan)` (product `|0...0>`),
`TreeTensorNetwork.from_order(order, structure=...)` (build the plan and the
product state in one step), or `TreeTensorNetwork.rand(plan, D=..., seed=...)`
(a random state, canonicalised around the root by default). `TreeOptimizer`
builds and evolves its state on this class, delegating all node/qubit naming and
geometry queries to it.

`TreeTensorNetwork.local_expectation(op, where, max_bond=None)` has two
backend-specific exact paths. Dense/nonfermionic TTNs move the centre to the
target physical node/subtree, cancel the ordinary isometric exterior, and contract only
the minimal Steiner subtree. Native fermionic TTNs insert the Symmray operator
without densifying it and contract the complete doubled tree, preserving every
graded boundary phase. For native fermionic states, `max_bond` is accepted for
API compatibility but cannot truncate this exact doubled-network contraction.
Observable readout deliberately belongs to the state, not to `TreeOptimizer`;
use `optimizer.tn.local_expectation(...)`.

Readout is gauge-preserving: a dense expectation restores the previously
tracked canonical centre/region, while an unknown dense gauge is evaluated on a
temporary state copy. Native fermionic expectations do not move the gauge.
The repeated normalized native-readout denominator is cached and invalidated by
state mutation, copying, caps, and canonical/gate updates.

`TreeTensorNetwork.local_expectations(terms, optimize=..., normalized=True)`
evaluates many observables at once, where `terms` maps each `where` (an int
site or a tuple of sites) to its operator. It delegates each term to
`local_expectation` with a *shared* `optimize` handle, so a reusable
`pepsy.build_optimizer(...)` caches one contraction path per topology, and it
reuses the memoized graded norm across the batch. Each returned value matches
the corresponding single-term call exactly. For a Hamiltonian-level energy
readout, `pepsy.TreeEnergyOptimizer` wraps this batch path and returns an
`EnergyEstimate` mirroring `MpsEnergyOptimizer`.

For the package-level product-state constructor, matching `ps_to_mps`, use
`pepsy.ps_to_ttn(n, theta=..., tree=...)`. It builds the requested tree,
initialises every physical site with `[cos(theta), sin(theta)]`, and optionally
expands the virtual bonds with `chi`. Pass `root_qubit=q` to build the plan
directly, or supply a matching root-site `TreePlan` through `tree=`.

For a native Symmray fermionic state, pass a `Fermion` model and occupations:
`pepsy.ps_to_ttn(n, tree=plan, fermion=fermion, occupations=..., chi=1)`.
Physical sites then carry the model's charge/parity sectors, virtual-only
internal nodes are neutral, and every tree edge uses conjugate Symmray virtual
indices.
The constructor selects a definite local Fock basis vector, not a random vector
inside a degenerate charge sector. For spinful `U1`/`Z2`, a scalar occupation
`1` selects the checkerboard `|up>, |down>, ...` representative; pass
`(n_up, n_down)` occupations to choose each spin explicitly. The completed
graded product tree is normalized by an exact graded norm contraction, so its
represented norm is one rather than an arbitrary constructor scalar.
`pepsy.hrs_to_ttn(..., chi=...)` creates the corresponding random symmetric
tree with the requested charge-sector bond dimension and accepts the same
`root_qubit=` option. These constructors keep the Symmray arrays native; they
do not materialize dense tensor data.

`pepsy.TreeSampler(state)` samples every registered physical site, including
the optional root site. Its cached canonical arrays use parent, physical, then
child axes, so probabilities and amplitudes retain normal `q0..q(n-1)` order.

`TreeTensorNetwork.show()` prints a top-down ASCII drawing of the tree -- the
tree analogue of a quimb MPS `show()` -- with the root at the top, structural
leaves at the bottom, physical sites labelled by qubit, and every branch
annotated with its current virtual bond dimension
(`ascii_tree()` returns the same drawing as a string).
`TreeOptimizer.show()` delegates to it.

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
predicts the worst-case multiplicative bond growth. The default
`objective="path"` remains the co-occurrence/path-length heuristic.
`objective="hybrid"` is useful when both replay cost and bond pressure matter:
it combines normalized path score, maximum edge load, and total edge load with
`hybrid_weights=(path, max_edge_load, total_edge_load)`. The
`weight_mode` / `layout_weight_mode` option accepts `count`, `auto`, `angle`, or
`operator_schmidt` for interaction-graph weighting.

`weight_mode="operator_schmidt"` is a cheap **two-qubit entangling-strength
proxy** used to form the spectral qubit order; it is not itself the exact
operator-Schmidt rank. Use `objective="congestion"` when selecting a tree: its
edge-load calculation uses the actual rank across each candidate tree cut (or
an MPO bond bound), which is the quantity that predicts TTN bond growth.

The structure is **not restricted to binary trees**. Internal nodes may have
any arity, controlled by two knobs on `TreeLayoutFinder` / `TreePlan.from_order`
/ `TreeOptimizer`:

- `max_arity` caps the children per internal node. It accepts a scalar (a
  single fixed tree: `2` reproduces the strictly-binary tree exactly, larger
  values give flatter `k`-ary trees with shorter geodesics, `None` leaves the
  arity unbounded) or an iterable of candidate arities to **search**. The
  default `(2, 3, 4)` searches those three and keeps the objective-best plan;
  pass `max_arity=2` to force a fixed binary tree.
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
`TreePlan.from_order(order, weights=..., structure=..., max_arity=...)`, or -- for
a fully hand-specified arbitrary-arity tree -- with
`TreePlan.from_children(children, qubit_of_leaf)`, which validates that the
children map and leaf assignment describe a single rooted tree covering qubits
`0..n-1` exactly once. `TreePlan.max_arity()` and `TreePlan.is_binary()` report
the shape.

For an automatic arity choice, call
`finder.recommend_arities((2, 3, 4))`. This is also what the finder and
`TreeOptimizer` do **by default** (their `max_arity` defaults to `(2, 3, 4)`),
so `TreeOptimizer(gate_stream, n=n, chi=chi)` already searches these arities --
and does so `chi`-aware, since the optimizer forwards its own `chi` (see below).
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

`refine="greedy"` is deterministic and bounded; it is opt-in so existing
fast/default layout construction remains unchanged. A balanced TTN turns a
well-aligned physical span `r` into a path with `O(log r)` tree hops, so the
hybrid score uses path length as a replay-cost proxy while edge loads estimate
the accuracy/bond-dimension cost.

For offline quality searches, add `search="nevergrad"`. Nevergrad starts from
the spectral/greedy plan, proposes leaf orders, and keeps its result only when
it improves the same chi-aware objective. It never acts on a live optimizer.
Install the optional dependency with `pip install pepsy[layout]`:

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
    refine="greedy",
    refine_budget=64,
    search="nevergrad",
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

The default search is made `chi`-aware automatically when a `chi` is available:
`TreeLayoutFinder(gate_stream, n=n, chi=chi)` biases its default `(2, 3, 4)`
search toward `chi`-exact structures, and `TreeOptimizer(gate_stream, n=n,
chi=chi)` forwards its own `chi` into the finder it builds -- so the everyday
`TreeOptimizer(gate_stream, n=n, chi=chi)` already prefers a tree that stays
exact at `chi`. A bare finder with no `chi` searches `chi`-blind.
Set `max_operator_qubits` to bound dense rank diagnostics and operator
allocation; wider native MPO events can still replay without dense
materialization. `TreeLayoutFinder(..., max_operator_qubits=...)` uses a conservative rank
proxy above that width. `report(plan, include_edge_loads=False)` skips the
event-by-edge congestion calculation for path-only diagnostics; when loads are
included, `peak_bond_growth_log2` remains finite even when the human-readable
`peak_bond_growth` would overflow floating point.

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
If a gate, sub-MPO, observable, or cap vector does not match, TreeOptimizer
converts it for compatibility but emits one warning per source/target
combination; this makes an unintended CPU/GPU transfer or dtype cast visible.
Mixed-backend initial states fail immediately because there is no unambiguous
safe execution backend. Internal Pauli/projector tensors follow the state
backend automatically.

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

## Diagnostics

The dominant lever for accuracy at fixed `chi` is the tree structure, so the
finder and optimizer expose diagnostics to choose it:

- `TreeLayoutFinder.report(plan=None)` summarises the physical-node geodesic
  lengths over the interaction graph (`score`, `max_path`, `mean_path`,
  `weighted_mean_path`) and compares against a balanced index tree
  (`balanced_score`, `score_ratio_vs_balanced`). It also reports
  `edge_loads`, `max_edge_load`, and `peak_bond_growth` for the rank-aware
  congestion estimate.
- `TreeOptimizer.bond_report()` reports the current `max_bond`, `mean_bond`,
  and tensor/bond counts -- bonds pinned at `chi` mean truncation is active.
- `TreeOptimizer.estimate_bonds()` performs the paper's non-mutating dry run:
  it multiplies the operator-Schmidt ranks of gates crossing each tree edge,
  returning the conservative Eq. (4) bound before replay. This is useful for
  choosing `chi`; it deliberately ignores cancellations and can overestimate
  the live dimensions.
- `TreeOptimizer.preflight(...)` turns that bound into explicit resource
  protection. `max_bond`, `max_operator_qubits`, and `max_subtree_nodes` can
  reject a replay with `MemoryError`; the same limits can be passed to the
  constructor for automatic checking before eager replay. The constructor
  defaults to `max_operator_qubits=8` and `max_subtree_nodes=128`; pass `None`
  to disable either guard. Product-Pauli measurements use a factorized parity
  projector and do not materialize a `4**k` dense operator.
- `TreeOptimizer.truncation_report()` exposes the per-edge compression and
  SVD-split history, including before/after bond dimensions. Pass
  `track_truncation=True` to also collect each local full singular spectrum's
  absolute discarded weight and relative discarded fraction. Dense states use
  the global spectrum; native Symmray states compare the full and actually
  retained charge-block spectra using the same sector-aware truncation rule as
  the live update. Spectrum probes are opt-in because they add local SVD work
  per truncation edge. The report also contains gate-level `updates`, grouping
  edge events by support and reporting the cumulative relative loss.
- `TreeOptimizer.convergence_sweep(gates, n, chi_values, ops=...)` replays the
  stream at several `chi` on one fixed tree and returns per-`chi` `max_bond`,
  `norm`, observable `expectations`, `fidelity` against the untruncated state
  (when `2**n <= dense_cap`), and observable `max_drift` between consecutive
  `chi` -- a reference-free convergence signal for large systems. Optional
  observables are evaluated by the underlying `TreeTensorNetwork`; they are
  not a `TreeOptimizer` readout API.

## Readout

`to_dense()` returns the dense statevector in index order `k0, k1, ..., k(n-1)`.
`run(progbar=True)` shows a tqdm replay bar with one-/two-/multi-qubit event
counts, current bond usage, state norm, and a norm-based truncation proxy.
Both dense and native fermionic replay report
`1 - (norm / reference_norm)^2`; the reference is established at run start and
reset after control or explicitly non-unitary events. This is display-only and
is not a replacement for the recorded truncation history. The bar is disabled
by default.
For dense two-level qubit TTNs, `measure(q, outcome=None)` projectively
measures a qubit in the computational basis and returns a bit; `reset(q)`
returns a qubit to `|0>`. Native fermionic TTNs deliberately do not expose
these qubit readouts.
For stream control events, `TreeOptimizer.measure_event`,
`cap_event`, `reset_event`, and `measure_reset_event` build the same tuple forms as
`MpsOptimizer`, including Pauli-basis measurement and reset. Their recorded
results are `(pauli, where, outcome, probability)` in `measurements`.
`cap(q, vec)` contracts and removes one physical site, shifting the remaining labels
above `q` down by one unless stable labels are requested.
For a non-unitary run, `normalize_every=True` (or `normalize_final=True`) keeps
the canonical working tensor numerically normalized and accumulates each
removed base-10 scale in `tn.exponent`; `norm()`, `to_dense()`, copies, and
full contractions continue to represent the original physical scale.
The normalization records expose both the per-event raw scale and the
accumulated exponent. The public `normalize()` method remains a physical
renormalization: it clears that exponent and rescales the represented state to
unit norm. `max_bond()` reports the largest virtual bond. Truncation details are
available through `truncation_report()`, `get_infidelities()`, and
`get_infidelity_samples()` when spectrum tracking is enabled.

## Performance and stability

- **Sibling fast path.** A two-qubit gate on two leaves that share a parent is
  applied as a single two-site update: the two leaves and their parent are
  contracted into one blob, the gate is applied, and the blob is re-split by
  two truncating SVDs against the (isometric) surrounding tree. This avoids the
  QR bond-threading and double-bond fusion of the general geodesic route and is
  the common case in a locality-aware layout.

- **Thread cap.** Tree tensors are moderate-rank (set by local arity and the
  optional root physical leg, with dimensions bounded by `chi`), so
  multi-threaded BLAS/OpenMP linear algebra is dominated by thread launch and
  synchronisation overhead. `TreeOptimizer` caps threads to `1` around gate
  application and the heavy read-outs by default (`threads=1`), which makes
  replay both markedly faster and stable in wall-clock time; pass
  `threads=None` to leave the ambient thread count untouched (worthwhile only
  in a large-`chi` regime where a single contraction is itself large). Thread
  limiting uses `threadpoolctl` when available and is a no-op otherwise.
- **Lazy canonical centre.** A freshly built product state has every virtual
  bond at dimension 1, so it is already canonical with the root as
  orthogonality centre; `from_plan` records that centre on the network rather
  than recomputing it on the first gate. Native fermionic product trees are
  additionally normalized by their exact graded norm readout.
- **Routed isometry reuse.** Dense geodesic and subtree QR routing retains each
  Q tensor's `left_inds`, allowing later canonical recovery to reuse the proven
  isometry without repeating the decomposition or entering Quimb's dense
  canonicalization kernel. Final path and subtree compression also consults
  that live proof: when the destination-side tensor is already isometric,
  Quimb uses one-sided `reduced="left"` compression and avoids its redundant
  reduction QR; otherwise it falls back to the full two-sided reduction. The
  network derives orientation diagnostics from those tensors; the optimizer
  does not keep a duplicate map. Native fermionic trees keep their separate
  explicit graded QR/SVD path.
- **State-owned centre.** The orthogonality centre lives on the
  `TreeTensorNetwork` (`orthogonality_center`, an `_EXTRA_PROPS` field), so the
  optimizer and the state cannot disagree and the centre is carried by
  `.copy()`. Incremental moves (`shift_orthogonality_center`) touch only the
  geodesic between old and new centre.
- **Self-healing tid cache.** Node-to-tensor lookups are cached and validated
  against the live tensor map, so the hot path avoids re-scanning tags while
  staying correct when a gate rebuilds a tensor.
- **Resource guards.** `max_intermediate_bond`, `max_operator_qubits`, and
  `max_subtree_nodes` provide preflight and direct-application limits; the
  latter two default to conservative finite values and accept `None` to opt
  out. A dense `k`-qubit operator still has `4**k` payload values, while
  product-Pauli measurement uses a factorized parity projector and recursive
  edge messages.
- **`copy()`.** Returns an independent optimizer that shares the immutable
  `TreePlan` but owns its own tensor network (which carries the tracked
  orthogonality centre), for branching experiments or trial gate sequences.


> API details are maintained as handwritten Markdown in this page.

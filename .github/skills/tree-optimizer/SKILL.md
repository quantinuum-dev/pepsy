---
name: tree-optimizer
description: 'Run, review, debug, benchmark, or extend pepsy.TreeOptimizer -- the rooted tree-tensor-network (TTN) gate-stream circuit simulator of Seitz et al. (Quantum 7, 964, 2023; arXiv:2206.01000). Use when the user asks to replay a circuit on a tree tensor network; build or score a TreeLayoutFinder / TreePlan (entanglement-adapted recursive spectral bisection); tune the exact 2q-gate threading + single canonical compression sweep; work on the sibling-leaf fast path, canonical single-/multi-site local_expectation (Steiner subtree), measure/reset, the BLAS thread cap, the canonical-centre tracker, the tid cache, convergence_sweep / bond_report diagnostics, or copy(); or asks why TTN truncation is more accurate than per-hop truncation, how the tree geodesic threading works, or how to add sampling / stream-wired control events. Not for MPS (pepsy.MpsOptimizer -> mps-optimizer skill), MERA (qmera-energy-optimizer), stabilizer TN (stabilizer-tensor-networks), or BP.'
---

# Tree Optimizer in pepsy

Use this skill for `pepsy.TreeOptimizer` (also `pepsy.optimizers.TreeOptimizer`)
and its implementation under `src/pepsy/optimizers/tree/`:

- [`optimizer.py`](../../../src/pepsy/optimizers/tree/optimizer.py) -- `TreeOptimizer` (state + gate replay + readout).
- [`layout.py`](../../../src/pepsy/optimizers/tree/layout.py) -- `TreePlan` (pure rooted-tree structure, any arity) and `TreeLayoutFinder` (entanglement-adapted structure search).
- [`__init__.py`](../../../src/pepsy/optimizers/tree/__init__.py) -- subpackage exports.
- Docs: [`docs/api/optimizers/tree.md`](../../../docs/api/optimizers/tree.md).
- Tests: [`tests/test_optimize_tree.py`](../../../tests/test_optimize_tree.py).

Read the docs page and the closest tests before editing. It is a thin
tensor-network glue layer: the heavy lifting (canonicalisation, compression,
tensor splitting, tree path finding) uses `quimb` primitives.

## What it implements

The rooted tree-tensor-network circuit simulator of *Simulating quantum
circuits using tree tensor networks* (Seitz, Medina, Cruz, Huang, Mendl;
Quantum 7, 964, 2023; arXiv:2206.01000). The state is a rooted TTN (internal
nodes of **any arity**; binary is the default, see *Non-binary trees* below)
whose leaves carry the physical qubit indices; a bundled gate stream
`[(gate, where), ...]` is replayed. `where` is an `int` (1q) or a pair of `int`
(2q); three-plus-qubit supports raise `NotImplementedError`.

## Tree state class (`TreeTensorNetwork`)

`pepsy.TreeTensorNetwork` (also `pepsy.optimizers.TreeTensorNetwork`, source
`src/pepsy/optimizers/tree/ttn.py`) is the tree analogue of quimb's
`MatrixProductState`: a geometry-owning subclass of
`quimb.tensor.TensorNetworkGenVector` (import from `quimb.tensor`, **not** the
deprecated `tensor_arbgeom`). It owns a `TreePlan` plus the node/site/index
naming, so all inherited quimb methods (`canonize_around`, `canonize_between`,
`compress_between`, `gate_inds`, `local_expectation`, `to_dense`, `copy`) work
directly.

- `_EXTRA_PROPS = ("_sites", "_site_tag_id", "_site_ind_id", "_plan",
  "_node_tag_id")` -- these are copied on `.copy()`/every quimb view. The
  `__init__` copy-branch guard `if isinstance(ts, TensorNetwork): super().
  __init__(ts, **o); return` lets the base copy the extra props without the
  fresh-construction defaults clobbering `_plan`.
- Each leaf tensor carries **both** the structural node tag `N{nid}` and the
  quimb site tag `I{q}` plus physical index `k{q}`; internal nodes carry only
  `N{nid}`. So quimb sees the leaves as the `nsites` sites and internal nodes as
  ancillary bond carriers -- `local_expectation(G, where=[q], max_bond=None,
  optimize="auto")` works (emits a cosmetic "not a compressed tree" warning).
- `node_tid(nid)` is a self-healing tid cache kept in `__dict__` (not
  `_EXTRA_PROPS`) so a copy starts with a fresh, independent cache.
- Builders: `from_plan(plan)` (product `|0...0>`), `from_order(order,
  structure=...)` (plan + product in one call), `rand(plan, D=, seed=,
  canonicalize=True)` (random state, canonicalised around the root).
- `show()` prints a top-down ASCII tree (root on top, qubit leaves `◆ q{q}` at
  the bottom, internal `●`, each branch annotated with its bond dim);
  `ascii_tree()` returns that string. `TreeOptimizer.show()` delegates to it.
- `TreeOptimizer.tn` **is** a `TreeTensorNetwork`; the optimizer delegates
  `_phys->tn.site_ind`, `_tag->tn.node_tag`, `_tid->tn.node_tid`,
  `_neighbors->tn.neighbors`, `_steiner_nodes->tn.steiner_nodes`, and
  `_build_product_state->TreeTensorNetwork.from_plan`. Keep these names/values
  identical (`k{q}`, `N{nid}`, `_tb{lo}_{hi}`) so behaviour is unchanged.

## Conventions (must stay consistent across optimizer + layout)

- Node ids are ints from `TreePlan`. Tensor tag = `N{nid}` (`TreeTensorNetwork.
  node_tag`, via optimizer `_tag`).
- Physical index of qubit `q` = `k{q}` (`TreeTensorNetwork.site_ind`, via
  optimizer `_phys`) -- ket-leg convention. Leaves also carry site tag `I{q}`.
- Virtual bond between adjacent nodes `u,v` = `_tb{lo}_{hi}` with `lo<hi`
  (`TreeTensorNetwork.bond` / optimizer `_bond_name`). This deterministic name
  lets splits keep tree-edge identity via `bond_ind=`.
- `plan.node_path(a, b)` is the inclusive node-id geodesic (unique in a tree);
  `plan.tree_distance(qa, qb)` is the leaf-to-leaf path length.

## Canonical-centre contract (core invariant)

The orthogonality centre is a single node id **owned by the `TreeTensorNetwork`**,
and it is the one-node case of the more general **canonical region** (a connected
node set) tracked in `ttn._canonical_region` (declared in `_EXTRA_PROPS` so it
survives `.copy()` and quimb views). `ttn.orthogonality_center` is *derived* from
that region: the sole node when the region has size 1, else `None` (honest "no
single centre"). `TreeOptimizer.center` is a thin property view onto it, so the
optimizer and the state can never disagree. It is **algorithm state**, not
cosmetic: every tensor outside the region must be isometric pointing inward so it
telescopes to identity between bra and ket.

- `TreeTensorNetwork.shift_orthogonality_center(node)` is the primitive: it walks
  the geodesic from the current centre to `node` with `canonize_between(absorb=
  "right")` per edge (lossless QR), touching only the path tensors (O(path
  length), not O(N)); it is idempotent when already centred and falls back once
  to `canonize_around_node_` (O(N)) when the centre is unknown (`None`). This is
  the tree analogue of quimb's MPS `shift_orthogonality_center` /
  `MpsOptimizer.info_c["cur_orthog"]`.
- `TreeOptimizer._move_center(target)` simply delegates to it.
- `ttn.is_canonical_form(center)` verifies the invariant directly (every
  non-centre tensor is an isometry toward the centre) — use it in tests/diagnostics.
- A freshly built product state is **already canonical at the root** (all
  virtual bonds are dim 1, so every tensor is trivially isometric). `from_plan`
  records `orthogonality_center = root`, so the first gate skips an O(N)
  canonicalisation. Do not reset this to `None`.
- `canonize_edge_` / `compress_edge_` advance the tracked centre by one hop when
  it starts on the isometric side, else set it to `None` (honest: a lone edge
  move cannot leave a global centre). The optimizer's hot paths call quimb
  `canonize_between` / `compress_between` **directly** and set `self.center`
  explicitly afterward — keep that.
- Unitary 1q gates preserve canonical form regardless of centre (absorbed into
  the leaf, no bond growth, no centre move). Non-unitary 1q operators
  (projectors in `measure`) keep the centre on that leaf but require a
  subsequent `normalize()`.
- `norm()` uses the single centre tensor when `center is not None` (one-site
  canonical norm); only the unknown-centre fallback does a full doubled-tree
  contraction. Keep this fast path.
- Any operation that moves/rebuilds the centre must update the tracked centre
  (via `self.center = ...`, i.e. `ttn.orthogonality_center`).

### Range / subtree canonicalisation

The centre generalises to a connected subtree — the tree analogue of an MPS
mixed-canonical range. Do **not** reintroduce a separate `_orthog_center` field:
`_canonical_region` is the single source of truth and the single centre is its
one-node case.

- `ttn.canonize_subtree_(nodes, span=False, absorb="right")` gauges every tensor
  outside a connected subtree inward via quimb `canonize_around_(tags,
  which="any")` (**`which="any"`** selects the union of region tags — `"all"`
  would intersect to empty). The whole norm concentrates on the region:
  `(region.H | region) ^ all` equals the full squared norm. Sets
  `_canonical_region`. `canonize_around_node_({nid})` is the one-node delegate.
- Disconnected `nodes` raise unless `span=True`, which expands to the minimal
  connected subtree via `ttn.subtree_span(nodes)` (union of tree paths from
  `nodes[0]`; generalises `steiner_nodes` to arbitrary internal nodes).
- `ttn.canonize_around_qubits_(qubits)` is the qubit-level "range" entry point =
  `canonize_subtree_(leaves_of(qubits), span=True)`.
- `ttn.is_subtree_canonical_form(nodes=None, span=False)` verifies every outside
  tensor is an inward isometry (defaults to the tracked region);
  `is_canonical_form` is its one-node case and delegates to it.
- `TreeOptimizer` mirrors all of this: `canonical_region` property,
  `canonize_subtree(nodes, span=...)`, `canonize_around_qubits(qubits)`,
  `is_subtree_canonical_form(nodes)` — all thin delegates to the state.


## Two-qubit gate = exact threading + one compression sweep

This is the paper's accuracy point (Figs. 3-6) -- do not regress it.

1. SVD-split the gate into left/right factors joined by a virtual bond
   (`cutoff=0.0`, exact rank `k <= 4`).
2. Move the centre to leaf `a`, absorb the left factor into `a`.
3. Thread the virtual bond **exactly** along the geodesic to leaf `b` via
   `_thread_hop` (economical **QR**, lossless, `absorb="right"`); the crossed
   bond grows transiently by at most `k <= 4`.
4. Absorb the right factor into leaf `b`.
5. Only now run `_compress_path` -- a single canonical compression sweep back
   along the geodesic, truncating every touched bond to `chi`.

Because both gate factors are present before any truncation, each SVD sees the
complete gate -- markedly more accurate at finite `chi` than truncating each hop
while threading. `compress_between` kwargs: `(tags1, tags2, max_bond, cutoff,
absorb, canonize_distance=0, ...)`; **`canonize=` is NOT a valid kwarg** (it is
forwarded to the SVD and raises `TypeError`). Unique `rand_uuid()` bonds avoid
"index appears more than twice" errors during threading.

### Sibling-leaf fast path (`_apply_2q_siblings`)

When `plan.parent[la] == plan.parent[lb]` the two leaves meet at a shared
parent, so no threading is needed. Move the centre to the parent, contract the
three tensors into one blob, apply the gate, and re-split by two truncating
SVDs (`absorb="right"` -> the two leaf factors are isometric, the parent is the
new centre). Both new bonds keep their canonical `_tb...` names via `bond_ind=`.
This is the common case in a good layout and avoids the QR hops and double-bond
fusion. Leaves are never directly bonded (both bond only to the parent), so the
correlation flows through the parent blob -- this is exact up to the truncation.

## Readout

- `local_expectation(op, where)`: single-site contracts only the centre tensor
  with `op` (canonical). **Multi-site contracts only the minimal Steiner
  subtree spanning the target leaves** (`_steiner_nodes` = union of
  `node_path(leaves[0], leaf)`): move the centre inside that subtree, rename the
  subtree-**internal** bonds in the bra (fresh `rand_uuid`) while keeping the
  **boundary** bonds shared between bra and ket. Sharing a boundary bond name
  implements the isometric-exterior identity `sum_b ket[..b] bra[..b] =
  sum_{b,b'} ket delta(b,b') bra`. In a binary tree the leaf-leaf path passes
  only through internal (physical-index-free) nodes, so the subtree's leaves are
  exactly the targets. Cost scales with operator spread, not `n`.
- `measure(q, outcome=None)`: move centre to the leaf, read exact Born
  probabilities from that one tensor (`w_i = sum_bond |t[i,bond]|^2`,
  normalise), sample via `self.rng.choice` or force `outcome`, project with a
  one-hot `apply_1q`, then `normalize()`. Returns the outcome bit. `reset(q)` =
  `measure` then conditional `X`. `seed` in `__init__` sets `self.rng`; `copy()`
  gets a fresh independent RNG.
- `to_dense()` returns the statevector in `k0, k1, ..., k(n-1)` order.
- `bond_report()` / `max_bond()` / `convergence_sweep(...)` are diagnostics.
  `convergence_sweep` builds the tree once and reuses it across `chi` so the
  comparison isolates truncation from layout; it reports `fidelity` (only when
  `2**n <= dense_cap`) and reference-free `max_drift`.

## Performance / stability (do not regress)

- **BLAS thread cap is the biggest perf lever.** Tree tensors are tiny (rank
  `<= 3`, bounded by `chi`), so multi-threaded BLAS/OpenMP is dominated by
  thread launch/sync overhead -- per-gate cost otherwise *grows* under
  oversubscription even at constant TN size. `threads=1` is the default; gate
  application and heavy read-outs run inside `self._thread_ctx()`
  (a `threadpoolctl` `ThreadpoolController().limit(...)`, built once at import,
  `contextlib.nullcontext()` when threadpoolctl is missing or `threads=None`).
  Measured ~12-45x on n=16 chi=16. Only raise `threads` in a large-`chi` regime.
- **Self-healing tid cache** (`_nid_to_tid`, `_tid`): caches node->tensor id,
  validates against `self.tn.tensor_map` (quimb changes a tensor's identity when
  rebuilt via `gate_inds_`); a stale entry just misses and is recomputed. Tensor
  ids are unique and never reused, so this is always safe.
- `copy()` shares the immutable `TreePlan`, owns `self.tn.copy()`, resets the
  tid cache, and gets a fresh RNG.

## Layout (TreeLayoutFinder / TreePlan)

`TreeLayoutFinder` reuses the MPS interaction-graph + recursive spectral
(Fiedler) partition machinery, but **keeps the recursion as the rooted tree**
(the MPS finder flattens it to 1D). Strongly coupled qubits become nearby
leaves, minimising the geodesic a 2q gate must thread.

- Partition uses `_similarity_weights()` = Seitz Eq. 1:
  `s(qi,qj) = |G(qi) & G(qj)| + 1/(deg_i + deg_j)`. The co-occurrence term is
  the accumulated `pair_weights`; the `1/(deg)` term is a tie-breaker.
- `score(plan)` uses the **pure** `pair_weights` (weighted geodesic sum, lower
  is better) -- keep it separate from the augmented partition similarity.
- `report(plan)` compares against a naive `structure="balanced"` index-order
  tree (`score_ratio_vs_balanced`). Quality must be `<=` balanced.
- `TreePlan.from_order(order, weights=, structure=, max_arity=2, community_frac=,
  star_frac=)`: `"quality"` = spectral reorder + split; `"balanced"` = split
  `order` directly (useful in tests to force sibling relationships, e.g.
  `range(4)` -> `(0,1)` and `(2,3)` are siblings); `"adaptive"` = community /
  clique-driven variable arity.

### Non-binary trees (arity is a knob, not a constraint)

The **data structures and all algorithms are already arity-agnostic** -- every
builder, geometry query, `ascii_tree`, the optimizer's geodesic threading +
sibling fast path, and `TreeSampler` loop over `plan.children[nid]` generically.
The *only* binary-specific piece was construction. Controls:

- `max_arity` (default `2`) caps children per internal node. `2` reproduces the
  strictly-binary tree **exactly** (cut points use `floor(i*L/k)` so the 2-way
  case matches the old `mid = L//2` bisection); larger values / `None` give
  flatter `k`-ary trees.
- `structure="adaptive"` branches per strongly coupled community
  (`community_frac` of the level's strongest edge) and collapses a near-clique
  (present-strong-edge fraction `>= star_frac`) into a flat **star** node
  (all intra-clique geodesics length 2 vs up to 3 for a bisection).
- `TreePlan.from_children(children, qubit_of_leaf, root=None)` validates and
  builds an arbitrary hand-specified tree (checks single parent, leaf/qubit
  coverage of `0..n-1`, reachability, no cycles).
- `TreePlan.max_arity()` / `TreePlan.is_binary()` report the shape.
- All these are plumbed identically through `TreeLayoutFinder`, `TreeOptimizer`,
  `TreeOptimizer.find_tree_layout`, `TreeOptimizer.convergence_sweep`, and
  `TreeTensorNetwork.from_order`. The `TreeLayoutFinder` default is still binary
  so existing behaviour is unchanged.

## Gotchas / teaching notes

- `convergence_sweep` observable `max_drift` can show a **false plateau**: a
  garbage low-`chi` state can have small drift while fidelity is ~0.01. Trust
  fidelity (small `n`) or push `chi`; drift alone lies.
- Do not truncate while threading -- it breaks the "each truncation sees the
  whole gate" accuracy property.
- Do not pass `canonize=` to `compress_between`.
- 2q gate reshape order is `(out_a, out_b, in_a, in_b)`.

## Roadmap / not yet implemented

- **Sampling** (batched bitstring sampling from the canonical tree) is the next
  natural feature and is *not* done. It should reuse the `measure` centre-on-leaf
  Born-probability read plus conditional propagation, mirroring `MpsSampler`.
- Stream-wired control events: `measure`/`reset` are explicit methods; the
  `[(gate, where)]` stream does not yet carry measurement/reset event markers
  (unlike `MpsOptimizer` control events).

## Validation

Activate the shared environment and use temp caches:

```bash
source ~/envs/py312/bin/activate
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mplconfig \
  PYTHONPYCACHEPREFIX=/tmp pytest -q tests/test_optimize_tree.py
```

For public-API/layout changes also run:

```bash
pytest -q tests/test_public_api.py tests/test_package_layout.py
python -m pyflakes src/pepsy/optimizers/tree tests/test_optimize_tree.py
sphinx-build -W -b html docs docs/_build/html
```

The safety-net tests are `test_tree_matches_statevector` (untruncated fidelity
must stay exactly 1.0) and the multi-site / sibling / measurement regressions.
Add a regression test for every new behaviour and prefer `structure="balanced"`
plans when a test needs deterministic sibling relationships.

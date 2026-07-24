---
name: tree-optimizer
description: 'Run, review, debug, benchmark, or extend pepsy.TreeOptimizer -- the rooted tree-tensor-network (TTN) gate-stream circuit simulator of Seitz et al. (Quantum 7, 964, 2023; arXiv:2206.01000). Use when the user asks to replay a circuit on a tree tensor network; build or score a TreeLayoutFinder / TreePlan (entanglement-adapted recursive spectral bisection); handle exact product-state handoff, TTN layout safety, or Torch/CuPy backend compatibility; tune the exact 2q-gate threading + single canonical compression sweep; work on TreeTensorNetwork canonical single-/multi-site local_expectation (Steiner subtree), measurement/reset, independent or coalesced noisy trajectories, the BLAS thread cap, the canonical-centre tracker, the tid cache, convergence_sweep / bond_report diagnostics, or copy(); or asks why TTN truncation is more accurate than per-hop truncation, how the tree geodesic threading works, or how to add sampling / stream-wired control events. Not for MPS (pepsy.MpsOptimizer -> mps-optimizer skill), MERA (qmera-energy-optimizer), stabilizer TN (stabilizer-tensor-networks), or BP.'
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
(2q); supports with `len(where) >= 3` route through
`apply_subtree_operator` (see *Multi-qubit / sub-MPO application*).

Preferred public handoff:

```python
finder = TreeLayoutFinder(gate_stream, n=n, objective="congestion")
choice = finder.recommend_arities((2, 3, 4))
optimizer = TreeOptimizer(gate_stream, tree=choice["plan"], chi=chi)
```

Alternatively pass the finder itself with `layout=finder`. An initial
non-product or entangled state must be passed explicitly as `state=` (alias
`tn=`). `tree=` and `layout=` accept only `TreePlan` / `TreeLayoutFinder`; a
`TreeTensorNetwork` passed there must raise a clear error rather than being
silently replaced by the default `|0...0>` state.

### State/layout handoff and backend contract

`TreeLayoutFinder` is **circuit-only**: it accepts a gate stream or explicit
supports, never a `TreeTensorNetwork`. Passing a TTN to it must raise a clear
`TypeError`; construct the plan from the circuit, then pass the state separately
to `TreeOptimizer`.

- An entangled `TreeTensorNetwork` (`max_bond() > 1`) can be installed only on
  its matching `TreePlan`. A different `tree=` / `layout=` must raise before
  tensor work: implicit relayout would be lossy and hide a fidelity decision.
- A product `TreeTensorNetwork` (`max_bond() == 1`) may be remounted **exactly**
  on a requested plan; warn that its geometry changed. Preserve the product
  vectors and any distributed global scalar/phase.
- A bond-one Quimb `MatrixProductState` is also a geometry-neutral product input
  and may be mounted exactly on the selected tree. Reject an entangled MPS;
  converting it to a tree is an explicit caller-controlled operation.
- All live state tensors must use one backend, dtype, and device.
  `backend_info()` reports this. Reject a mixed state at construction or
  `set_tn`, rather than choosing an arbitrary execution backend.
- Callers should convert every gate/operator/sub-MPO/observable/cap vector with
  the same backend converter as the state. Explicit array mismatches are
  coerced for compatibility with a one-time `UserWarning` per source/target
  signature; untyped Python lists/scalars are convenience inputs and are
  materialized silently. Internal Pauli, projector, reset, and tree-MPO helper
  tensors must be built on the state backend without warning.
- Backend-aware contractions must stay in Autoray/Quimb. Convert only scalar
  readouts intentionally (`to_float`); `to_dense()` intentionally returns a
  host NumPy vector for interoperability while the live TTN remains native.

## Tree state class (`TreeTensorNetwork`)

`pepsy.TreeTensorNetwork` (also `pepsy.optimizers.TreeTensorNetwork`, source
`src/pepsy/optimizers/tree/ttn.py`) is the tree analogue of quimb's
`MatrixProductState`: a geometry-owning subclass of
`quimb.tensor.TensorNetworkGenVector` (import from `quimb.tensor`, **not** the
deprecated `tensor_arbgeom`). It owns a `TreePlan` plus the node/site/index
naming, so all inherited quimb methods (`canonize_around`, `canonize_between`,
  `compress_between`, `gate_inds`, `to_dense`, `copy`) work directly. The
  state overrides `local_expectation` with a canonical Steiner-subtree
  contraction that also supports native Symmray observables.

- `_EXTRA_PROPS = ("_sites", "_site_tag_id", "_site_ind_id", "_plan",
  "_node_tag_id", "_canonical_region", "_symmetry", "_fermionic",
  "_physical_sectors")` -- these are copied on `.copy()`/
  every quimb view. The `__init__` copy-branch guard
  `if isinstance(ts, TensorNetwork): super().__init__(ts, **o); return` lets
  the base copy the extra props without the fresh-construction defaults
  clobbering `_plan`.
- Each leaf tensor carries **both** the structural node tag `N{nid}` and the
  quimb site tag `I{q}` plus physical index `k{q}`; internal nodes carry only
  `N{nid}`. So quimb sees the leaves as the `nsites` sites and internal nodes as
  ancillary bond carriers --
  `ttn.local_expectation(G, where=[q], max_bond=None, optimize="auto")` uses
  the tree's canonical contraction for dense states and an exact complete
  doubled-tree contraction for native fermionic states.
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
- Newly created virtual bonds between adjacent nodes `u,v` use `_tb{lo}_{hi}`
  with `lo<hi` (`optimizer._bond_name`), but Quimb may mint UUIDs during
  threading or canonicalisation. `TreeTensorNetwork.bond(u, v)` resolves the
  live shared index; use it for diagnostics and readout.
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
  to `canonize_around_node_` (O(N)) only when the centre and canonical region
  are both unknown. This is the tree analogue of quimb's MPS
  `shift_orthogonality_center` /
  `MpsOptimizer.info_c["cur_orthog"]`.
- `TreeOptimizer._move_center(target)` simply delegates to it.
- When the centre is unknown but `_canonical_region` contains multiple nodes,
  `shift_orthogonality_center` first peels that region with lossless QR and
  then walks only the remaining path. Do not regress this regional recovery to
  an unconditional O(N) recanonicalisation.
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
- `norm()` uses the single centre tensor for dense/nonfermionic states when
  `center is not None`; only their unknown-centre fallback contracts the full
  doubled tree. Native fermionic states use a one-tensor
  `TensorNetwork.H` contraction when a centre is known, so Symmray applies the
  graded outer-leg phase flips; unknown-centre fermionic states use the exact
  complete doubled-network contraction. Keep the backend dispatch separate.
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

Native fermionic trees take an isolated version of this kernel:
`_fermionic_thread_hop` explicitly calls the native Symmray QR and carries its
graded factor, while `TreeTensorNetwork._fermionic_compress_edge_` forms the
two-node tensor and performs the native block SVD. Dense/nonfermionic trees
retain the generic Quimb edge wrappers.

### Sibling-leaf fast path (`_apply_2q_sibling_factors`)

When `plan.parent[la] == plan.parent[lb]` the two leaves meet at a shared
parent, so no threading is needed. Both direct-SVD and Quimb-MPO factors are
absorbed into their leaves, then the two leaves and parent are contracted into
one blob and re-split by two truncating SVDs (`absorb="right"` -> the two leaf
factors are isometric, the parent is the new centre). Both new bonds keep their
canonical `_tb...` names via `bond_ind=`. This is the common case in a good
layout and avoids QR hops and double-bond fusion. Leaves are never directly
bonded (both bond only to the parent), so the correlation flows through the
parent blob -- this is exact up to the truncation.

## Multi-qubit / sub-MPO application (`apply_subtree_operator`)

`apply_subtree_operator(op, where, *, max_bond=None, cutoff=None,
renormalize=False)` applies a general operator on `k >= 1` qubits in one shot --
a `k`-qubit gate, a multi-site **non-unitary / Kraus** operator, or a whole
**Trotter block**. It extends the two-factor path-thread kernel to the whole
spanning subtree: the tree analogue of a sub-MPO applied over a
covering range then compressed (quimb's `gate_with_submpo` is `MatrixProductState`
-only; the tree base `TensorNetworkGenVector` has no such method).

1. `snodes = _steiner_nodes(leaves)` -- minimal connected subtree spanning the
   target leaves.
2. Move the centre onto a target leaf (`_move_center(leaves[0])`, incremental)
   so the **whole exterior is isometric toward the subtree**.
3. Factor `op` into an exact tree-MPO on the same Steiner tree by packing each
   `(output,input)` physical pair into a dimension-four leg and applying
   leaf-to-hub SVDs.
4. Absorb the tree-MPO into copied local state tensors. For each
   `_peel_order(snodes)` edge, QR-split the child message while retaining all
   physical and exterior state legs, then contract its new state bond into the
   parent together with the old state/operator bonds. No dense state tensor for
   the whole Steiner subtree is formed; the last node is the hub.
5. Recover the hub centre by QR, then make one depth-first canonical SVD sweep:
   every affected tree edge is truncated once, after the complete operator has
   arrived. `renormalize=True` renormalises afterwards (for Kraus/projection).

State bonds are always read from the live tensors because gate application can
rename them. New state message bonds are fresh per-update names, while operator
bonds are private to the temporary tree-MPO. `apply_gate` routes
`len(where) >= 3` here; `k == 1`/`k == 2` still take the optimised
leaf-absorb / threading paths (but `k == 1` non-unitary and `k == 2` Kraus
can be sent here explicitly).

### Native streamed sub-MPOs

An explicit `("submpo", mpo, where)` stream event first attempts the native
leaf-to-hub QR-routing sweep. Quimb MPO payloads expose their active site tags,
tensor map, and operator bond indices, so their virtual bonds can be carried
through the TTN without calling `mpo.to_dense()`, then compressed once over the
affected subtree. `estimate_bonds()` uses the product of MPO bond dimensions
crossing a cut as a conservative operator-Schmidt bound. Payloads without that
interface use the dense `to_dense()` fallback and remain subject to
`max_operator_qubits`.

## Readout

- `TreeTensorNetwork.local_expectation(op, where)`: dense/nonfermionic
  single-site readout contracts the centre tensor; dense multi-site readout
  contracts the minimal Steiner subtree. Native fermionic readout instead
  inserts the Symmray operator with `contract=False` and contracts the complete
  doubled tree. This preserves graded boundary phases and deliberately avoids
  the ordinary isometric-exterior shortcut. Its `max_bond` argument is
  compatibility-only: the exact native contraction is not truncated. Dense
  readout restores a known canonical centre/region and uses a temporary copy
  when the gauge is unknown; native readout leaves the gauge untouched.
  Normalized native readout reuses a state-versioned norm denominator until a
  mutation invalidates it.
- `measure(q, outcome=None)`: move centre to the leaf, read exact Born
  probabilities from that one tensor (`w_i = sum_bond |t[i,bond]|^2`,
  normalise), sample via `self.rng.choice` or force `outcome`, project with a
  one-hot `apply_1q`, then `normalize()`. Returns the outcome bit. `reset(q)` =
  `measure` then conditional `X`. `seed` in `__init__` sets `self.rng`; `copy()`
  derives a deterministic child seed for a fresh independent RNG.
- Stream control events follow the MPS tuple/mapping contract for `measure`,
  `cap`, `reset`, and `measure_reset`. Pauli measurements support product observables
  on distinct qubits, use `+1`/`-1` eigenvalue outcomes, and append
  `(pauli, where, outcome, probability)` to `measurements`; reset measurements
  are internal and are not recorded. A cap contracts and removes one leaf,
  compacts labels above it by default, and absorbs into the unique tree parent;
  `stable_labels=True` / `compact_labels=False` preserves caller-facing labels
  while storage remains compact. `measure_pauli` returns outcome and Born
  probability directly; `project_pauli(..., renormalize=False)` preserves the
  branch norm, and both can return support/span/bond/norm diagnostics.
- `to_dense()` returns a host NumPy statevector in `k0, k1, ..., k(n-1)` order;
  it is a readout boundary, not evidence that a Torch/CuPy live state moved.
- `run(progbar=True)` shows a tqdm replay bar with one-/two-/multi-qubit
  counts, current bond usage, norm, and a norm-based truncation proxy. Dense and
  native fermionic replay use the same `1 - (norm / reference_norm)^2` proxy;
  the reference resets after control or explicitly non-unitary events. This is
  display-only, not a substitute for truncation history.
- `bond_report()` / `estimate_bonds()` / `max_bond()` /
  `convergence_sweep(...)` are diagnostics. `estimate_bonds()` is the
  non-mutating Eq. (4) dry run: it multiplies operator-Schmidt ranks on each
  crossed edge and can conservatively flag a `chi` that will truncate.
- `TreeTensorNetwork.validate()` checks the live tensor/bond structure against
  its `TreePlan`; use `check_canonical=True` for the more expensive isometry
  check. `TreeOptimizer.preflight(...)` adds `max_bond`,
  `max_operator_qubits`, and `max_subtree_nodes` resource limits before replay;
  the constructor defaults to finite dense/operator-subtree guards, and
  `None` disables either guard. Product-Pauli measurement uses a factorized
  parity projector rather than a dense `4**k` matrix.
  `convergence_sweep` builds the tree once and reuses it across `chi` so the
  comparison isolates truncation from layout; it reports `fidelity` (only when
  `2**n <= dense_cap`) and reference-free `max_drift`.
- `record_history=False` disables retained per-edge and per-update history for
  long replays. `TreeTensorNetwork` invalidates its canonical-region metadata
  and native norm cache after direct Quimb mutators; use
  `invalidate_canonical_form()` after raw tensor edits. The optimizer's
  state-aware wrappers restore a known centre only for operations proven to
  preserve canonicality.
- `truncation_report()` returns the per-edge compression / split history with
  before/after dimensions. `track_truncation=True` additionally probes the
  untruncated local singular spectrum and records absolute discarded weight and
  relative discarded fraction; native Symmray reports use the actually kept
  charge-block spectrum. Leave it false on performance runs: the spectrum
  probe adds local SVD work per truncation edge. The report's gate-level
  `updates` group edge events by support and include cumulative relative loss,
  analogous to the MPS infidelity trace.

## Noisy trajectory replay

`run_trajectory_shots` and `run_coalesced_trajectory_shots` support
`TreeOptimizer` factories as well as MPS and stabilizer-TN factories. Use them
for trajectory simulation without forming a density matrix:

- Independent replay samples random-unitary mixtures, Pauli/depolarizing
  channels, and state-dependent Kraus channels. For a Kraus event, the runner
  applies each branch to a copied TTN, obtains its squared norm, samples the
  conditional probability, then applies and normalizes the selected branch on
  the live TTN.
- Coalesced replay shares deterministic prefixes and branches exact
  mid-circuit `measure`, `reset`, and `measure_reset` events. Tree measurement
  probabilities come from `TreeOptimizer.expectation_pauli`; each resulting
  leaf remains normalized.
- The runner converts generated dense matrices through the live state backend.
  When constructing a direct Tree stream, use matrix-valued gate payloads such
  as `pepsy.h()`; textual MPS gate aliases are not normalized by the Tree gate
  parser.
- Regression coverage lives in `tests/test_trajectory_noise.py`, including
  Tree state-dependent Kraus sampling and coalesced measurement branching.

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
  tid cache, and derives a deterministic child seed for a fresh independent
  RNG.

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
  tree (`score_ratio_vs_balanced`). The path-objective quality layout should be
  `<=` balanced; congestion mode is selected by edge-load cost instead.
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

- `max_arity` defaults to the candidate search `(2, 3, 4)`; pass scalar `2` to
  force a strictly binary tree. `2` reproduces the original binary partition
  exactly (cut points use `floor(i*L/k)` so the 2-way case matches the old
  `mid = L//2` bisection); larger values / `None` give flatter `k`-ary trees.
- `structure="adaptive"` branches per strongly coupled community
  (`community_frac` of the level's strongest edge) and collapses a near-clique
  (present-strong-edge fraction `>= star_frac`) into a flat **star** node
  (all intra-clique geodesics length 2 vs up to 3 for a bisection).
- `TreePlan.from_children(children, qubit_of_leaf, root=None)` validates and
  builds an arbitrary hand-specified tree (checks single parent, leaf/qubit
  coverage of `0..n-1`, reachability, no cycles).
- `TreePlan.max_arity()` / `TreePlan.is_binary()` report the shape.
- `TreeLayoutFinder.recommend_arities((2, 3, 4))` compares candidate arities
  using path cost or the rank-aware congestion objective and reports local
  virtual degree, edge load, and peak bond growth. `report(plan)` also exposes
  `is_binary`, `max_arity`, and `arity_histogram`.
- `objective="congestion"` compares interaction, congestion-aware, and
  balanced candidates using predicted `log2` operator-Schmidt rank load on
  each edge. `objective="path"` remains the backward-compatible default.
- `weight_mode` accepts `count`, `auto`, `angle`, and `operator_schmidt` for
  interaction-graph event weighting. `TreeOptimizer` exposes these as
  `layout_objective` and `layout_weight_mode`.
- `layered(block_size=...)` is deliberately a fixed construction: it uses the
  spectral order but does not score block sizes or use `chi`. Prefer
  `recommend_layered((2, 3, 4), chi=...)` for selection; when `chi` is omitted
  it inherits the finder's configured value (pass `chi=None` to compare blind).
  `weight_mode="operator_schmidt"` is a two-site entangling-strength proxy;
  `objective="congestion"` performs the actual rank-per-tree-edge selection.
- `objective="hybrid"` combines normalized weighted path, maximum edge load,
  and total edge load using `hybrid_weights=(path, max_edge_load,
  total_edge_load)`. It is a fixed-tree replay/runtime-and-bond-cost proxy;
  it does not permit layout rewrites during simulation.
- `recommend_layered` / `recommend_arities` accept `refine="greedy"` for a
  bounded deterministic pre-simulation adjacent leaf-label swap pass. It keeps
  parent/child topology fixed and returns planning metadata per candidate.
  `search="nevergrad"` is a separate optional offline search, seeded and
  budgeted per candidate, requiring `pepsy[layout]`; it starts from the
  spectral/greedy plan and retains only objective improvements. Keep both
  opt-in: default layout construction stays fast and reproducible.
- Layout hot paths matter for long streams: `edge_loads` must traverse only the
  support's Steiner subtree, and a dense gate's Schmidt-rank cache key must use
  wire positions rather than global labels so one repeated CNOT/CZ is factored
  once per distinct partition.
- All these are plumbed identically through `TreeLayoutFinder`, `TreeOptimizer`,
  `TreeOptimizer.find_tree_layout`, `TreeOptimizer.convergence_sweep`, and
  `TreeTensorNetwork.from_order`.

## Gotchas / teaching notes

- `convergence_sweep` observable `max_drift` can show a **false plateau**: a
  garbage low-`chi` state can have small drift while fidelity is ~0.01. Trust
  fidelity (small `n`) or push `chi`; drift alone lies.
- Do not truncate while threading -- it breaks the "each truncation sees the
  whole gate" accuracy property.
- Do not pass `canonize=` to `compress_between`.
- 2q gate reshape order is `(out_a, out_b, in_a, in_b)`.

## Roadmap / not yet implemented

- Chain-only MPS execution modes (`svd`, `dmrg`, `mpo`, `swap`, `perm`, `su`,
  and `mix`) are not meaningful on arbitrary tree geometry. Native structured
  sub-MPO payloads are routed through the Quimb MPO site interface; only
  payloads that do not expose that interface use the guarded dense fallback.
  Pauli and computational-basis measurement helpers are dense two-level qubit
  APIs and intentionally reject native fermionic TTNs, whose local observables
  must be supplied through the fermion/Symmray model layer.

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
must stay exactly 1.0), the multi-site / sibling / measurement regressions, and
the state-handoff/backend cases: exact product TTN/MPS mounting, rejected
entangled relayouts, native Torch controls/readout, and mixed-backend rejection.
Add a regression test for every new behaviour and prefer `structure="balanced"`
plans when a test needs deterministic sibling relationships.

For noisy trajectory changes, also run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_trajectory_noise.py -k 'not benchmark'
```

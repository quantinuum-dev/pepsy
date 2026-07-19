# `pepsy.optimizers.tree`

`TreeOptimizer` simulates a quantum circuit by replaying a canonical bundled
gate stream `[(gate, where), ...]` on a **rooted tree tensor network**, after
*Simulating quantum circuits using tree tensor networks* (Seitz, Medina, Cruz,
Huang, Mendl; Quantum 7, 964, 2023; [arXiv:2206.01000](https://arxiv.org/abs/2206.01000)).

The state is stored with one leaf tensor per qubit. Internal nodes may have
**any arity** -- the default is a strictly-binary tree, but flatter `k`-ary
trees or gate-connectivity-driven communities (see *Tree structure*) work
through the same machinery. Gates are absorbed into the tree:

- **single-qubit gates** are contracted into the leaf tensor with no bond
  growth; a unitary one-qubit gate preserves the tree canonical form regardless
  of where the orthogonality centre sits;
- **two-qubit gates** on leaves `a` and `b` are split by SVD into two factors
  joined by a virtual bond; the factors are absorbed into the two leaves and the
  bond is *threaded exactly* (lossless economical QR) along the tree path from
  `a` to `b`. Only once **both** factors are in place is a single canonical
  compression sweep run back along the path, truncating every touched bond to
  `chi` -- so each truncation sees the complete gate, markedly more accurate at
  finite `chi` than truncating each hop as the bond is threaded (Seitz et al.,
  Figs. 3-6).
- **operators on three or more qubits** -- a `k`-qubit gate (Toffoli, Fredkin),
  a multi-site non-unitary / Kraus operator, or a whole Trotter block -- are
  applied *in one shot* over their minimal spanning subtree by
  `apply_subtree_operator`; `apply_gate` routes any support with `len(where) >= 3`
  there automatically (see *Multi-qubit / sub-MPO application*).

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

## Range / subtree canonicalisation

The single orthogonality centre generalises to a connected **canonical region**
-- the tree analogue of an MPS mixed-canonical range. `canonical_region` is a
frozenset of node ids tracked on the `TreeTensorNetwork` alongside (in fact,
underlying) `orthogonality_center`, which is simply the one-node special case:
when the region spans more than one node `orthogonality_center` honestly reads
`None`. `TreeTensorNetwork.canonize_subtree_(nodes)` gauges every tensor
*outside* a connected subtree to point inward (Quimb `canonize_around` with
`which="any"`), so the whole state norm is carried by the region tensors --
contracting just the region against its conjugate reproduces the squared norm,
exactly as the single centre tensor does for a one-node region. Disconnected
`nodes` raise unless `span=True` auto-expands to the minimal connected subtree
that spans them (`subtree_span`). `canonize_around_qubits_(qubits)` is the
qubit-level entry point: it canonicalises around the minimal subtree spanning
those qubits' leaves, so the reduced state on a set of qubits is captured by one
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
only). The operator is contracted onto the **minimal connected subtree** (Steiner
subtree) spanning the target leaves and the result re-split back into that
subtree with truncating SVDs. The orthogonality centre is first moved onto a
target leaf so the whole exterior is isometric and every re-split truncation sees
the complete operator against an isometric environment; the centre is left inside
the subtree, so the state stays in canonical form.

`op` acts on `len(where)` qubits: an array reshaped to `(2,) * 2k` with output
indices first, `op[o_0..o_{k-1}, i_0..i_{k-1}]` (a `(2**k, 2**k)` matrix is
accepted). It need **not** be unitary; pass `renormalize=True` to renormalise
afterwards (e.g. after a Kraus/projection operator). `max_bond` / `cutoff`
default to the optimizer's `chi` / `cutoff`. `apply_gate` dispatches `len(where)
== 1` and `== 2` to the optimised leaf-absorb / geodesic-threading paths and any
larger support to `apply_subtree_operator`; the cost scales with the operator's
spread (the boundary of its spanning subtree) rather than the whole tree.

## Tree state class

`TreeTensorNetwork` is the tree analogue of Quimb's `MatrixProductState`: a
geometry-owning subclass of Quimb's arbitrary-geometry vector class
`quimb.tensor.TensorNetworkGenVector`. It *is* a Quimb tensor network, so all of
Quimb's arbitrary-geometry methods (`canonize_around`, `canonize_between`,
`compress_between`, `gate_inds`, `local_expectation`, `to_dense`, `copy`, ...)
apply directly; the class adds the naming and geometry glue on top of a
`TreePlan`:

- every node (leaf **and** internal) is one tensor tagged with the structural
  node tag `node_tag_id.format(nid)` (default `"N{}"`);
- leaf tensors additionally carry the Quimb site tag `site_tag_id.format(q)`
  (default `"I{}"`) and physical index `site_ind_id.format(q)` (default `"k{}"`)
  for qubit `q`, so the inherited site / `local_expectation` machinery treats the
  leaves as the sites;
- adjacent nodes share the deterministic virtual bond `_tb{lo}_{hi}`.

Because the geometry (`plan`) and naming live in `_EXTRA_PROPS`, they survive
`.copy()` and every Quimb view, exactly like `site_ind_id` does for an MPS.
Build one with `TreeTensorNetwork.from_plan(plan)` (product `|0...0>`),
`TreeTensorNetwork.from_order(order, structure=...)` (build the plan and the
product state in one step), or `TreeTensorNetwork.rand(plan, D=..., seed=...)`
(a random state, canonicalised around the root by default). `TreeOptimizer`
builds and evolves its state on this class, delegating all node/qubit naming and
geometry queries to it.

`TreeTensorNetwork.show()` prints a top-down ASCII drawing of the tree -- the
tree analogue of a quimb MPS `show()` -- with the root at the top and the qubit
leaves at the bottom, internal nodes marked `●`, leaves `◆` labelled by their
qubit, and every branch annotated with its current virtual bond dimension
(`ascii_tree()` returns the same drawing as a string).
`TreeOptimizer.show()` delegates to it.

## Tree structure

The tree structure is chosen by `TreeLayoutFinder`, which builds a weighted
interaction graph from the two-qubit supports of the gate stream and applies
recursive spectral (Fiedler) partition, keeping the recursion as the rooted
tree (`structure="quality"`). This reuses the interaction-graph and spectral
machinery of `pepsy.optimizers.mps.layout`; where the MPS finder flattens the
recursion into a 1D order, the tree finder keeps the tree. Strongly coupled
qubits become nearby leaves, minimising the tree-path length that two-qubit
gates thread across. `structure="balanced"` splits the qubit index order in
half at each level. `TreeLayoutFinder.score(plan)` returns the total
interaction-weighted tree-path length that the structure minimises.

The structure is **not restricted to binary trees**. Internal nodes may have
any arity, controlled by two knobs on `TreeLayoutFinder` / `TreePlan.from_order`
/ `TreeOptimizer`:

- `max_arity` caps the children per internal node. The default `2` reproduces
  the strictly-binary tree exactly; larger values give flatter `k`-ary trees
  with shorter geodesics; `None` leaves the arity unbounded.
- `structure="adaptive"` reads the gate-stream interaction graph and lets each
  level branch into as many children as it has strongly coupled communities
  (edges above `community_frac` times the level's strongest edge). A densely
  coupled block -- a near-clique with a present-strong-edge fraction of at
  least `star_frac` -- is collapsed into a single flat **star** node, so all
  its pairwise geodesics are length two instead of the up-to-`log2 m` of a
  bisection. Binary trees remain a valid special case (`max_arity=2`).

A caller may bypass the finder entirely by passing an explicit `TreePlan` via
`TreeOptimizer(..., tree=plan)`. Build one with
`TreePlan.from_order(order, weights=..., structure=..., max_arity=...)`, or -- for
a fully hand-specified arbitrary-arity tree -- with
`TreePlan.from_children(children, qubit_of_leaf)`, which validates that the
children map and leaf assignment describe a single rooted tree covering qubits
`0..n-1` exactly once. `TreePlan.max_arity()` and `TreePlan.is_binary()` report
the shape.

## Diagnostics

The dominant lever for accuracy at fixed `chi` is the tree structure, so the
finder and optimizer expose diagnostics to choose it:

- `TreeLayoutFinder.report(plan=None)` summarises the leaf-to-leaf geodesic
  lengths over the interaction graph (`score`, `max_path`, `mean_path`,
  `weighted_mean_path`) and compares against a balanced index tree
  (`balanced_score`, `score_ratio_vs_balanced`).
- `TreeOptimizer.bond_report()` reports the current `max_bond`, `mean_bond`,
  and tensor/bond counts -- bonds pinned at `chi` mean truncation is active.
- `TreeOptimizer.convergence_sweep(gates, n, chi_values, ops=...)` replays the
  stream at several `chi` on one fixed tree and returns per-`chi` `max_bond`,
  `norm`, observable `expectations`, `fidelity` against the untruncated state
  (when `2**n <= dense_cap`), and observable `max_drift` between consecutive
  `chi` -- a reference-free convergence signal for large systems.

## Readout

`to_dense()` returns the dense statevector in index order `k0, k1, ..., k(n-1)`.
`local_expectation(op, where)` returns `<psi|op|psi> / <psi|psi>`. A single-site
operator contracts only the canonical centre tensor; a multi-site operator
contracts only the **minimal subtree spanning the target leaves**, with the
centre moved inside that subtree so the isometric exterior cancels to the
identity on the shared boundary bonds (cost scales with the operator's spread,
not the whole tree). `measure(q, outcome=None)` projectively measures a qubit in
the computational basis -- reading the exact Born probabilities from the
centred leaf, sampling (or forcing) an outcome, and renormalising -- and
`reset(q)` returns a qubit to `|0>`. `normalize()` rescales the represented
state to unit norm and `max_bond()` reports the largest virtual bond.

## Performance and stability

- **Sibling fast path.** A two-qubit gate on two leaves that share a parent is
  applied as a single two-site update: the two leaves and their parent are
  contracted into one blob, the gate is applied, and the blob is re-split by
  two truncating SVDs against the (isometric) surrounding tree. This avoids the
  QR bond-threading and double-bond fusion of the general geodesic route and is
  the common case in a locality-aware layout.

- **Thread cap.** Tree tensors are small (rank `<= 3`, bounded by `chi`), so
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
  than recomputing it on the first gate.
- **State-owned centre.** The orthogonality centre lives on the
  `TreeTensorNetwork` (`orthogonality_center`, an `_EXTRA_PROPS` field), so the
  optimizer and the state cannot disagree and the centre is carried by
  `.copy()`. Incremental moves (`shift_orthogonality_center`) touch only the
  geodesic between old and new centre.
- **Self-healing tid cache.** Node-to-tensor lookups are cached and validated
  against the live tensor map, so the hot path avoids re-scanning tags while
  staying correct when a gate rebuilds a tensor.
- **`copy()`.** Returns an independent optimizer that shares the immutable
  `TreePlan` but owns its own tensor network (which carries the tracked
  orthogonality centre), for branching experiments or trial gate sequences.

```{eval-rst}
.. automodule:: pepsy.optimizers.tree.optimizer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pepsy.optimizers.tree.ttn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pepsy.optimizers.tree.layout
   :members:
   :undoc-members:
   :show-inheritance:
```

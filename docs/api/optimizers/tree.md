# `pepsy.optimizers.tree`

`TreeOptimizer` simulates a quantum circuit by replaying a canonical bundled
gate stream `[(gate, where), ...]` on a **rooted tree tensor network**, after
*Simulating quantum circuits using tree tensor networks* (Seitz, Medina, Cruz,
Huang, Mendl; Quantum 7, 964, 2023; [arXiv:2206.01000](https://arxiv.org/abs/2206.01000)).

The state is stored with one leaf tensor per qubit. Gates are absorbed into the
tree:

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

The orthogonality centre is tracked as a node id and moved along the tree
geodesic with per-edge canonicalisation (Quimb `canonize_between`), mirroring
the `info_c["cur_orthog"]` centre tracking of `MpsOptimizer`. When the centre is
unknown it is established once with Quimb `canonize_around`.

## Tree structure

The tree structure is chosen by `TreeLayoutFinder`, which builds a weighted
interaction graph from the two-qubit supports of the gate stream and applies
recursive spectral (Fiedler) bisection, keeping the bisection dendrogram as the
rooted binary tree (`structure="quality"`). This reuses the interaction-graph
and spectral machinery of `pepsy.optimizers.mps.layout`; where the MPS finder
flattens the recursion into a 1D order, the tree finder keeps the tree. Strongly
coupled qubits become nearby leaves, minimising the tree-path length that
two-qubit gates thread across. `structure="balanced"` splits the qubit index
order in half at each level. `TreeLayoutFinder.score(plan)` returns the total
interaction-weighted tree-path length that the structure minimises.

A caller may bypass the finder by passing an explicit `TreePlan` via
`TreeOptimizer(..., tree=plan)`; build one with
`TreePlan.from_order(order, weights=..., structure=...)`.

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
  orthogonality centre; the centre is declared there rather than recomputed on
  the first gate.
- **Self-healing tid cache.** Node-to-tensor lookups are cached and validated
  against the live tensor map, so the hot path avoids re-scanning tags while
  staying correct when a gate rebuilds a tensor.
- **`copy()`.** Returns an independent optimizer that shares the immutable
  `TreePlan` but owns its own tensor network and centre tracker, for branching
  experiments or trial gate sequences.

```{eval-rst}
.. automodule:: pepsy.optimizers.tree.optimizer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pepsy.optimizers.tree.layout
   :members:
   :undoc-members:
   :show-inheritance:
```

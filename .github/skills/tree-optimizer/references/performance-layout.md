# Tree optimizer performance and layout contract

This reference contains the lower-frequency details moved out of the main
Tree Optimizer skill so the upload-facing `SKILL.md` stays concise.

## Performance and stability

- **BLAS thread cap is the biggest performance lever.** Tree tensors are
  moderate-rank (set by local arity and an optional root physical leg, with
  dimensions bounded by `chi`), so multi-threaded BLAS/OpenMP is dominated by
  thread launch/sync overhead. `threads=1` is the default; gate
  application and heavy readouts run inside `self._thread_ctx()` using
  `threadpoolctl` when available. Only raise `threads` in a large-`chi` regime.
- The self-healing tid cache (`_nid_to_tid`, `_tid`) validates cached tensor
  ids against `self.tn.tensor_map`; a stale entry is recomputed safely.
- Dense path and subtree routing preserve each QR-produced Q tensor's
  `left_inds`. Canonical recovery therefore recognizes an already-isometric
  routed branch without repeating its decomposition. Native fermionic routing
  deliberately retains explicit graded QR recovery.
- `copy()` shares the immutable `TreePlan`, owns `self.tn.copy()`, resets the
  tid cache, and derives a deterministic child seed for an independent RNG.

## Layout (`TreeLayoutFinder` / `TreePlan`)

`TreeLayoutFinder` reuses the MPS interaction-graph plus recursive spectral
(Fiedler) partition machinery, but keeps the recursion as a rooted tree.
Strongly coupled qubits become nearby leaves, reducing the geodesic a two-qubit
gate must thread.

- Partition uses `_similarity_weights()` from Seitz Eq. 1:
  `s(qi,qj) = |G(qi) & G(qj)| + 1/(deg_i + deg_j)`.
- `score(plan)` uses pure `pair_weights` (weighted geodesic sum, lower is
  better); keep it separate from augmented partition similarity.
- `report(plan)` compares against an index-order `structure="balanced"` tree.
  Path-quality layouts should be no worse than balanced; congestion mode is
  selected by edge-load cost.
- `TreePlan.from_order(..., structure=...)` supports `quality` (spectral),
  `balanced` (direct order, useful for deterministic sibling tests), and
  `adaptive` (community/clique-driven arity).

### Non-binary trees

The data structures and algorithms are arity-agnostic. `max_arity=2` forces a
binary tree; `(2, 3, 4)` is the default candidate set for recommendations.
`structure="adaptive"` can make variable-arity communities or near-clique
stars. `TreePlan.from_children(...)` accepts validated hand-built trees.

`recommend_arities(...)` compares path or rank-aware congestion candidates and
reports virtual degree, edge load, and peak bond growth. `objective` supports
`path`, `congestion`, and `hybrid`; `weight_mode` supports `count`, `auto`,
`angle`, and `operator_schmidt`. `recommend_layered` and
`recommend_arities` may opt into bounded `refine="greedy"` or the separate,
optional seeded `search="nevergrad"` path. Defaults remain fast and
reproducible.

Layout hot paths must traverse only a support's Steiner subtree, and dense
gate Schmidt-rank caches must key on wire positions rather than global labels.
The same controls are exposed through `TreeLayoutFinder`, `TreeOptimizer`,
`find_tree_layout`, `convergence_sweep`, and `TreeTensorNetwork.from_order`.

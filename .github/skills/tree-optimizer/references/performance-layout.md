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
- **Native central-edge compression.** A native compression call receives a
  reduction hint separately from its destination tensor. For a proven
  one-sided reduction (`reduced="left"`), let `A` be the active endpoint and
  `B` the destination endpoint, with `B†B = I` on the non-shared legs. The
  implementation computes the lossless graded factorization
  `A = Q_A R_A`, then SVDs only `R_A = U S V†`. It installs
  `Q_A U` on `A` and absorbs `S V†` into `B`. Thus the expensive SVD scales
  with the active endpoint's QR carry and the live bond, rather than the
  full fused two-node tensor. The proof is structural:
  `can_skip_canonize(A, B, absorb="left")` must accept the destination's
  `left_inds` and aligned Symmray charge maps.
- If that one-sided proof is absent but `reduced=True`, both endpoints are
  QR-reduced and only the contracted `R_A L_B` core is SVD'd. Unknown hints
  retain the complete two-node graded SVD as a compatibility fallback. The
  truncating step always remains the explicit native block SVD with the
  configured `max_bond`, `cutoff`, and `cutoff_mode`; the native
  `stabilized=False` policy applies only to lossless QR.
- The one-sided path uses fresh intermediate QR/SVD bond names because the
  original live edge label is still present in `R_A` while that factor is
  decomposed. After both contractions, the new compressed bond is reindexed
  to the original live edge. Reusing the old label during the SVD creates a
  repeated index and can route the contraction into an unsupported Symmray
  hyperedge.
- A 6x6, 48-gate, chi=64, complex64 Torch-CPU calibration with 12 Torch/tree
  threads reduced Tree evolution from 127.77 s to 6.21 s. The same run's MPS
  evolution was 2.85 s, so the remaining Tree/MPS ratio was 2.18x. The
  pre-fix central SVDs were 4096x4096; after the QR reduction they were
  384x384. Layout planning (~26.6 s in that run) is setup cost and is not
  included in the evolution comparison. The remaining replay gap is mainly
  gate-update/threading/contraction work; use `profile=True` to inspect
  `update` envelopes and nested `edge_compress` events before changing path
  geometry or observable contractions.
- Dense path and subtree routing preserve each QR-produced Q tensor's
  `left_inds`. Canonical recovery therefore recognizes an already-isometric
  routed branch without repeating its decomposition or entering Quimb's dense
  canonicalization kernel. Path and subtree compression also reads that proof
  before selecting one-sided `reduced="left"` compression, avoiding the
  redundant reduction QR only when the destination tensor is proven
  isometric. Missing proofs fall back to two-sided reduction. Native Symmray
  routing preserves the same proof when its charge maps are aligned; native
  canonical recovery skips only that proven lossless QR, while truncating
  native compression remains an explicit graded SVD. The network derives
  orientation views directly from live tensors; do not cache a duplicate map
  in the optimizer.
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

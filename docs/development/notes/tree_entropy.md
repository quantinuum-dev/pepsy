# Tree entropy diagnostic — 2026-09-15

## Upstream audit

The installed upstream interfaces were inspected before adding the tree
diagnostic:

- Quimb `1.15.1.dev51+g2e99c793e`: `Tensor.entropy(left_inds, method="svd")`
  obtains squared singular values from a local tensor split. Its generic
  `array_svals` path is NumPy-oriented for this development build.
- Autoray `0.11.1.dev3+g1b476b305`: `linalg.svd`, `linalg.eigh`, `fuse`, and
  backend arithmetic dispatch correctly for NumPy and Torch arrays.
- Cotengra `0.8.3.dev7+g1d7fd333f`, Cotengrust `0.2.1`: no contraction path is
  required by this local spectrum diagnostic.
- Symmray `0.3.2.dev8+g6c6dd34b5`: native arrays expose sector-aware
  `svd(absorb=None)` and `svd_via_eig(absorb=None)`, returning a compact
  singular-value vector.
- Torch `2.6.0+cu124`: backend linalg remains selected through Autoray.

Disposition: **adopt** Quimb's private-copy canonicalization plus local
Schmidt-spectrum pattern. The Pepsy implementation uses Autoray linalg for
dense states and Symmray's native SVD for structured states; it does not
densify the TTN or alter the live canonical gauge. Cotengra and Cotengrust are
not involved.

## Pepsy contract

`TreeTensorNetwork.tree_edge_entropies()` returns one normalized base-2 entropy
per deterministic `(parent, child)` TreePlan edge. `entropy(edge)` measures one
edge, and `TreeOptimizer` exposes thin delegates. A single root canonicalization
is shared across the full edge scan, so the cost is one O(N) gauge preparation
plus one local spectrum calculation per edge. `return_edges=True` makes the
tree bipartition labels explicit.

Focused probes cover product and Bell states, Gram-matrix entropy, native U1
trees, Torch CPU trees, and live-state/gauge preservation. The full tree test
suite remains the required follow-up validation.

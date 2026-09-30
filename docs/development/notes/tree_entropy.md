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
edge, and `TreeOptimizer` exposes thin delegates. Root canonicalization and a
depth-first centre walk are shared across the edge scan, costing O(N) gauge moves
plus one local spectrum calculation per edge. `return_edges=True` makes the
tree bipartition labels explicit.

Focused probes cover product and Bell states, Gram-matrix entropy, native U1
trees, Torch CPU trees, and live-state/gauge preservation. The full tree test
suite remains the required follow-up validation.

## Physical spectrum correction — 2026-09-16

The initial product/Bell tests missed a physical error: tensors away from the
root centre are isometries and their singular values yield `log2(bond_dim)`,
not the entanglement entropy. Each local SVD now acts on the centre tensor.
The all-edge query walks a private copy depth first, crossing each bond at
most twice and retaining deterministic output ordering. TreeSampler keeps
the captured canonical tree and delegates to that calculation; source
mutation does not change its snapshot until `refresh()`.

The maintenance audit checked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and the
[Symmray repository](https://github.com/jcmgray/symmray).
The requested Symmray Abelian documentation endpoint was unavailable; the
installed source and callable signatures were inspected instead.

Installed versions: Quimb `1.15.1.dev46+g0ad529894`, Autoray
`0.11.1.dev1+gc56f64427`, Cotengra `0.8.3.dev7+g1d7fd333f`, Cotengrust `0.2.1`,
Symmray `0.3.2.dev7+gd63bb4e3f`, NumPy `2.4.4`, Torch `2.11.0`, CuPy `14.0.1`.

API probes confirmed `TensorNetwork.canonize_between(..., absorb='right',
**canonize_opts)`, `canonize_around(..., absorb='right', **canonize_opts)`,
`Tensor.transpose(*output_inds, inplace=False)`, and Symmray
`ArrayCommon.svd(**kwargs)`. Autoray resolves `linalg.svd`, `linalg.eigh`, and
`fuse` for NumPy, Torch, and CuPy. The existing native QR safeguard remains
unchanged. **Adopt** the existing public QR/SVD dispatch and centre-movement
APIs; **defer** newer Quimb randomized/cutoff defaults and unrelated Autoray,
Cotengra, and Symmray changes. No compatibility shim or dependency update is
needed: this was a Pepsy choice of tensor, not an upstream API regression.

New checks compare every physical cut of nonuniform complex states with a
full small-state SVD, including a physical root, both edge orientations,
NumPy/Torch/CuPy, native U1 states, and sampler snapshot/gauge preservation.

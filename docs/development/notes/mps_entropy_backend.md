# Native MPS and tree entropy backends — 2026-09-16

## Upstream audit

The active environment was inspected before changing the entropy diagnostic:

- Quimb `1.15.1.dev51+g2e99c793e`: `MatrixProductState.entropy(i, info=None,
  method='svd')`, `canonicalize(..., inplace=False)`,
  `schmidt_values(i, info=None, method='svd')`, and
  `Tensor.singular_values(left_inds, method='svd')` are available.
- Autoray `0.11.1.dev3+g1b476b305`: `do(fn, *args, like=None, **kwargs)`
  dispatches backend linalg, fusion, and scalar operations.
- Cotengra `0.8.3.dev7+g1d7fd333f`: no contraction path is needed by the
  local MPS or tree Schmidt-spectrum diagnostic.
- Symmray `0.3.2.dev8+g6c6dd34b5`: native arrays provide sector-aware `svd`
  and `svd_via_eig` spectrum paths.
- Torch `2.6.0+cu124`: `linalg.svdvals` supports the configured CUDA driver;
  Pepsy's `TorchLinalgConfig` remains the single linalg registration point.

The upstream changelogs and APIs were checked in the [Quimb
changelog](https://quimb.readthedocs.io/en/latest/changelog.html), [Autoray
repository](https://github.com/jcmgray/autoray), [Cotengra
documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray Abelian-array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
and [repository](https://github.com/jcmgray/symmray).

Disposition: **adopt** backend-native values-only SVD for ordinary dense Torch
paths, retain Pepsy's configured full SVD path for stabilized/custom policies,
and **compatibility shim** the scalar boundary through backend `.item()` before
Autoray's NumPy conversion. Symmray spectra continue to use its native SVD.
No installed upstream package is modified.

## Pepsy contract

`pepsy.tensors.mps_entanglement_entropy` measures a finite open MPS bond on a
private canonicalized copy. Dense Torch and CuPy arrays remain on their device;
only the final scalar reductions are read back. `MpsOptimizer.entropy`,
`MpsOptimizer.entanglement_entropy`, and `MpsSampler.entanglement_entropy`
delegate to this helper. Tree entropy uses the same values-only dense linalg
policy and the existing native Symmray path.

Focused coverage checks NumPy, Torch CPU/CUDA when available, CuPy when
available, the eigenvalue alias, cyclic/invalid inputs, optimizer and sampler
delegation, source/gauge preservation, and host-conversion guards.

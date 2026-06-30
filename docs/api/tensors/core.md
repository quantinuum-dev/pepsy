# `pepsy.tensors.core`

`reg_rel_svd_torch` is the preferred torch SVD registration for tensor-network
autodiff. It installs the relative-regularized SVD backward rule used by
`reg_complex_svd_torch`, and its CPU forward path falls back to SciPy `gesvd`
if `torch.linalg.svd` fails.
The SVD/QR registration helpers are also available directly from `pepsy`, e.g.
`import pepsy as py; py.reg_rel_svd_torch()`. Torch exposes
`reg_rel_svd_torch()`, `reg_real_svd_torch()`, `reg_complex_svd_torch()`,
`reg_real_qr_torch()`, and `reg_complex_qr_torch()`. JAX exposes SVD aliases
`reg_rel_svd_jax()`, `reg_real_svd_jax()`, and `reg_complex_svd_jax()` for the
same custom-VJP SVD registration.

```{eval-rst}
.. automodule:: pepsy.tensors.core
   :members:
   :undoc-members:
   :show-inheritance:
```

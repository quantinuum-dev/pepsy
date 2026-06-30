# `pepsy.tensors.core`

`reg_rel_svd_torch` is the preferred torch SVD registration for tensor-network
autodiff. It installs the relative-regularized SVD backward rule used by
`reg_complex_svd_torch`, and its CPU forward path falls back to SciPy `gesvd`
if `torch.linalg.svd` fails.
The registration helpers `reg_rel_svd_torch()`, `reg_complex_svd_torch()`,
and `reg_complex_svd_jax()` are also available directly from `pepsy`, e.g.
`import pepsy as py; py.reg_rel_svd_torch()`.

```{eval-rst}
.. automodule:: pepsy.tensors.core
   :members:
   :undoc-members:
   :show-inheritance:
```

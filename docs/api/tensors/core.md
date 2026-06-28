# `pepsy.tensors.core`

`reg_rel_svd_torch` is the preferred torch SVD registration for tensor-network
autodiff. It installs the relative-regularized SVD backward rule used by
`reg_complex_svd_torch`, and its CPU forward path falls back to SciPy `gesvd`
if `torch.linalg.svd` fails.

```{eval-rst}
.. automodule:: pepsy.tensors.core
   :members:
   :undoc-members:
   :show-inheritance:
```

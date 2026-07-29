# `pepsy.tensors.core`

`reg_rel_svd_torch` is the preferred torch SVD registration for tensor-network
autodiff. It installs the relative-regularized SVD backward rule used by
`reg_complex_svd_torch`, and its CPU forward path falls back to SciPy `gesvd`
if `torch.linalg.svd` fails.
The SVD/QR registration helpers are also available directly from `pepsy`, e.g.
`import pepsy as py; py.reg_rel_svd_torch()`. Torch exposes
`reg_rel_svd_torch()`, `reg_real_svd_torch()`, `reg_complex_svd_torch()`,
`reg_real_qr_torch()`, and `reg_complex_qr_torch()`. JAX exposes SVD aliases
`reg_rel_svd_jax()`, `reg_real_svd_jax()`, and `reg_complex_svd_jax()` for a
thin-SVD custom VJP that preserves JAX's native derivative while safely
restoring cotangents from Quimb fixed-rank truncation.


> API details are maintained as handwritten Markdown in this page.

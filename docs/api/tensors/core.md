# `pepsy.tensors.core`

`register_torch_linalg()` keeps native Torch SVD/QR as the default. Pass
`stabilized=True` to opt into Pepsy's relative-regularized SVD and validated
real-QR rules for tensor-network autodiff. The explicit
`reg_rel_svd_torch()` helper remains available when only the stabilized SVD
rule is wanted; its CPU forward path falls back to SciPy `gesvd` if
`torch.linalg.svd` fails.
Use `register_jax_linalg()` for the same native-versus-stabilized choice on
JAX, or call `reg_native_svd_torch()` / `reg_native_svd_jax()` directly.
The SVD/QR registration helpers are also available directly from `pepsy`, e.g.
`import pepsy as py; py.reg_rel_svd_torch()`. Torch exposes
`reg_native_svd_torch()`, `reg_rel_svd_torch()`, `reg_real_svd_torch()`,
`reg_complex_svd_torch()`,
`reg_real_qr_torch()`, and `reg_complex_qr_torch()`. JAX exposes SVD aliases
`reg_native_svd_jax()`, `reg_rel_svd_jax()`, `reg_real_svd_jax()`, and
`reg_complex_svd_jax()` for a
thin-SVD custom VJP that preserves JAX's native derivative while safely
restoring cotangents from Quimb fixed-rank truncation.
Registration helpers are idempotent: repeated calls do not re-register the
same Autoray implementation, while switching Torch between native/stabilized
or real/complex modes, and JAX between native/stabilized modes, intentionally
updates the active implementation. Stabilized real
QR supports square, tall, wide, and batched reduced QR. Its rank policy can be
`warn`, `native`, or `error`, and the tolerance is configurable. Complex mode
keeps native `torch.linalg.qr`; the explicit complex QR compatibility wrapper
uses the same conjugate-aware native VJP but is not registered because it
recomputes QR during backward.
Calling `reset_linalg_registrations()` restores native Torch/JAX mappings and
clears Pepsy's registration caches.


> API details are maintained as handwritten Markdown in this page.

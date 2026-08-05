# `pepsy.tensors.core`

Use one public setup call for a Torch-autodiff PEPS run:

```python
py.register_torch_linalg(
    mode="real", stabilized=True, quimb_split_drivers=True,
)
```

It configures both Autoray's ordinary Torch SVD/QR operations and Quimb's
raw-block split drivers. Set `quimb_split_drivers=False` (the default) for
ordinary Torch tensor work that does not use native Symmray PEPS boundary
autodiff. `PepsEnergyOptimizer` applies the same canonical configuration
automatically from its input dtype.

The Quimb split-driver registration is process-global. Passing
`quimb_split_drivers=False` does not undo a previous registration; use
`py.reset_linalg_registrations("torch")` to explicitly restore Quimb's and
Autoray's native Torch rules.

`stabilized=True` selects Pepsy's relative-regularized SVD and validated
real-QR rules. The lower-level `reg_*_torch()` helpers remain for advanced
compatibility use; ordinary applications should not combine them. The
stabilized SVD CPU forward path falls back to SciPy `gesvd` if
`torch.linalg.svd` fails.

The Torch QR split driver uses the zero-safe phase convention `phase(0)=1`.
This preserves a lossless QR/LQ reconstruction even for rank-deficient dense
or Symmray blocks. At a nonzero singular QR pivot its backward uses a
scale-relative regularized right inverse rather than discarding the entire
block gradient. An exactly zero block has no preferred QR gauge or scale and
therefore retains the explicit zero-VJP convention.

Use `register_jax_linalg()` for the same native-versus-stabilized choice on
JAX. The lower-level Torch/JAX helpers remain available directly from `pepsy`
for advanced compatibility workflows.
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

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
real/complex QR rules. Stabilized Autoray calls accept the normal thin-SVD
`full_matrices=False` and reduced-QR `mode="reduced"` keywords; full SVD and
complete QR are intentionally rejected because their custom VJPs are thin and
reduced only. A real policy rejects complex input rather than silently using a
real-only VJP. The lower-level `reg_*_torch()` helpers remain for advanced
compatibility use; ordinary applications should not combine them. The
stabilized SVD CPU forward path falls back to SciPy `gesvd` if
`torch.linalg.svd` fails.

The Torch QR split driver uses the zero-safe phase convention `phase(0)=1`.
This preserves a lossless QR/LQ reconstruction even for rank-deficient dense
or Symmray blocks. A pivot at or below the scale-relative QR epsilon uses a
regularized right inverse in backward rather than discarding the entire block
gradient. The same epsilon sets both the near-singular detection threshold and
the Tikhonov shift (`1e-6` times block scale for float64/complex128; a larger
float32 safety floor applies). An exactly zero block has no preferred QR gauge
or scale and therefore retains the explicit zero-VJP convention.

Use `register_jax_linalg()` for the same native-versus-stabilized choice on
JAX. The lower-level Torch/JAX helpers remain available directly from `pepsy`
for advanced compatibility workflows.
Registration helpers are idempotent: repeated calls do not re-register the
same Autoray implementation, while switching Torch between native/stabilized
or real/complex modes, and JAX between native/stabilized modes, intentionally
updates the active implementation. Stabilized real
QR supports square, tall, wide, and batched reduced QR. Its rank policy can be
`warn`, `native`, or `error`, and the tolerance is configurable. Stabilized
real and complex modes use the finite rank-aware QR VJPs; native mode keeps
Torch's direct QR implementation.
Calling `reset_linalg_registrations()` restores native Torch/JAX mappings and
clears Pepsy's registration caches.


> API details are maintained as handwritten Markdown in this page.

# `pepsy.sampling.samplers`

## MPS sampler quick API

`MpsSampler.sample_batch(...)` is the preferred batched interface for new code:

```python
sampler = MpsSampler(psi, one_d_to_two_d, backend="native")
batch = sampler.sample_batch(n_samples=4096, seed=0)

configs = batch.configs
probs = batch.probs
```

With dense Torch or CuPy MPS tensors, `backend="native"` keeps `configs` and
`probs` on the tensor device. Use `batch.to_numpy()` to copy to CPU NumPy, or
`batch.to_sample_result()` when the legacy list/grid `MpsSampleResult` is
needed. `sample_arrays(...)` remains available for direct tuple unpacking, and
`sample(...)` preserves the original `MpsSampleResult` behavior.
The native path builds backend-native right environments once, so it does not
require quimb to canonicalize Torch or CuPy tensors before sampling.
It caches those environments for the current MPS tensors: after modifying the
MPS, call `sampler.refresh()` before sampling again.

Torch sampling is inference-only by default, avoiding an autograd graph for
the discrete draw and its sampled probabilities. Use `track_grad=True` only
when those sampled Born probabilities need gradients:

```python
configs, probs = sampler.sample_arrays(1024, track_grad=True)
```

For repeated, device-resident, unseeded Torch batches, construct with
`torch_compile=True` to opt into `torch.compile`. If the installed compiler
cannot support the workload, the sampler automatically uses its eager path.
Seeded sampling and calls that convert results to NumPy always use eager mode.

For evaluating existing configurations, use:

```python
amps = sampler.amplitudes(configs, to_numpy=False)
probs = sampler.probabilities(configs, to_numpy=False)
```

Both methods evaluate the whole batch in one backend-native sweep.

```{eval-rst}
.. automodule:: pepsy.sampling.samplers
   :members:
   :undoc-members:
   :show-inheritance:
```

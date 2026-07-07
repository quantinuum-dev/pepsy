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

# `pepsy.MpsStabSampler`

`MpsStabSampler` samples the physical stabilizer-tensor-network state
`|psi> = C|nu>` without constructing a dense statevector. It accepts a live
`MpsStabOptimizer` or the pair `(C, nu)`, maps requested local X/Y/Z
measurements through the tableau frame, and samples the resulting Pauli
projectors on the coefficient MPS using shared-prefix branching.

```python
sampler = pepsy.MpsStabSampler(stabilizer_optimizer)

batch = sampler.sample_batch(4096, basis="random", seed=7, chunk_size=1024)
configs = batch.configs       # shape (4096, n)
probs = batch.probs            # exact Born probability of each config
resolved_basis = batch.basis   # one X/Y/Z label per qubit
```

For direct construction from a tableau and coefficient MPS:

```python
sampler = pepsy.MpsStabSampler(C, nu, backend="native")
configs, probs = sampler.sample_arrays(4096, chunk_size=1024)
```

With a Torch- or CuPy-backed coefficient MPS, `backend="native"` keeps the
returned `configs` and `probs` on that backend. Use `to_numpy=True` on
`sample_arrays` or `sample_batch` when CPU NumPy arrays are required.

`basis` may be a global `"X"`, `"Y"`, or `"Z"`, a length-`n` pattern such as
`"XYZX"`, or `"random"` for one random basis pattern shared by the batch.
`sample`, `sample_batch`, `sample_arrays`, `iter_samples`, and `probabilities`
follow the corresponding MPS sampler conventions. Sampling uses the
`frame_pauli` strategy by default and does not mutate the optimizer state.

For an explicit tableau `C` and coefficient MPS `nu`:

```python
sampler = pepsy.MpsStabSampler.from_tableau_and_state(tableau, nu)
```

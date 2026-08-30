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
probs = batch.probs            # p(config | resolved_basis)
resolved_basis = batch.basis   # one X/Y/Z label per qubit
```

For ``chi=None`` (the default), ``probs`` are exact Born probabilities up to
floating-point precision. With finite ``chi``, each conditional branch is
compressed using the optimizer settings, so ``probs`` are the probabilities of
the compressed conditional state. Use ``get_sampling_diagnostics()`` to
inspect the corresponding retained-norm and projector-loss signals. Sampling
is inference-only: ``track_grad=True`` is intentionally unsupported and raises
``NotImplementedError``.

For direct construction from a tableau and coefficient MPS:

```python
sampler = pepsy.MpsStabSampler(C, nu, backend="native")
configs, probs = sampler.sample_arrays(4096, chunk_size=1024)
```

Construction options for the coefficient-MPS optimizer can be supplied
directly when passing ``(C, nu)``. ``chi``, ``mode``, and ``cutoff`` then
control compression of copied measurement branches. Set ``disentangle=True``
to localize each measured frame Pauli to ``+/- Z_k`` and update the tableau on
that branch:

```python
sampler = pepsy.MpsStabSampler(
    C,
    nu,
    chi=16,
    mode="dmrg2",
    cutoff=1e-12,
    disentangle=True,
)
```

The default ``disentangle=False`` applies the frame projector directly as a
sub-MPO. The legacy ``absorb_basis`` keyword remains accepted as an alias.
With disentangling enabled, the frame changes independently on each
collapsed branch, so later frame images are recomputed for that branch. In
both modes the original optimizer, tableau, and coefficient MPS remain
unchanged; ``chi`` is taken from the underlying optimizer.

``probability_bits(bits)`` and ``probability_bits_many(bitstrings)`` use the
same chain-rule branch engine as sampling and do not mutate the optimizer.
``probabilities(configs, ...)`` is the batch-shaped convenience wrapper.

After a sampling call, inspect the temporary conditional branches with
``get_sampling_diagnostics()``:

```python
sampler.sample_batch(4096, seed=7, chunk_size=1024)
for record in sampler.get_sampling_diagnostics():
    print(
        record["qubit"],
        record["branch_probability"],  # Born probability, not fidelity
        record["local_infidelity"],    # conditional retained-norm loss
    )
```

Each record describes one non-final prefix branch. ``local_infidelity`` is
the total compression loss of that conditional update. Its components are
reported as ``localizer_infidelity`` and ``projector_infidelity``. In
``absorb_basis=True`` mode, ``compression_events`` contains the individual
localizing-Clifford truncations, each with its own ``local_infidelity``.
``norm_events`` contains the projected-branch event. Born probabilities are
kept separately in ``branch_probability``, ``prefix_probability``, and
``joint_probability``; they must not be interpreted as compression fidelity.
If a very small ``chi`` makes an absorbed localizer turn a sampled branch
into a numerical zero state, the sampler rolls that branch back and uses the
direct frame projector; ``condition_strategy`` records this as
``"frame_projector_fallback"``.
The diagnostics refer to the most recently sampled batch (or most recently
yielded chunk when using an iterator).

With a Torch- or CuPy-backed coefficient MPS, `backend="native"` keeps the
returned `configs` and `probs` on that backend. Use `to_numpy=True` on
`sample_arrays` or `sample_batch` when CPU NumPy arrays are required.

`basis` may be a global `"X"`, `"Y"`, or `"Z"`, a length-`n` pattern such as
`"XYZX"`, or `"random"` for one random basis pattern shared by the batch.
`sample`, `sample_batch`, `sample_arrays`, `iter_samples`, and `probabilities`
follow the corresponding MPS sampler conventions. Sampling uses the
`frame_pauli` strategy by default and does not mutate the optimizer state.
The currently supported strategy names are `"auto"` and `"frame"`; both select
the frame-projector implementation.

For an explicit tableau `C` and coefficient MPS `nu`:

```python
sampler = pepsy.MpsStabSampler.from_tableau_and_state(tableau, nu)
```

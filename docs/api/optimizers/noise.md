# `pepsy.optimizers.noise`

`PauliErrorModel` samples independent **physical Pauli trajectories**, not a
density matrix. Each non-identity X/Y/Z fault is inserted into a concrete gate
stream after every target of an ordinary gate. The resulting stream can be
replayed by either `MpsOptimizer` or `MpsStabOptimizer`; for STN, every sampled
fault is a Clifford that is absorbed by the Stim tableau.

```python
import pepsy

noise = pepsy.PauliErrorModel.depolarizing(1e-3)
result = pepsy.run_noisy_shots(
    lambda: pepsy.MpsStabOptimizer(6, chi=32),
    gates,
    noise,
    shots=1_000,
    seed=7,
)

# Each trajectory can be measured/read independently.
samples = [sim.sample_bits(100, seed=shot) for shot, sim in enumerate(result.optimizers)]
```

For ordinary MPS replay, use the same function with a factory that constructs a
fresh state for every shot:

```python
result = pepsy.run_noisy_shots(
    lambda: pepsy.MpsOptimizer(initial_mps, chi=64, mode="mpo"),
    gates,
    pepsy.PauliErrorModel.bit_flip(0.01),
    shots=100,
    seed=7,
)
```

`result.gate_streams` holds the sampled, replayable streams and `result.faults`
holds concise `(gate_index, site, pauli)` records. Use
`sample_noisy_gate_stream(...)` or `sample_noisy_gate_streams(...)` when only
stream construction is needed.

The current scope is Stim-native stochastic one-qubit Pauli channels:
depolarizing, bit flip, phase flip, bit-phase flip, or a custom
`PauliErrorModel(p_x=..., p_y=..., p_z=...)`. It intentionally does not claim
to simulate non-Pauli channels such as amplitude damping.

```{eval-rst}
.. automodule:: pepsy.optimizers.noise
   :members:
   :undoc-members:
```

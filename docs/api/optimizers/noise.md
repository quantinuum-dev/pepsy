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

## User-defined quantum trajectories

`TrajectoryEvent` is the general independent noise-simulation interface. Put
one directly inside an ordinary gate stream and run independently sampled shots
with either `MpsOptimizer` or `MpsStabOptimizer`. It does not require Stim or a
density matrix.

Use a `mixture` for a user-defined random-unitary channel. Its outcomes have
explicit probabilities, so `sample_trajectory_stream(...)` can make a concrete
noisy stream without an optimizer:

```python
import numpy as np
import pepsy

x = np.array([[0, 1], [1, 0]], dtype=complex)
bit_flip = pepsy.TrajectoryChannel.mixture([
    ("I", 0.99, np.eye(2)),
    ("X", 0.01, x),
])
stream = [
    (pepsy.h(), 0),
    pepsy.TrajectoryEvent(bit_flip, 0),
]
sample = pepsy.sample_trajectory_stream(stream, seed=7)
```

Use `kraus` when the branch probability must be computed from the evolving
state. Each selected branch is normalized before the later stream entries run;
this supports non-Pauli channels such as amplitude damping:

```python
stream = [
    (pepsy.x(), 0),
    pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.amplitude_damping(0.02), 0),
    (pepsy.h(), 0),
]
result = pepsy.run_trajectory_shots(
    lambda: pepsy.MpsStabOptimizer(1, chi=32),
    stream,
    shots=10_000,
    seed=7,
)

# One named result per noise event in each shot.
print(result.records[0])
```

`TrajectoryChannel.kraus([("no_jump", K0), ("jump", K1)])` accepts any
complete local qubit channel (`sum(K.conj().T @ K) == I`) on the corresponding
one- or multi-qubit `TrajectoryEvent` support. For ordinary MPS replay, replace
the factory above with a fresh `MpsOptimizer(initial_mps, ...)` and pass its
usual options through `run_kwargs`.

## Reading a Stim circuit

`compile_stim_circuit(...)` accepts `stim.Circuit` or Stim source text. It
compiles one- and two-qubit Clifford gates, Pauli measurements/resets, and
**every native Stim stochastic error instruction**, then reuses that plan for
all shots:

```python
circuit = """
H 0
CX 0 1
PAULI_CHANNEL_2(0,0,0, 0,0.01,0,0, 0,0,0,0, 0,0,0,0) 0 1
HERALDED_PAULI_CHANNEL_1(0, 0, 0, 0.02) 0
"""

result = pepsy.run_stim_shots(
    lambda: pepsy.MpsStabOptimizer(2), circuit, shots=10_000, seed=7,
)
print(result.faults[0])
print(result.heralds[0])
```

Supported Stim error channels are `X_ERROR`, `Y_ERROR`, `Z_ERROR`,
`DEPOLARIZE1`, `DEPOLARIZE2`, `PAULI_CHANNEL_1`, `PAULI_CHANNEL_2`,
`CORRELATED_ERROR`/`E`, `ELSE_CORRELATED_ERROR`, `HERALDED_ERASE`,
`HERALDED_PAULI_CHANNEL_1`, `I_ERROR`, and `II_ERROR`. Herald bits are retained
in the sample result; detector/observable annotations are intentionally left to
Stim or a decoder, since they do not affect the quantum trajectory. Stim itself
only represents Pauli noise, so amplitude damping is not a missing Stim channel.

```{eval-rst}
.. automodule:: pepsy.optimizers.noise
   :members:
   :undoc-members:
```

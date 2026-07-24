# `pepsy.optimizers.noise`

Pepsy's native noise design is **stream-local**: put stochastic entries directly
where the hardware schedule says the channel acts, then choose trajectory
sampling settings (`shots`, `seed`, independent/coalesced replay, and
`run_kwargs`) at the runner.

```python
stream = [
    ("h", 0),
    ("x_error", 1e-4, 0),
    ("cnot", 0, 1),
    ("depolarize2", 1e-3, 0, 1),
    ("t", 0),
    ("pauli_channel1", {"z": 2e-4}, 0),
]

result = pepsy.run_coalesced_trajectory_shots(
    lambda: pepsy.MpsStabOptimizer(2, chi=64),
    stream,
    shots=10_000,
    seed=7,
)
```

Equivalently, select the sampling strategy on the trajectory runner:

```python
result = pepsy.run_trajectory_shots(
    lambda: pepsy.MpsStabOptimizer(2, chi=64),
    stream,
    shots=10_000,
    seed=7,
    strategy="coalesced",   # or "independent" / "auto"
    max_branches=256,
)
```

Supported first-cut stochastic entries are:

- `("x_error", p, q)`, `("y_error", p, q)`, `("z_error", p, q)`
- `("depolarize1", p, q)`, `("depolarize2", p, q0, q1)`
- `("pauli_channel1", probs, q)`, where `probs` is `(p_x, p_y, p_z)` or a mapping
- `("pauli_channel2", probs, q0, q1)`, using Stim's 15 non-identity two-qubit Pauli labels
- `("amplitude_damping", gamma, q)`, sampled with the state-dependent trajectory runner

Stateful leakage entries are also Pepsy-native trajectory events:

- `("leakage", p, q)` or `("leak", p, q)` marks `q` leaked with probability `p`
- `("leakage_return", p, q)`, `("seepage", p, q)`, or `("unleak", p, q)`
  returns an already leaked qubit to a random computational-basis branch with
  probability `p`
- `("measure_leaked", q)` records a ternary PECOS/Selene-style result:
  `0` or `1` for a normal computational-basis measurement, and `2` when the
  trajectory state knows the qubit is leaked
- `("leak2depolar", enabled)` makes later `("leakage", p, q)` events use a
  full one-qubit depolarizing replacement instead of marking leakage
- `("leakage_depolarize", p, q)` applies that depolarizing replacement for a
  single event regardless of the current `leak2depolar` mode

Leakage state is carried per shot outside the qubit MPS. While a qubit is
leaked, ordinary gates touching it are suppressed. `reset` and `measure_reset`
clear the leakage flag; `measure_reset` first records the leaked-qubit
measurement as bit `1`. The sampled diagnostics live in
`TrajectoryShotResult.leakage_records` as `LeakageRecord` objects. Because this
state changes which later gates are replayed, leakage entries currently use
independent trajectories; `run_trajectory_shots(..., strategy="auto")` stays
independent, and explicit `strategy="coalesced"` raises for leakage streams.

`PauliErrorModel` remains a convenience macro for clean deterministic streams.
It samples independent **physical Pauli trajectories**, not a density matrix.
Each non-identity X/Y/Z fault is inserted into a concrete gate stream after every
target of an ordinary gate. The resulting stream can be replayed by either
`MpsOptimizer` or `MpsStabOptimizer`; for STN, every sampled fault is a Clifford
that is absorbed by the Stim tableau. Do not mix this macro with stream-local
stochastic entries; use `run_trajectory_shots(...)` or
`run_coalesced_trajectory_shots(...)` when the stream already contains noise.

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

## Exact coalesced ensembles for rare noise

When the total fault rate is small, avoid replaying the same no-error prefix
once per shot. `run_coalesced_noisy_shots(...)` holds one optimizer state per
distinct sampled branch and its number of represented shots. It runs an ideal
prefix once, samples exact multinomial branch counts at each Pauli channel,
and copies an MPS only when two nonempty branches genuinely diverge:

```python
result = pepsy.run_coalesced_noisy_shots(
    lambda: pepsy.MpsStabOptimizer(6, chi=32),
    gates,
    pepsy.PauliErrorModel.depolarizing(1e-3),
    shots=100_000,
    seed=7,
)

assert sum(result.counts) == 100_000
for leaf in result.leaves:
    print(leaf.count, leaf.faults)
```

The represented samples are still independent draws; only their identical
state evolution is shared. `run_coalesced_trajectory_shots(...)` provides the
same exact tree for `TrajectoryEvent` mixtures and state-dependent Kraus
channels. It also branches mid-circuit `measure`, `reset`, and
`measure_reset` controls with exact binomial counts, which is useful for
ancilla-based circuits. Its leaf `measurements` records the selected outcomes.

This is normally more useful than `torch.vmap` for rare faults: after a fault
or collapse, states have different tensor data and often different bond
profiles, while a coalesced no-error group stays one ordinary MPS/STN replay.
For terminal readout, call `result.sample_bits(...)` (or
`sample_coalesced_bits(result, ...)`). It invokes one batched `MpsSampler`
call per ordinary-MPS leaf and the STN tree sampler per STN leaf, returning
only terminal rows plus the source `leaf_indices`—never one optimizer per row:

```python
samples = result.sample_bits(seed=8)
assert samples.shots == result.shots
# samples.configs: (shots, n) computational-basis rows
# samples.leaf_indices: source coalesced leaf for each row
```

### Conservative automatic strategy

`run_noisy_shots(...)` keeps its backward-compatible independent replay by
default. For ordinary Pauli gate streams, `strategy="auto"` selects exact
count coalescing only when the expected per-shot number of non-identity faults
is small:

```python
result = pepsy.run_noisy_shots(
    factory,
    gates,
    pepsy.PauliErrorModel.depolarizing(1e-3),
    shots=512,
    strategy="auto",
    max_branches=128,
)
```

The automatic threshold is `lambda = (# noisy gate targets) *
(p_x + p_y + p_z) <= 0.1`. This is deliberately conservative: coalescing is
strongest when most shots take the no-fault path. An unforced `measure`,
`reset`, or `measure_reset` makes the policy choose independent trajectories,
because its physical collapse branches can dominate even when noise is rare.

The live-leaf cap is exact safety control, not truncation. If a selected
coalesced run would retain more than `max_branches`, automatic mode discards
the partial tree and restarts the whole ensemble independently. Pass
`strategy="coalesced"` to request coalescing explicitly; its same cap raises
instead of silently changing strategy. `auto_max_expected_faults` can tune the
default `0.1` threshold when profiling a different workload.

## User-defined quantum trajectories

`TrajectoryEvent` is the general independent noise-simulation interface. Put
one directly inside an ordinary gate stream and run independently sampled shots
with `MpsOptimizer`, `TreeOptimizer`, or `MpsStabOptimizer`. It does not require
Stim or a density matrix.

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
one- or multi-qubit `TrajectoryEvent` support. For ordinary MPS or TTN replay,
replace the factory above with a fresh `MpsOptimizer(initial_mps, ...)` or
`TreeOptimizer(...)` and pass its usual options through `run_kwargs`.

For `MpsStabOptimizer(track_infidelity=True)`, a selected Kraus outcome is a
normalized trajectory boundary, just like a measurement/reset: its Born weight
is retained in the trajectory record but is not treated as compression loss.
`sim.norm_diagnostics()["norm"]` is the square root of the product of all
completed/current segment survivals, so it remains meaningful after state
renormalization. The older `total_norm_proxy` key remains as a compatibility
alias.

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

`run_coalesced_stim_shots(...)` has the same output shape as the coalesced
ordinary trajectory runner and supports the complete compiled native Stim
noise set, including two-qubit, heralded, and `E`/`ELSE_CORRELATED_ERROR`
chains. It shares all ideal segments and records per-leaf Pauli faults and
herald bits.

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

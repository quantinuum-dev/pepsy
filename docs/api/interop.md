# `pepsy.interop`

## Guppy circuits

Guppy definitions can be compiled to a Pepsy-compatible static stream when
they contain a straight-line sequence of standard quantum operations:

```python
from guppylang import guppy
from guppylang.std.quantum import cx, h, qubit
from pepsy import guppy_gate_stream, MpsStabOptimizer

@guppy
def circuit() -> None:
    q0 = qubit()
    q1 = qubit()
    h(q0)
    cx(q0, q1)
    q0.discard()
    q1.discard()

stream = guppy_gate_stream(circuit)
sim = MpsStabOptimizer(stream.n_qubits, stream).run()
```

The default `format="matrix"` is accepted by dense MPS, MPO, PEPS, and
stabilizer-MPS replay. `format="named"` uses Pepsy's named entries where
possible and is useful when inspecting a stabilizer stream. The returned
`GuppyGateStream` is a list, with `n_qubits`, `initial_bits`, and
`measurements` metadata attached.

`angle(x)` in Guppy is converted from half-turn units to radians. Guppy
`MeasureFree` becomes a Pepsy Z-measure event. The fixed-site MPS is retained,
because deleting a measured qubit requires an explicit Pepsy `cap` policy.

The adapter rejects branches, loops, calls, dynamic qubit allocation, dynamic
rotation angles, and unsupported HUGR operations. Such programs need a
trajectory/control-flow executor or should be decomposed into a static circuit
before conversion. It also emits ordinary dense qubit gates; it does not infer
native fermionic or U(1)/U(1)×U(1)/Z2 charge metadata. Native Symmray streams
must still be constructed with Pepsy's symmetry-aware gate builders.

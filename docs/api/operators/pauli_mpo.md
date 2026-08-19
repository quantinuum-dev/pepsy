# Pauli-basis MPO

`PauliMPO` is Pepsy's sparse qubit-operator front end for the four local
Pauli basis elements `I`, `X`, `Y`, and `Z`. Exact Pauli algebra remains in the
sparse basis; general tensor-network work is compiled through Pepsy's existing
`FirstDegreeMPO` and Quimb MPO boundary.

```python
from pepsy.operators import PauliMPO

op = PauliMPO.from_terms(
    3,
    [
        (0.35, "ZIZ"),  # one full-chain term
        (-1.0, "ZZ"),   # translated nearest-neighbor sum
        (1.5, "X"),     # translated onsite sum
    ],
)
```

Short strings are translated over the chain. Use `boundary="periodic"` for
wrapped translations, or explicit support when a single placement is wanted:

```python
op = PauliMPO.from_terms(
    8,
    [(0.2, (0, 7), "ZZ")],
    boundary="periodic",
)
```

The exact Pauli-native operations are:

```python
op.trace()
op.dagger()
op.conjugate()
op.transpose()
op @ other
op.commutator(other)
op.partial_trace((0, 1))
op.norm()
```

Canonicalization combines equal Pauli words, orders them deterministically,
and can prune small coefficients without leaving the Pauli basis:

```python
op = op.canonicalize(atol=1.0e-12, rtol=1.0e-10)
```

For MPO-style bond canonicalization and compression while retaining Pauli
physical legs, use the native coefficient tensor-train boundary:

```python
canonical = op.canonicalize_native(center=op.nsites - 1)
compressed, report = op.compress_pauli(
    max_bond=64,
    cutoff=1.0e-10,
    form="right",  # also "left" or an integer center
    return_report=True,
)

assert all(core.shape[2] == 4 for core in compressed.to_pauli_cores())
print(report.final_bond_dimensions, report.discarded_singular_weight)
```

The native cores have shape `(left_bond, right_bond, 4)` with physical order
`I, X, Y, Z`. QR canonicalization and SVD rounding act on the coefficient
tensor train, never on a dense Hilbert-space matrix. The result remains a
`PauliMPO`; `to_mpo()` is only the execution boundary. The convenience form
`op.compress(basis="native", ...)` is equivalent to `compress_pauli`, while
the default `op.compress(...)` continues to return a numerically compressed
Quimb MPO. Compression follows Quimb's `max_bond`, `cutoff`,
`cutoff_mode` (`rel`, `abs`, `sum1`, `rsum1`, `sum2`, `rsum2`), `form`, and
`renorm` conventions for the native SVD backend. The native canonical forms
are `"right"`, `"left"`, and an integer center. `form="flat"` performs
disjoint left and right sweeps without promising a canonical center. The
native backend supports full (`"svd"`), randomized (`"rsvd"`), and
iterative-style (`"svds"`/`"isvd"`) methods. Absolute and sum cutoffs are
converted for the Hilbert-Schmidt normalization of the physical Pauli
operators, so rank selection agrees with the compiled Quimb MPO.

Small dense local gates can be brought into the same basis automatically. The
decomposition enumerates `4**k` local strings for a `k`-qubit gate, so this is
intended for one- and few-qubit gates:

```python
gate_op = PauliMPO.from_dense(gate)
# or inspect the local expansion directly
from pepsy.operators import decompose_pauli
terms = decompose_pauli(gate)
```

`apply_gate` transforms an existing Pauli expansion locally and combines the
result back into canonical Pauli strings. It supports non-contiguous sites:

```python
rotated = op.apply_gate(gate, where=(2, 5), mode="conjugate")
heisenberg = op.apply_gate(gate, where=2, mode="heisenberg")
left_product = op.apply_gate(gate, where=2, mode="left")
```

The modes mean `G P G†`, `G† P G`, `G P`, and `P G`, respectively. Kraus maps
are available through `apply_channel`, with either Heisenberg or Schrödinger
picture:

```python
observable_after_channel = op.apply_channel(kraus, where=2)
operator_after_channel = op.apply_channel(
    kraus, where=2, picture="schrodinger"
)
```

MPS and MPO operations use the established tensor-network boundary:

```python
value = op.expectation(mps)
new_mps = op.apply(mps, max_bond=128, cutoff=1.0e-10)
mpo = op.to_mpo()
compressed, report = op.compress(max_bond=64, return_report=True)
```

Numerical compression returns a Quimb `MatrixProductOperator`, because a
generic low-rank MPO does not remain sparse in the Pauli basis. Similarly,
`exp()` and `time_evolution()` return the existing semantic/numerical MPO
objects rather than claiming that a generally dense exponential is a sparse
Pauli expansion.

`partial_trace` traces out the supplied sites by default. Pass `keep=True` to
interpret the supplied sites as the sites retained in the reduced operator.
The default is the ordinary unnormalized partial trace; use
`normalized=True` for normalized Pauli coefficients.

The current implementation targets finite one-dimensional qubit MPOs. A
two-dimensional square-lattice operator should use a PEPO or a deliberate
2D-to-1D mapping; the Pauli basis does not remove the associated contraction
cost. Native 2D Pauli-PEPO support is intentionally a separate follow-up from
this mature 1D PauliMPO boundary.

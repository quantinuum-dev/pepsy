# `pepsy.optimizers.stabilizer_tn`

The Stabilizer Tensor Network (STN) simulator (Masot-Llima & Garcia-Saez, PRL
133, 230601, 2024; arXiv:2403.08724). A state is stored as `|psi> = C |nu>`: a
stim tableau Clifford `C` (the stabilizer basis `B(S, D)`) times a coefficient
MPS `|nu>` (the paper's coefficient state, exposed as `.p`). Clifford gates
update only the tableau (free, `|nu>` unchanged); non-Clifford gates and
measurements update `|nu>`.

`MpsStabOptimizer` is an `MpsOptimizer`-style gate-stream simulator and is
available at top level as `pepsy.MpsStabOptimizer` (state container:
`pepsy.STNState`). Supported gate-stream entries include Clifford gates,
non-Clifford Pauli rotations, `("t", q)` / `("tdg", q)`, explicit `(matrix,
where)` gates, `("submpo", mpo, where)` events, `("measure", pauli, where[,
outcome[, absorb_basis]])`, and `("reset", where)`.

## Measurement, reset, and magic-state injection

- `measure(pauli, where, *, outcome=None, absorb_basis=False)` — fixed-basis
  projector `(I +- M)/2` by default; `absorb_basis=True` uses the basis-updating
  (canonical Lemma-3) form that disentangles the measured qubit from `|nu>`.
- `reset(where)` — return qubit(s) to `|0>` (measure-`Z` absorb + conditional
  `X`), disentangling them so ancillas can be recycled.
- `prepare_magic(ancilla, *, angle=pi/4)` + `inject_rz(data, ancilla, phi)`
  (with `inject_t` / `inject_tdg` wrappers) — apply `Rz(phi)` by magic-state gate
  teleportation for `phi` a multiple of `pi/4` (Clifford correction), keeping the
  non-Clifford cost on the pre-loaded ancilla.

## Scalable sampling

- `sample_bits(shots, *, seed=None)` — computational-basis bitstrings sampled by
  the chain rule (sequential `Z`-measurement), avoiding an `O(2**n)` statevector.
- `probability_bits(bits)` — `|<bits|psi>|**2` as a product of conditional Born
  probabilities.

```{eval-rst}
.. automodule:: pepsy.optimizers.stabilizer_tn.mps_stab_optimizer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pepsy.optimizers.stabilizer_tn.stn_state
   :members:
   :undoc-members:
   :show-inheritance:
```

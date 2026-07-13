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
- `run_with_injection(gates, *, ancillas, ...)` and the `with_injection(n_data,
  gates, *, n_ancilla=1, ...)` classmethod — a circuit-rewrite front end that
  auto-teleports every `t` / `tdg` / `pi/4`-`rz` in a stream through `inject_rz`,
  recycling the ancilla pool (a single ancilla suffices). It picks the nearest
  clean ancilla to each data qubit, so a spread pool shortens the localizer span.
  Arbitrary (non-`pi/4`) `rz` angles stay on the exact rotation path (injecting
  them has no scaling benefit; compile to Clifford+T to inject).

## Scalable sampling

- `sample_bits(shots, *, seed=None)` — computational-basis bitstrings by **perfect
  (tree) sampling**: shots sharing a measured prefix share the collapsed state, so
  the collapse work is done once per distinct prefix, not per shot (a large saving
  for structured/low-rank `|nu>`). No `O(2**n)` statevector is formed.
- `probability_bits(bits)` — `|<bits|psi>|**2` as a product of conditional Born
  probabilities.

## Correctness and failure semantics

- Bitstrings contain exactly binary integer values, Pauli supports use distinct
  in-range sites, and forced measurement outcomes are exactly `+1` or `-1`.
- Impossible postselection raises `ValueError` before changing the tableau,
  coefficient MPS, measurement log, or diagnostics.
- `run()` removes successfully applied entries from its queue. If a later entry
  fails, that entry and the remaining suffix stay queued, so retrying does not
  replay the successful prefix.
- `norm()` and normalized measurements account for Quimb's separate MPS
  `exponent` scale as well as the tensor data.

## Performance boundaries

Tree sampling shares work for repeated prefixes, but high-entropy states can
still generate a number of live branches proportional to the shot count.
Arbitrary dense `k`-qubit matrices use a `4**k` Pauli decomposition and are
therefore intended for few-qubit gates. Multi-qubit Clifford-angle Pauli
rotations currently build a dense tableau on each use. Prefer named Clifford
gates, Pauli rotations with small support, and structured sub-MPOs for larger
operators. `track_infidelity=True` performs additional exact-reference overlaps
and is a diagnostic mode rather than the fastest execution path.

## Backends (GPU / torch / JAX)

Pass `to_backend=` (e.g. `pepsy.backend_torch(dtype=torch.complex128,
device="cuda")`, `pepsy.backend_cupy(...)`, `pepsy.backend_jax(...)`) to the
constructor or `with_injection`.  The coefficient MPS `|nu>` and every gate/MPO
applied to it are then placed on that backend, so the heavy MPS contractions
(SVD, `swap+split`, sub-MPO application) run on GPU/torch/JAX.  The stim tableau
(classical Clifford tracking) stays on the CPU.  Constant gate matrices are
cached per backend; expectation/fidelity scalars are converted back to Python
floats.  `to_statevector` / `amplitude` bring `|nu>` back to NumPy and are for
small-`n` validation only.

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

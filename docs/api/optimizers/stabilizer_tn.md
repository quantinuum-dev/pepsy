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
outcome[, absorb_basis]])`, `("reset", where[, basis])`,
`("measure_reset", basis, where[, outcome[, absorb_basis]])`, and guarded
physical cap events `("cap", where, vec[, absorb])`.

You can initialize from an ordinary computational-basis qubit MPS directly:
`MpsStabOptimizer(p)` or `MpsStabOptimizer.from_mps(p)` wraps `p` with the
identity tableau, so initially `C = I` and `|psi> = |p>`. Use this for an
already-prepared MPS ground state. Pass `inplace=False` to copy the supplied
MPS before evolution.

`disentangle_cliffords(sweeps=1, *, bonds=None, tol=None)` is an optional
Clifford-gauge optimization: it applies a local coefficient-frame Clifford
`D`, then absorbs `D^dagger` into the tableau, so `(C D^dagger)(D |nu>) = C
|nu>`. It can also be placed at an exact point in a stream as
`("disentangle",)` or `("disentangle", {"sweeps": 1, "bonds": ..., "tol": ...})`.
It preserves the represented physical state up to its explicitly selected
numerical cutoff, changes no infidelity samples, and records a bond-history
point.

`apply_layout("auto")` installs a **static STN frame layout** before replay:
it dry-runs the queued Clifford/basis-update skeleton, collects the expensive
coefficient-frame supports `C^dagger O C`, and chooses an MPS site order that
keeps those supports short. This is intentionally different from
`MpsOptimizer`'s physical gate-stream layout. The operation is exact and cheap
only while the coefficient MPS is product (`state.max_bond() == 1`); apply it
before non-Clifford evolution entangles `|nu>`. The constructor accepts
`layout="auto"` / `layout_kwargs={...}`, and `current_frame_layout(...)`
returns the plan without mutating the simulator.

## Measurement, reset, and magic-state injection

- `measure(pauli, where, *, outcome=None, absorb_basis=False)` — fixed-basis
  projector `(I +- M)/2` by default; `absorb_basis=True` uses the basis-updating
  (canonical Lemma-3) form that disentangles the measured qubit from `|nu>`.
- `reset(where)` — return qubit(s) to `|0>` (measure-`Z` absorb + conditional
  `X`), disentangling them so ancillas can be recycled. Pass `basis="X"` or
  `"Y"` to reset to the corresponding `+1` Pauli eigenstate.
- `measure_reset(pauli, where, *, outcome=None, absorb_basis=True)` — record a
  Pauli measurement, then reset the same qubit(s) to the `+1` eigenstate of
  that basis. Stream aliases `mrx`, `mry`, and `mrz` are accepted.
- `cap(where, vec, *, absorb="left")` — contract a physical qubit with `vec`
  and rebuild an `(n - 1)`-qubit identity-frame STN. This is a dense fallback
  guarded by `max_dense_cap_qubits`; for scalable DEM-style capping, use the
  structured weighted-XOR/coin stream rather than generic physical cap.
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
- Dense-operator coefficient pruning is controlled by `operator_tol`, never by
  the MPS SVD `cutoff`. With `operator_tol=None`, the threshold is relative to
  the matrix scale and input dtype; an explicit value is an absolute tolerance.
- Fallback Pauli decomposition is limited by
  `max_pauli_decomposition_qubits=2` before its `4**k` enumeration begins.
  Set a larger integer, or `None`, only when accepting that cost explicitly.
  Clifford matrices and one-qubit unitaries use specialized paths and bypass
  this fallback limit. After decomposition and frame mapping, sparse Pauli sums
  with at most four product terms are applied as one exact coefficient-frame
  sub-MPO with MPO bond dimension at most four; denser sums use the balanced
  branch-sum reducer.
- Physical cap events are limited by `max_dense_cap_qubits=10` because they
  reconstruct a dense statevector and then rebuild a coefficient MPS. Set a
  larger integer, or `None`, only when accepting that exponential cost.
- A zero operator produces a valid zero-norm MPS. `norm()`, dense amplitudes,
  and dense probabilities remain available, while expectation, sampling,
  conditional-probability, reset, and measurement APIs raise `ValueError`
  because normalized probabilities are undefined for a zero state.

## Performance boundaries

Tree sampling shares work for repeated prefixes, but high-entropy states can
still generate a number of live branches proportional to the shot count.
Fallback dense matrices use a `4**k` Pauli decomposition and are limited to two
qubits by default. Sparse results such as `I + XX`, `I + YY`, `I + ZZ`, or
small mixtures of those terms are applied as a single exact sub-MPO; dense
results are combined with a balanced, streaming MPS sum. This improves
reduction depth but does not remove the exponential number of candidate Paulis.
For larger physical-frame operators, decompose into named gates or Pauli
rotations. A `submpo` event is appropriate only when the MPO is already
expressed in the coefficient frame. Clifford-angle Pauli rotations are
synthesized directly as linear-size Stim basis-change and parity circuits, then
cached; they do not form a `2**k x 2**k` dense matrix. Prefer structured
coefficient-frame sub-MPOs where applicable. `track_infidelity=True` performs
no reference-state copy or overlap contraction. For normalized unitary
evolution it records the cumulative proxy `1 - ||nu||**2` after compressed
coefficient-MPS updates, reading the norm from the tracked one-site canonical
centre. Unitary updates are not renormalized, so lost norm remains visible.
Projectors, measurements, coefficient-frame sub-MPOs, and arbitrary
non-unitary matrices do not emit infidelity samples; an unnormalized
non-unitary map also suspends later samples until projection restores a
normalized baseline. The public `infidelities` trace is therefore sparse and
historical, is not index-aligned with `bond_history`, and must not be summed.
Projective measurement/reset boundaries do preserve the current segment before
normalization in `norm_events`: the event records the pre-collapse norm,
`1 - ||nu||**2` for that segment, the Born branch probability separately, and
the actual post-projector norm before renormalization. Comparing the actual
post-projector norm with `pre_norm**2 * branch_probability` gives a separate
`projector_infidelity` proxy for compression done while applying the projector.
The post-collapse state is then normalized. Use `norm_diagnostics()` to form
product/geometric-mean survival summaries across completed segments plus the
current open segment; these summaries multiply unitary- and projector-
compression survival factors, but not measurement probabilities.
The proxy is not exact overlap fidelity or a discarded-SVD-weight report;
validate physical accuracy independently when that distinction matters.

The disentangling sweep scores the local Schmidt spectrum for the 20 two-qubit
Clifford classes modulo output-local Cliffords, rather than brute-forcing all
11,520 two-qubit Cliffords or copying the full MPS per candidate. Its `tol`
controls both the relative numerical-rank decision and the local SVD split:
use `tol=0` to retain every numerical singular value, or the normal MPS cutoff
to remove round-off-sized values and expose the lower bond dimension.

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

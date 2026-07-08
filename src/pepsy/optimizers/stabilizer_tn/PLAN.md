# PLAN.md — `pepsy.optimizers.stabilizer_tn` roadmap

Status: living document
Scope: the Stabilizer Tensor Network (STN) simulator subpackage only.

Implements Masot-Llima & Garcia-Saez, *Stabilizer Tensor Networks: universal
quantum simulator on a basis of stabilizer states*, PRL 133, 230601 (2024),
arXiv:2403.08724. See `.github/skills/stabilizer-tensor-networks/` for the
method reference and the verified implementation shortcuts.

State model: `|psi> = C |nu>` — a stim tableau Clifford `C` (basis `B(S, D)`)
times a coefficient MPS `|nu>` (quimb). Clifford gates update the tableau only;
non-Clifford gates and measurements update `|nu>`.

---

## Done (validated against dense/stim; `tests/test_stabilizer_tn.py`)

- **State container** — `STNState`: stim tableau + `|nu>` MPS, `|0...0>` init,
  `to_statevector` (`C|nu>`), `copy`, `nu_frame_pauli`, `pseudo_stabilizer_rank`.
- **Clifford update** — tableau-only; `|nu>` stays chi=1.
- **Non-Clifford rotations** — `exp(-i theta/2 O)` via `M = C^dagger O C`
  (`nu_frame_pauli`); exact bond-dim-2 MPO (`pauli_combo_mpo`) + optional chi
  truncation; single-support M applied as a 2x2 gate.
- **Clifford-angle rotations are free** — angle a multiple of pi/2 routes to the
  tableau (from their reference `n_half_pis`), keeping chi minimal.
- **Explicit gate matrices** — Clifford via `stim.Tableau.from_unitary_matrix` +
  `do_tableau`; 1q non-Clifford via ZYZ -> rotations; 2q non-Clifford rejected.
- **Measurement (Lemma 3)** — `expectation` = `<nu|M|nu>`; `measure` collapses
  with the fixed-basis projector `(I +- M)/2` + renormalize; Born sampling +
  forced outcomes; `("measure", ...)` stream entry.
- **Sub-MPO events** — `("submpo", mpo, where)` applied to `|nu>` (coefficient
  frame), matching the `MpsOptimizer` contract.
- Simulator front end: `MpsStabOptimizer` (gate stream, `chi`, `track_infidelity`,
  `infidelities`, `bond_history`, `set_gates`/`add_gates`/`run`/`apply`).
- **Initial states** — `STNState.zero/from_bits/ghz/from_tableau_and_nu` and the
  matching `MpsStabOptimizer.from_bits/ghz/from_tableau_and_nu` classmethods.
- **Progress bar + diagnostics** — `run(progbar=True)` (tqdm, reports running chi
  and cumulative infidelity); `norm()` returns the `|nu>` norm.
- `StabilizerMps` is kept as a backward-compatible alias for `MpsStabOptimizer`.

## Cross-checked against the reference (bsc-quantic/stabilizer-TN v1.1/v1.2)

Their `gen_clifford` (Qiskit `Clifford` + quimb MPS) confirms our
`gate_decomposition` (= `nu_frame_pauli`), T -> `Rz(pi/4)`, and the
`<nu|M|nu>` + Born measurement. We use stim instead of Qiskit and an exact
bond-dim-2 MPO instead of their CNOT cascade.

---

## Roadmap — improvements from the literature (citation scan of PRL 133, 230601)

Ordered by value/effort. None are started.

### R1. Magic state injection (highest value)
- Nakhl, Harper, West, et al., *Stabilizer Tensor Networks with Magic State
  Injection*, arXiv:2411.12482 (PRL 134, 190602).
- Inject magic states via gate teleportation instead of applying `T`/non-Clifford
  gates directly to `|nu>`. Cost drops from exponential to `O(poly N)` when the
  non-Clifford count `t <~ N` (shown to 200 qubits; 4000 qubits / 320 T-gates on
  hidden bit shift).
- Impact: turns our exact non-Clifford path from chi-growing into poly-scaling for
  T-doped circuits. Needs an ancilla-qubit + measurement-conditioned Clifford
  correction protocol layered on `MpsStabOptimizer`.

### R2. Clifford disentangling sweep (repo-aligned)
- CAMPS: Qian, Huang, Qin, PRL 133, 190402 (arXiv:2405.09217); Clifford-dressed
  TDVP, arXiv:2407.01692 / 2407.03202; authors' own *Limits of Clifford
  disentangling*, arXiv:2602.15942 (the repo v1.2 `disentangling experiments/`).
- Periodically sweep 2-qubit Clifford disentanglers over `|nu>` and absorb them
  into the tableau `C` to reduce `|nu>` bond dimension — the paper's "store
  potential entanglement in the basis" future-work item.
- Impact: directly attacks chi growth; keeps exact semantics. Medium effort.

### R3. Basis-updating (canonical Lemma-3) measurement
- Reference `meas_tableau` + `P_k` projection: absorb the measured observable into
  the stabilizer group and project qubit `k` to `|0>`, keeping `|nu>` support
  compact. Add as `measure(..., absorb_basis=True)` (keep fixed-basis default).
- Impact: smaller `|nu>` after measurement-heavy circuits. Low/medium effort.

### R4. Sampling & observables
- Computational-basis shot sampling via `pepsy.MpsSampler` on `|nu>` mapped
  through `C`; batched expectation values.

### R5. Packaging & examples
- Optionally expose `MpsStabOptimizer` at top-level `pepsy.*` + `docs/api/`
  (already exposed as `pepsy.optimizers.MpsStabOptimizer`; top-level would need
  updating `tests/test_public_api.py`).
- A small deterministic example: `|T>^n` at chi=1, and a magic-vs-chi growth demo
  (paper Fig. 2).

### R7. Long-range gates / layout (subtle — different from `MpsOptimizer`)
`MpsOptimizer.LayoutFinder` reorders MPS sites from the *physical* gate supports to
avoid long-range (SWAP-heavy, chi-growing) gates. That pre-pass does **not** carry
over directly to the STN: a *physical* local non-Clifford gate becomes an operator
`M = C^dagger O C` on `|nu>` whose support is set by the *running tableau* `C` and
can be spread/non-contiguous, and it changes as Clifford gates accumulate. So the
`|nu>`-side "gate stream" is dynamic, not statically known from the input circuit.
Options, in order of value:
- Reduce spread at the source with the **R2 Clifford disentangling sweep** (absorb a
  2-qubit Clifford into `C` to localize/shrink the current `M`) — the principled fix.
- Center the multi-qubit rotation on the innermost affected `|nu>` site (paper Fig. 4)
  and/or adapt the TN geometry to connectivity to hit the `4·chi` (not `16·chi`) bound.
- A dynamic per-step layout: permute `|nu>` sites (= relabel destabilizer generators,
  a basis choice tracked in the tableau) to cluster the current `M` support before a
  spread rotation. This is a real feature but must stay consistent with the tableau.
Do **not** reuse `MpsOptimizer.gate_stream_layout` on the physical stream expecting it
to minimize `|nu>` long-range gates.

### R6. Further leads (not prioritized)
- Hybrid Stabilizer MPO (arXiv:2405.06045) for operator/density-matrix simulation.
- Multiple-basis representation (arXiv:2411.03110) generalizing the ansatz.
- Fux et al. (arXiv:2410.09001): states with #non-Clifford `<~ N` are fully
  disentanglable — informs R2 stopping criteria.

---

## Validation

Run `pytest -q tests/test_stabilizer_tn.py`. Every new update rule must be
checked against a dense statevector (fidelity, up to global phase, tol ~1e-6 —
stim `to_unitary_matrix` is single precision) and, where relevant, against stim.

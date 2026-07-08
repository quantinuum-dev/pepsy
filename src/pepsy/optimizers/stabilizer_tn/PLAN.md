# PLAN.md — `pepsy.optimizers.stabilizer_tn` roadmap

Status: living document
Scope: the Stabilizer Tensor Network (STN) simulator subpackage only.

Implements Masot-Llima & Garcia-Saez, *Stabilizer Tensor Networks: universal
quantum simulator on a basis of stabilizer states*, PRL 133, 230601 (2024),
arXiv:2403.08724. See `.github/skills/stabilizer-tensor-networks/` for the
method reference and the verified implementation shortcuts.

State model: `|psi> = C p` — a stim tableau Clifford `C` (basis `B(S, D)`)
times a coefficient MPS `p` (quimb; the paper's `|nu>`). Clifford gates update the
tableau only; non-Clifford gates and measurements update `p`. The coefficient
MPS is exposed as `.p` (matching `MpsOptimizer.p`); `.nu`, `p_dense`/`nu_dense`,
`frame_pauli`/`nu_frame_pauli`, and `from_tableau_and_state`/`from_tableau_and_nu`
are kept as back-compat aliases.

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
  `do_tableau`; 1q non-Clifford *unitary* via ZYZ -> rotations; **any other
  `k`-qubit matrix (any `k`, unitary or non-unitary)** via Pauli decomposition
  `G = sum_a c_a P_a` -> `M = C^dagger G C = sum_a c_a (C^dagger P_a C)`, applied
  to `p` as a compressed sum of signed Pauli-string branches (`pauli_decomposition`
  + `_apply_dense_gate`/`_apply_operator_sum`). Non-unitary `G` is represented
  without renormalization (coefficient norm tracks `|G|psi>|`).
- **Measurement (Lemma 3)** — `expectation` = `<nu|M|nu>`; `measure` collapses
  with the fixed-basis projector `(I +- M)/2` + renormalize; Born sampling +
  forced outcomes; `("measure", ...)` stream entry.
- **Basis-updating (canonical) measurement** — `measure(..., absorb_basis=True)`:
  a Clifford `V` localises the frame image `M = C^dagger O C` to a single
  coefficient qubit (`V M V^dagger = +-Z_k`, built via `_localizing_clifford`),
  `V` is applied to `|nu>` and `V^dagger` absorbed into the basis
  (`STNState.absorb_basis_clifford`, `|psi>` preserved), then qubit `k` is
  projected to a definite value — so the measured qubit **leaves** `|nu>`
  (disentangled). Validated to match the fixed-basis form (fidelity 1.0) and a
  dense projector across Z/X/Y/ZZ and both outcomes. This is R3 below.
- **Magic-state injection (R1, T-gate)** — `prepare_magic(a)` loads the
  offline-prepared `|A> = T|+>` onto a fresh coefficient site (product, chi=1);
  `inject_t(data, ancilla)` teleports a `T` via `CNOT` (tableau) +
  basis-updating `Z`-measurement of the ancilla + conditional Clifford `S`. The
  ancilla's magic is consumed and absorbed out of `|nu>`. Validated == direct
  `T` (dense) for both outcomes, including on a Clifford-entangled data register
  with `|nu>` bond staying bounded.
- **Sub-MPO events** — `("submpo", mpo, where)` applied to `p` (coefficient
  frame; any MPO, unitary or not), matching the `MpsOptimizer` contract. A
  *physical*-frame few-qubit operator goes through a dense `(matrix, where)`
  entry instead (frame-mapped automatically).
- Simulator front end: `MpsStabOptimizer` (gate stream, `chi`, `track_infidelity`,
  `infidelities`, `bond_history`, `set_gates`/`add_gates`/`run`/`apply`).
- **Initial states** — `STNState.zero/from_bits/ghz/from_tableau_and_state` and the
  matching `MpsStabOptimizer.from_bits/ghz/from_tableau_and_state` classmethods.
- **Progress bar + diagnostics** — `run(progbar=True)` (tqdm, reports running chi
  and cumulative infidelity); `norm()` returns the `|nu>` norm.
- `StabilizerMps` is kept as a backward-compatible alias for `MpsStabOptimizer`.
- **Amplitude / observable API** — `amplitude(bits)`/`probability(bits)`;
  `expectation(pauli, where=None)` (also full-register strings like `"ZIZ"`);
  `expectation_pauli_sum(terms)` for `H = sum c_k P_k`; `sample(...)` (Born
  outcomes, no collapse).
- **Speedups** — non-Clifford rotations/projectors apply a **windowed sub-MPO on
  its true sites** via `gate_with_submpo_` (`pauli_combo_submpo` + `MatrixProduct
  Operator(sites=..., L=...)`), so only the `[lo,hi]` region is compressed — cost is
  O(depth·window), **independent of `n`** for local circuits (n=16/24/32 ~0.28s at
  fixed depth). `STNState` caches `current_inverse_tableau` (invalidated on basis
  changes). All 63 tests pass; correctness vs quimb = 1.0.

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
- **STATUS: first cut DONE** — `prepare_magic` + `inject_t` (built on the R3
  basis-updating measurement). Only the `T` gate is covered so far, because its
  correction `S = Rz(2*pi/4)` is Clifford. **Next:**
  - General diagonal `Rz(phi)` injection: **DONE for `phi` a multiple of `pi/4`**
    via `inject_rz(data, ancilla, phi)` (+ `inject_t`/`inject_tdg` wrappers;
    `prepare_magic(a, angle=phi)`), where the correction `Rz(2*phi)` is Clifford.
    Angles that are *not* a multiple of `pi/4` still need a *recursive* /
    repeat-until-success gadget (raises for now).
  - Preallocate a magic-ancilla register + a `t_via_injection`/circuit-rewrite
    front end that replaces every `("t", q)` stream entry with an injection so a
    T-doped circuit never touches the `|nu>` rotation path.
  - Benchmark: T-doped-Clifford circuit at large `N`, fixed `t` — show `|nu>`
    bond bounded by ~`2^t` independent of `N` (paper Fig. 2).

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
- **STATUS: DONE** — `measure(..., absorb_basis=True)` localises `M` with a
  Clifford `V` (`_localizing_clifford`), applies `V` to `|nu>`, absorbs
  `V^dagger` into the basis (`STNState.absorb_basis_clifford`), and single-site
  projects/disentangles the pivot qubit. Fixed-basis remains the default.
  Follow-ups: choose the pivot / CNOT-ladder to minimise the transient bond
  (currently `k = min(support)`); reuse the R2 disentangler to pre-localise `M`.

### R4. Sampling & observables
- Computational-basis shot sampling via `pepsy.MpsSampler` on `|nu>` mapped
  through `C`; batched expectation values.
- **STATUS: first cut DONE** — `sample_bits(shots, seed)` does chain-rule
  (sequential `Z`-measurement) sampling on a per-shot copy, and
  `probability_bits(bits)` returns `|<bits|psi>|^2` as a product of conditional
  Born probabilities — both `O(n)` MPS measurements instead of an `O(2^n)`
  statevector. **Next:** batched/tree sampling (avoid the per-shot copy) and a
  `MpsSampler`-backed path for many shots.
- **Micro-perf (absorb path): DONE** — the basis-updating measurement's CNOT
  ladder now pivots on the *median* of the support and merges nearest sites
  first, minimising the MPS swap distance (`swap_sites_with_compress` was the
  dominant cost; ~3.6x faster on a spread `n=20` measurement).

### R5. Packaging & examples
- Optionally expose `MpsStabOptimizer` at top-level `pepsy.*` + `docs/api/`
  (already exposed as `pepsy.optimizers.MpsStabOptimizer`; top-level would need
  updating `tests/test_public_api.py`).
- A small deterministic example: `|T>^n` at chi=1, and a magic-vs-chi growth demo
  (paper Fig. 2).
- **STATUS: DONE** — `pepsy.MpsStabOptimizer` and `pepsy.STNState` are exported at
  top level (symbol map + `__all__` + eager import + `tests/test_public_api.py`),
  documented at `docs/api/optimizers/stabilizer_tn.md`, and demonstrated by
  `examples/stabilizer_tn_magic_injection.py` (Clifford-free GHZ, `inject_t` +
  ancilla recycling, scalable sampling, all dense-validated). **Next:** a
  large-`N`/fixed-`t` magic-vs-chi benchmark (paper Fig. 2).

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

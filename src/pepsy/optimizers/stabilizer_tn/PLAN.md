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

## Implementation-quality review (2026-07-13)

Overall assessment: the `C|p>` decomposition, cached frame mapping, localized
measurement, and true-support sub-MPO updates form a strong research-grade
core. Production readiness depends on completing the correctness and
performance hardening below. Correct state semantics take priority over adding
new simulator features.

### Completed correctness hardening

- Quimb's base-10 MPS `exponent` is included in norm and expectation
  calculations and remains correct when a projected state is normalized.
- Forced measurement outcomes require exactly `+1` or `-1`. An impossible
  postselection is detected before tableau/MPS mutation, including the
  basis-updating path.
- `run()` consumes each successfully applied queue entry. If an entry fails,
  that entry and its suffix remain queued without replaying the successful
  prefix on retry.
- Tree-sampled rows receive a final uniform permutation, preserving the shared
  prefix optimization while producing exchangeable i.i.d. row positions.
- Bitstrings, Pauli axes/supports, and tableau/MPS sizes are validated without
  lossy integer coercion or silent support truncation.
- Dense-operator pruning now has an independent, dtype-aware `operator_tol`;
  MPS `cutoff` is used only for SVD compression. Empty Pauli decompositions
  install a valid compact zero MPS, and normalized observable/measurement APIs
  reject zero-norm states before mutation.
- Clifford-angle Pauli rotations are synthesized as linear-size Stim circuits
  using local basis changes and a parity network. No `2**k x 2**k` dense
  matrix is formed, and repeated axis/angle patterns reuse a cached tableau.

### Remaining priorities from the review

1. Bound the arbitrary dense-gate API explicitly. Its Pauli decomposition has
   `4**k` terms; structured or larger operators should use an MPO path and
   branch sums should use balanced reduction.
2. Redesign infidelity diagnostics: use MPS-specialized overlaps, include
   sub-MPO truncation, and distinguish per-step loss from cumulative fidelity.
3. Enforce magic-ancilla pool contracts (unique, in range, initially clean, and
   untouched by ordinary stream entries).
4. Add optional JAX/CuPy coverage before claiming parity with the tested NumPy
   and Torch paths; avoid mutating caller-owned MPOs during backend conversion.
5. Clarify the simulator-facing API with typed measurement/diagnostic records
   and consider `StabilizerMpsSimulator` as the preferred name while preserving
   `MpsStabOptimizer` as a compatibility alias.
6. Reconcile roadmap references to any intentionally removed benchmark/example
   files so documentation and the executable repository remain aligned.

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
  Pauli-coefficient pruning uses `operator_tol` independently of MPS SVD
  `cutoff`; a zero operator is represented by a valid zero-norm MPS.
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
  changes). Correctness is validated by the focused STN and stress suites.

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
    For an *arbitrary* angle injection has **no scaling benefit** (the resource
    `Rz(phi)|+>` is itself prepared with a `|nu>` rotation), so those stay on the
    exact rotation path or should be compiled to Clifford+T (gridsynth) and
    injected — a RUS gadget is intentionally not added.
  - Injection locality: `run_with_injection` picks the **nearest clean ancilla**
    to each data qubit (shorter localizer span); a spread ancilla pool then cuts
    the swap cost the scaling benchmark exposed for a single trailing ancilla.
  - Preallocate a magic-ancilla register + a `t_via_injection`/circuit-rewrite
    front end that replaces every `("t", q)` stream entry with an injection so a
    T-doped circuit never touches the `|nu>` rotation path. **DONE** —
    `run_with_injection(gates, ancillas=...)` and the `with_injection(n_data,
    gates, n_ancilla=...)` classmethod auto-teleport every `t`/`tdg`/`pi/4`-`rz`
    through `inject_rz`, recycling the ancilla pool (size 1 suffices; inject frees
    the ancilla immediately). Validated == direct replay on the data marginal.
  - Benchmark: T-doped-Clifford circuit at large `N`, fixed `t` — show `|nu>`
    bond bounded by ~`2^t` independent of `N` (paper Fig. 2). **DONE** —
    `benchmarks/stabilizer_tn_magic_scaling.py` sweeps `N` at fixed `t` (direct
    and injection modes, any backend) and reports the max `|nu>` bond / rank /
    time; confirms the bond stays `<= 2^t` and flat in `N`. Smoke-tested in
    `tests/test_stabilizer_tn.py`.

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
  (currently the support median with nearest-first merging); reuse the R2
  disentangler to pre-localise `M`.

### R4. Sampling & observables
- Computational-basis shot sampling via `pepsy.MpsSampler` on `|nu>` mapped
  through `C`; batched expectation values.
- **STATUS: first cut DONE** — `sample_bits(shots, seed)` does **perfect (tree)**
  sampling (shots sharing a measured prefix share the collapsed state — one state
  copy per genuine branch, not per shot), and `probability_bits(bits)` returns
  `|<bits|psi>|^2` as a product of conditional Born probabilities — both `O(n)`
  MPS measurements instead of an `O(2^n)` statevector. **Next:** a
  `MpsSampler`-backed path for very large shot counts.
- **Micro-perf (absorb path): DONE** — the basis-updating measurement's CNOT
  ladder now pivots on the *median* of the support and merges nearest sites
  first, minimising the MPS swap distance (`swap_sites_with_compress` was the
  dominant cost; ~3.6x faster on a spread `n=20` measurement).
- **Backend / GPU: DONE** — `MpsStabOptimizer(..., to_backend=...)` (e.g.
  `pepsy.backend_torch` / `backend_cupy` / `backend_jax`) places `|nu>` and every
  gate/MPO on that backend; the stim tableau stays on the CPU. Validated: torch
  `|nu>` matches the NumPy path (fidelity 1.0) for gates, absorb-measurement,
  injection, and sampling.

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

## Big directions (research, not scheduled)

### R8. Stabilizer PEPS (Clifford tableau x 2D coefficient PEPS)
Generalise `|psi> = C |nu>` from an MPS `|nu>` to a **2D PEPS** coefficient state,
keeping the stim tableau `C` for the Clifford/stabilizer part. Motivation: a 2D
system's physical entanglement obeys a 2D area law, so an MPS `|nu>` must snake
through 2D and its bond blows up, while a PEPS `|nu>` respects the lattice; the
tableau absorbs the stabilizer entanglement (free), leaving a compact "magic
PEPS". Natural fit for 2D stabilizer codes (surface/color) with magic/coherent
noise — the natively-2D analog of the decoder idea below.
- Reuse: `STNState` tableau + `frame_pauli`; `pepsy.ps_to_peps`, `PepsOptimizer`,
  `SimpleUpdateGen` for gate application; `build_bra_ket` -> `BdyMPS` ->
  `contract_boundary` + `normalize`/`infidelity` for expectations/normalisation;
  `SymDMRG2` on the **boundary MPS** inside the PEPS contraction.
- Hard parts: (1) frame image `M = C^dag O C` can spread across the lattice and a
  spread operator on a PEPS is expensive -> the **localizer + R2 disentangling
  become load-bearing** (absorb Cliffords so each `M` is geometrically local
  before touching the PEPS); (2) PEPS contraction is approximate (boundary bond
  `chi_b`), so measurement/normalisation lose the 1D STN's exactness; (3) R2 both
  shrinks the PEPS bond and localises the frame operators.
- Tradeoff: 1D STN buys exactness; 2D buys geometry. Wins iff the entanglement is
  mostly Clifford (large for codes, marginal for generic). First target: a
  surface-code patch + a few `T`s.
- Minimal de-risking slice: `PepsStabOptimizer` skeleton (tableau + `ps_to_peps`
  `|nu>`); one **local** non-Clifford rotation via frame-map -> localizer ->
  small gate/PEPO -> truncate; expectation via `contract_boundary`; validate
  `|psi> = C|nu>` against dense / `MpsStabOptimizer` on a 2x2 / 2x3 lattice
  (exact) before any surface-code demo.

### R9. STN-native DEM / maximum-likelihood decoder
Decode a detector error model (DEM) directly on `MpsStabOptimizer`. A DEM gate
stream (Tensy `_dem_gate_stream`) is **all-Clifford XORs (CNOTs) plus one
single-qubit non-unitary branch weight per mechanism**, which is exactly the STN
split: XORs -> tableau (free), branch weights -> `|nu>`.
- **Key structure.** Branch weight `[[w0,0],[w1,0]]` on a fresh `|0>` error site
  `e_i` gives `w0|0> + w1|1>` (a biased-coin *product* magic state), so after all
  mechanisms `|nu>` is a **product state (bond 1)** and the linear XOR parity map
  lives entirely in the tableau `C`. All the decoding cost is in **conditioning
  the detectors on the observed syndrome**: measuring `Z_{d_j}` has frame image
  `M_j = C^dag Z_{d_j} C = Z_{d_j} * prod_{i in j} Z_{e_i}` (the detector's error
  neighborhood — geometrically local), and projecting `|nu>` onto the syndrome
  signs is what grows the coefficient bond.
- **Why small chi should work.** After conditioning, `|nu>`'s entanglement is the
  entanglement of the *posterior* `P(e | syndrome)`. Below threshold the posterior
  has a short correlation length (sparse, local errors) -> area law -> small `chi`;
  `chi` grows with the posterior correlation length as `p` -> threshold. So `chi`
  is a principled knob that **interpolates MWPM (chi=1) <-> exact MLD (chi=inf)**,
  capturing the leading soft-decoding corrections to MWPM.
- **p -> 0 (and beta -> inf) becomes easy.** `w1 = p^beta -> 0`, so each error site
  `-> |0>` (stabilizer) and `|nu> -> |0...0>` (bond 1); the posterior concentrates
  on the minimum-weight error, i.e. decoding reduces to MWPM and is exact at
  `chi=1`. Equivalently `chi` sets the order of the low-`p` series: bond
  `2^k` captures error configurations up to `k` excess faults. (Tempering
  `beta -> inf` coincides with the `p -> 0` concentration.)
- **Parallelism / tricks.**
  - Apply all branch weights first (product, free); the tableau `C` and the frame
    operators `M_j = C^dag Z_{d_j} C`, `M_L = C^dag Z_L C` are computed **once**,
    independent of the syndrome.
  - Across shots the operators are identical and only the projector **signs**
    (the syndrome) differ -> **vmap/batch the signs** (mirrors Tensy
    `decode_tn_signed` with `chunk_size`).
  - Both logical cosets come from one conditioned `|nu>` via
    `<nu| (I +- M_L)/2 |nu>` -> a single margin read.
  - Order the `e_i` sites with the Tensy `LayoutFinder` so each `M_j` is
    contiguous (local projections -> small `chi`); process detectors in a
    sweep/frontier order so `|nu>` only carries a local entangled frontier
    (matches the MPS-frontier layout objective).
  - `absorb_basis=True` conditioning keeps the projected detector qubits out of
    `|nu>`; R2 disentangling localises spread `M_j`.
- **First slice.** Name->index adapter from Tensy `GateStreamModel`
  (`e*`/`k*` string sites, `kind="branch_weight"|"xor"`) to `MpsStabOptimizer`
  int qubits + `(matrix, where)` / Clifford entries; decode by conditioning on a
  Stim detector sample and reading the `M_L` margin; validate against the exact
  DEM-TN contraction on a distance-3 surface code, then sweep `chi` vs `p`.

#### R9a. Capped variant (detectors as sites, errors summed out)
The dual construction that matches the existing Tensy output-MPS decoders
(`DemTn(coalesce_errors=True)`, `to_mps` output-only). Here the MPS **sites are
the detectors (+ logical)** and each error mechanism is **capped/summed** (leg
contracted by `[1,1]`, weights on cap or edges).
- **Capping = a weighted-XOR gate.** Summing error `e_i` collapses mechanism `i`
  to `M_i = (1-p) I + p * X_{S_i}` on its incident detectors+logical `S_i`
  (non-unitary, non-Clifford; only **two** Pauli terms).
- **It is still "CNOTs + non-Clifford".** With pivot `s0 in S_i` and
  `V = prod CNOT(s0 -> s)`, `V X_{s0} V^dag = X_{S_i}`, so
  `M_i = V [ (1-p)I + p X_{s0} ] V^dag` = CNOT fan-out (Clifford) . single-qubit
  non-unitary coin . CNOT fan-out. So the capped stream is CNOTs + 1-qubit
  non-unitary coins, now running **detector<->detector** with coins on detector
  sites (error ancillas folded away).
- **Capping does NOT kill the stabilizer split** *iff* the fan-out CNOTs stay in
  the **tableau** (free) instead of being contracted into an MPS bond. Then
  `|nu>` is a **detector-site** coefficient MPS with the linear/XOR structure
  offloaded to `C` — the synthesis of "detectors as sites" + "linear map free".
  A *plain* capped detector-MPS (direct tensor contraction, no tableau) instead
  bakes that XOR into the bond; that is the representation to beat. Realise `M_i`
  either as `_apply_dense_gate` (2-branch `(1-p)I + p C^dag X_{S_i} C`) or as the
  explicit coin + tableau CNOTs.
- **Where the bond grows.** Un-capped: `|nu>` product until conditioning. Capped:
  coins hit shared, already-entangled detector sites, so `|nu>`'s bond builds up
  **during the sweep** (same total entanglement, paid earlier). Duals across the
  check matrix (error-site MPS <-> detector-site MPS); which has the smaller bond
  depends on the DEM degree structure (errors vs detectors), so pick the side
  with the lower-degree/more-local operators.
- **chi as the accuracy dial (both variants).** Applying a coin **grows** the raw
  bond (its frame image `C^dag X_S C` is generally spread -> ~doubles bond on its
  support); the variational/`chi`-truncation step then **caps** it. So the sweep
  is grow(gate) -> compress(optimize) -> repeat; the optimizer keeps `chi`
  bounded, it does not raise it. `chi = 1` ~ BP/MWPM, `chi -> inf` = exact MLD, so
  you **raise the `chi` cap until the logical margin `<nu|(I +- M_L)/2|nu>`
  plateaus** (the tracked truncation infidelity signals when `chi` is too small).
  Needed `chi` is small below threshold, grows toward threshold, `= 1` exact at
  `p -> 0` / `beta -> inf`.
- **The tableau is always there (correction).** When the capped stream is built
  *on `MpsStabOptimizer`*, every fan-out CNOT goes into the tableau and only the
  single-qubit coin touches `|nu>`, so the Clifford split is preserved by
  construction — capping never "gives it up". (Earlier wording that plain capping
  discards the tableau was wrong: `MpsStabOptimizer` always factors out the
  Clifford; only a *plain* MPS build with no tableau, i.e. `to_mps`, keeps the XOR
  in the bond.) Concretely, one error flipping `d1,d2` gives marginal
  `w0|00> + w1|11>` = **bond 2 as a plain MPS** (`to_mps`), but
  `= CNOT(d1->d2) (w0|0>+w1|1>)_{d1} |0>_{d2}` = **one tableau CNOT x a bond-1
  coin** — so the tableau turns bond 2 into bond 1.
- **What `|nu>`'s bond measures: image vs kernel.** The tableau absorbs the
  *injective* part of the syndrome map (unique error <-> syndrome, tree-like); it
  is the **image** of the check matrix and is free. `|nu>`'s bond grows only from
  **degeneracy** — several error patterns giving the *same* syndrome, i.e. the
  **kernel/cycles** of the check matrix (stabilizer relations + logicals),
  weighted by `p`. So `chi(|nu>) <= chi(to_mps)`, with `|nu>` bond `= 1` for a
  tree-like Tanner graph or `p -> 0`, and `> 1` only from the code's degeneracy —
  exactly the decoding-relevant part. `to_mps` (no tableau) instead carries the
  full image+kernel correlation in its bond.
- **Logical via `[1,-1]` cap (free).** Capping the logical leg with `[1,-1]`
  reads the **margin** `P(L=0) - P(L=1)` directly and folds each logical-flipping
  mechanism into a **sign** on its coin (`M_i = (1-p)I +- p X_{S_i^det}`); the
  logical is not a kept site. Clamping the observed syndrome on the detector legs
  is then the margin.
- **Gate/truncation counts.** Per mechanism: `2(|S_i|-1)` CNOTs (graph-like `~2`,
  boundary `0`) -> tableau (free); exactly **one** coin (non-Clifford) -> `|nu>`.
  So `~2 * #mechanisms` free CNOTs and `#mechanisms` truncation-bearing coins.
- **Cap at the beginning vs end vs don't-cap (same object, different cost).** The
  final `|nu>` bond is identical (it is the kernel/degeneracy structure); only the
  path differs:
  - *Cap at the beginning* (stream weighted-XOR gates onto the **detector-only**
    register): incremental, `chi`-truncation-controlled sweep — best for building
    a reusable open-leg marginal MPS (build once, clamp many syndromes).
  - *Cap at the end* (build the bond-1 un-capped STN on errors+detectors, then sum
    the error legs): bigger register, and all bond growth dumped into one
    poorly-controlled final contraction — worse as a build.
  - *Don't cap — condition*: keep the trivially cheap bond-1 un-capped STN and
    project the detectors onto each observed syndrome; the tableau stays and
    `|nu>` grows only by the conditioned degeneracy — the cheapest path for actual
    decoding (operators are syndrome-independent; batch over shots by flipping
    projector signs).
- **Experiment.** Build the capped weighted-XOR stream (`("cnot", ...)` fan-outs
  in the tableau + `((1-p)I + p X, s0)` coins) on `MpsStabOptimizer`, decode a
  distance-3 patch, and plot logical margin + truncation infidelity vs the `chi`
  cap; diff `|nu>` bond against the existing plain capped detector-MPS to see if
  offloading the fan-out CNOTs to the tableau shrinks it.
- **Empirical (2026-07, `/tmp/stn_dem_bond.py`, exact `chi=None`).** For graph-like
  DEMs the tableau does **not** shrink the marginal bond: repetition-code d=5 gives
  `chi(plain)=chi(STN)=2` (states identical, fid=1.0). Reason: the bond-2->bond-1
  win holds only for an *isolated* mechanism; once mechanisms **overlap** (share a
  detector) a later coin's frame image `C^dag X_{pivot} C` spreads through the
  accumulated tableau, so `|nu>` reacquires the bond. So the image/kernel split is
  defeated by mechanism overlap, and the plain `to_mps` marginal (already bond ~2
  for graph-like codes) is not beaten by the STN. Still worth checking on
  higher-degree / circuit-level (hyperedge) DEMs where the plain bond is larger,
  but the graph-like case is a negative result. **Caught + fixed a real
  `MpsStabOptimizer` bug in the process:** `stim.Tableau.from_unitary_matrix` does
  not verify unitarity, so a near-Clifford *non-unitary* gate (the coin
  `(1-p)I+pX`) was silently accepted as the identity tableau; `_apply_matrix` now
  guards with `_is_unitary(gate)` first (regression tests added).

---

## Validation

Run `pytest -q tests/test_stabilizer_tn.py`. Every new update rule must be
checked against a dense statevector (fidelity, up to global phase, tol ~1e-6 —
stim `to_unitary_matrix` is single precision) and, where relevant, against stim.

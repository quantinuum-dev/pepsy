---
name: stabilizer-tensor-networks
description: 'Learn and implement the Stabilizer Tensor Network (STN) universal quantum simulator from Masot-Llima & Garcia-Saez (PRL 133, 230601, 2024 / arXiv:2403.08724) inside pepsy. Use when the user asks to implement, prototype, port, study, or extend stabilizer tensor networks, the generalized tableau formalism, a stabilizer-basis MPS, |nu>/coefficient-state simulation, Clifford+non-Clifford circuit simulation with an amplitude MPS, or stim-tableau + quimb/pepsy-MPS hybrid simulation. Also use for questions about the STN update rules (Clifford, non-Clifford rotation, measurement), the stabilizer basis B(S,D), pseudo-stabilizer rank, or bond-dimension growth bounds from that paper.'
argument-hint: 'e.g. "implement the Clifford update" or "build the |nu> MPS + stim tableau state"'
---

# Stabilizer Tensor Networks (STN) in pepsy

Implement the hybrid simulator from **"Stabilizer Tensor Networks: universal quantum
simulator on a basis of stabilizer states"** (S. Masot-Llima, A. Garcia-Saez, PRL 133,
230601, 2024; arXiv:2403.08724). The method stores a quantum state as

$$|\psi\rangle = \sum_i \nu_i\, \hat{d}_{\hat i}\,|\psi_{\mathcal S}\rangle$$

i.e. a **stabilizer basis** $\mathcal B(\mathcal S,\mathcal D)$ (a tableau of $n$
stabilizer + $n$ destabilizer generators, tracked with **stim**) plus a **coefficient
state** $|\nu\rangle$ (an $n$-qubit **MPS** in pepsy/quimb). Entanglement lives in the
basis; magic / non-stabilizerness lives in $|\nu\rangle$.

## When to Use
- Implement or extend an STN simulator in this repo (tableau + amplitude MPS).
- Add/verify one of the three update rules: Clifford, non-Clifford rotation, measurement.
- Answer conceptual questions about $\mathcal B(\mathcal S,\mathcal D)$, $|\nu\rangle$,
  free operations, pseudo-stabilizer rank $\tilde\xi$, or $\chi$-growth bounds.

## Do NOT use for
- Plain stabilizer-only (Clifford) simulation → use stim directly.
- Plain MPS/PEPS circuit simulation with no stabilizer basis → use pepsy `gate`/`gate_simple`.

## Substrate (decided for this repo)
- **Tableau / basis $\mathcal B(\mathcal S,\mathcal D)$** → `stim` (`stim.TableauSimulator`,
  `stim.Tableau`, `stim.PauliString`). Never hand-roll the $O(n^2)$ tableau updates
  unless stim genuinely cannot express the step; the paper's own outlook recommends stim.
- **Coefficient MPS $|\nu\rangle$** → pepsy/quimb MPS.
  - Build initial state with `pepsy.ps_to_mps`; norm/fidelity via `pepsy.tn_norm` /
    `pepsy.tn_fidelity`; expectations via `pepsy.expec_mpo` / `pepsy.measure_obs`; sampling
    via `pepsy.MpsSampler`.
  - **Apply the non-Clifford / measurement gate streams (CNOT cascades + central rotation)
    with `pepsy.MpsOptimizer`** — it consumes a canonical bundled stream
    `((gate, where), ...)`, applies it to the MPS, and is the truncation engine (below).
    Use `pepsy.gate` / `pepsy.gate_simple` only for one-off/exact application of a single
    gate when an optimizer is overkill.
- **Exact vs approximate (bounded-$\chi$) mode** — both go through `MpsOptimizer`:
  - *Exact*: `MpsOptimizer(nu, gates, mode="exact")` — no truncation, ground truth for tests.
  - *Approximate*: pick a compressed mode (`"dmrg"`, `"svd"`, `"mpo"`, `"swap"`, `"mix"`) with
    a `chi` cap; monitor accuracy via its `infidelities` / `losses` traces
    (`track_infidelity=True`). This is how the STN caps $\chi$ growth from non-Clifford gates.
- Keep the wrapper thin and Pepsy-idiomatic (see repo `AGENTS.md`): prefer upstream
  quimb/autoray/cotengra behavior over reimplementation.

## Reference implementation (for cross-checking, not vendoring)
`github.com/bsc-quantic/stabilizer-TN` — `v1.1` is a single `stabilizers.py`; `v1.2` (latest)
is packaged under `src/` and adds disentangling experiments. Its main class `gen_clifford`
inherits **Qiskit's** `Clifford` for the tableau and a **quimb** MPS for the complex
coefficient vector; `gen_clifford.compose(...)` accepts non-Clifford unitaries, decomposed by
their own methods. **We deliberately differ:** use **stim** (not Qiskit `Clifford`) for the
tableau and **`MpsOptimizer`** for the MPS side. Consult their `stabilizers_example.ipynb`
and `.compose` logic to validate the decomposition/update math, but do not copy internals
(repo `AGENTS.md`).

## Read the method first
The dense equations, the three update rules, the tableau→basis Pauli decomposition, the
CNOT-cascade rotation, and a worked 5-qubit example live in
[references/method.md](./references/method.md). Read it before writing update-rule code.
The concrete pepsy + stim call surface is in
[references/pepsy_stim_api.md](./references/pepsy_stim_api.md).

## Implementation status (already built in `src/pepsy/optimizers/stabilizer_tn/`)
Phases 1–4 exist and are validated against dense/stim (`tests/test_stabilizer_tn.py`):
`STNState` (tableau + `|nu>`), Clifford update, non-Clifford rotations, explicit gate
matrices, sub-MPO events, and Pauli measurement — all exposed through
`pepsy.optimizers.MpsStabOptimizer`, an `MpsOptimizer`-style gate-stream simulator.

**Key verified shortcut (use this, not the CNOT-cascade masks).** Because
$|\psi\rangle = C|\nu\rangle$ with $C$ the tableau Clifford, a physical Pauli operator
$O$ acts on $|\nu\rangle$ as $M = C^\dagger O C$ — a signed Pauli obtained directly from
stim by conjugating through the tableau:

```python
M = state.nu_frame_pauli(P)   # == sim.current_inverse_tableau()(P); sign is real ±1
```

This collapses all of Lemma 2/3's $I_x,I_y,I_z$ mask algebra into one call:
- **Non-Clifford rotation** $\exp(-i\theta/2\,O) \to \exp(-i\theta/2\,M)$ on $|\nu\rangle$.
- **Measurement** $\langle O\rangle = \langle\nu|M|\nu\rangle$; collapse projector
  $\tfrac{I\pm O}{2}\to\tfrac{I\pm M}{2}$ on $|\nu\rangle$ (basis fixed), then renormalize.
  This fixed-basis collapse is `measure(pauli, where)` (the default), is self-consistent
  (repeated measurement is deterministic), and does **not** absorb $O$ into the stabilizer
  group. The **basis-updating** form is `measure(pauli, where, absorb_basis=True)`: a
  Clifford $V$ localizes $M=C^\dagger O C$ to $\pm Z_k$, is applied to $|\nu\rangle$ with
  $V^\dagger$ absorbed into the basis ($|\psi\rangle$ preserved), and qubit $k$ is projected
  out — so the measured qubit **disentangles from** $|\nu\rangle$.
- **Reset / injection** built on the basis-updating measurement: `reset(where)` returns
  qubit(s) to $|0\rangle$ (disentangled); `prepare_magic(a)` + `inject_t(data, a)` apply a
  `T` by magic-state gate teleportation (Clifford `CNOT` + $Z$-measure + conditional `S`),
  keeping the non-Clifford cost on the pre-loaded ancilla rather than growing $|\nu\rangle$.
  Stream entries: `("measure", pauli, where[, outcome[, absorb_basis]])` and
  `("reset", where)`.

Both $\exp(-i\theta/2\,M)$ and $\tfrac{I\pm M}{2}$ are exact **bond-dim-2 MPOs**
(`c·I + coef·P`) built by `pepsy.optimizers.stabilizer_tn.operators.pauli_combo_mpo`; apply via
`mpo.apply(nu)` then `nu.compress(max_bond=chi)`. Single-support $M$ is applied as a
bond-preserving 2×2 gate. Explicit matrices: `stim.Tableau.from_unitary_matrix` +
`sim.do_tableau` for Clifford; ZYZ→rotations for 1q non-Clifford. Rotations whose angle
is a multiple of $\pi/2$ are Clifford and route to the tableau (free, χ unchanged).

## Roadmap / future improvements
The prioritized roadmap (magic state injection, Clifford disentangling sweep,
basis-updating measurement, sampling, packaging) lives in
`src/pepsy/optimizers/stabilizer_tn/PLAN.md`, with citations from the PRL-133-230601 citation scan.
**R3 (basis-updating measurement) and a first cut of R1 (magic-state injection, `T`-gate
only) are done** (`absorb_basis=True`, `reset`, `prepare_magic`/`inject_t`); the open R1
work is general $R_z(\phi)$ injection (recursive gadget) and a large-$N$/fixed-$t$
poly-scaling benchmark. When extending the simulator, update that PLAN and add a
dense-validated test.

## Implementation Workflow

Implement incrementally, validating each phase against a dense statevector / stim before
moving on. Prefer a small module (e.g. `src/pepsy/optimizers/stabilizer_tn/`) that composes
public APIs; do NOT bolt STN state onto unrelated modules.

### Phase 0 — Internalize
1. Read [references/method.md](./references/method.md) end to end.
2. Confirm the invariant: **Clifford gates change only the basis; non-Clifford gates and
   measurements change only $|\nu\rangle$** (the basis is updated separately for
   measurements). Entanglement in the circuit becomes "potential entanglement" stored in
   the basis, not fictitious $\chi$ in $|\nu\rangle$.

### Phase 1 — State container
1. Define an `STNState` holding: an `n`-qubit stim tableau (basis) and an `n`-qubit
   coefficient MPS `nu`.
2. Initial state $|\psi\rangle=|0\rangle^{\otimes n}$: tableau = identity
   ($s_i=Z_i,\ d_i=X_i$), `nu` = product MPS with $\nu_0=1$ (all qubits `|0>`,
   `pepsy.ps_to_mps`). So $|\nu\rangle=|0\dots0\rangle$, $\chi=1$.
3. Add a `to_statevector()` reconstruction (small $n$ only) for testing: expand
   $\sum_i \nu_i\,\hat d_{\hat i}|\psi_{\mathcal S}\rangle$ from the tableau + dense `nu`.
   **Validate:** random Clifford circuit → STN statevector equals stim's.

### Phase 2 — Clifford update (basis only)
1. For each Clifford gate $G$ (H, S, CNOT, Pauli, …): update ONLY the stim tableau by
   conjugation. Leave `nu` untouched (Eq. 4: $G|\nu\rangle=|\nu\rangle$).
2. Optionally support the alternate policy from the paper's outlook: apply a Clifford
   directly to $|\nu\rangle$ instead of the basis (kept for future $\chi$-allocation
   experiments; default = update the basis).
   **Validate:** any Clifford circuit keeps `nu` a trivial $\chi=1$ MPS and matches stim.

### Phase 3 — Non-Clifford single-qubit rotation (Lemma 2)
Compile the circuit to `{CNOT, RX, RY, RZ}` so every non-Clifford op is a single-qubit
rotation $\mathcal R_P(2\theta)=\cos\theta\,I - i\sin\theta\,P$ with $P\in\{X,Y,Z\}$ on
physical qubit $q$.
1. Decompose the axis Pauli $P_q$ into the current basis:
   $P_q = \alpha\,\delta_{\hat d}\,\sigma_{\hat s}$ (X-part $\hat d$ over destabilizers,
   Z-part $\hat s$ over stabilizers) using the tableau symplectic products — see
   references/method.md §"Pauli → basis decomposition".
2. Apply the Clifford part $\delta_{\hat d}\sigma_{\hat s}$ to `nu` as $X_{\hat d}Z_{\hat s}$
   (Eq. 16), then apply the multi-qubit rotation $\mathcal R_{X^{I_x}Y^{I_y}Z^{I_z}}(2\theta)$
   with masks $I_x,I_y,I_z$ from Eq. 20/33, implemented as a **CNOT cascade + one central
   single-qubit rotation** (references/method.md §"CNOT-cascade rotation"). Emit this as a
   bundled stream `((pepsy.cnot(), where), ..., (pepsy.rx(2θ), q_center), ...)` and apply it
   through `pepsy.MpsOptimizer` (`mode="exact"` or a compressed mode + `chi`).
3. Corollary 2.1 fast path: if the axis is (up to Clifford) a single basis generator, the
   op is a **local** $R$ on one qubit of `nu` — no $\chi$ growth. Detect and take it.
   **Validate:** single T-gate / RZ on a random Clifford tableau reproduces the expected
   statevector; the $|T\rangle^{\otimes n}$ state stays $\chi=1$ (paper Eq. 11–12).

### Phase 4 — Measurement / observable (Lemma 3)
1. Decompose observable $\mathcal O=\alpha\,\delta_{\hat n}\sigma_{\hat m}$ in the basis.
2. Expectation: $\langle\mathcal O\rangle=\alpha\,\langle\nu|X_{\hat n}Z_{\hat m}|\nu\rangle$
   (Eq. 29) via `pepsy.expec_mpo` / `pepsy.measure_obs`.
3. Sample outcome $m\in\{+,-\}$ with $p_+=(1+\langle\mathcal O\rangle)/2$.
4. Update the stim tableau for the measurement, then apply the **non-unitary**
   $\tilde{\mathcal R}=\tfrac12 I \pm \alpha(-i)^{|I_y|}X^{I_x}Y^{I_y}Z^{I_z}$ (CNOT cascade
   with the central 1-qubit op of Eq. 44 as a custom gate tensor in the `MpsOptimizer`
   stream), a projector $P_k=|0\rangle\langle0|_k$ on qubit $k$ (first 1 of $\hat n$), and
   renormalize by $\sqrt{(1\pm\langle\mathcal O\rangle)/2}$ (Eq. 43). Masks from Eq. 33.
   **Validate:** measurement statistics + post-measurement state match a dense simulator
   over many shots.

### Phase 5 — End-to-end validation, truncation & resources
1. Compare full `{CNOT,RX,RY,RZ}` circuits against dense statevector for small $n$ using
   `MpsOptimizer(mode="exact")`.
2. Track `nu.max_bond()` per gate; confirm empirical $\chi$ growth stays within the paper's
   bounds ($\chi' \le 4\chi$ MPS-local, $\le 16\chi$ with SWAPs; avg $\sim 2^{2.46}$ per
   T-gate, Fig. 2).
3. **Approximate mode:** run the same streams through `MpsOptimizer` with a compressed mode
   + `chi` cap and `track_infidelity=True`; expose the accumulated `infidelities`/`losses`
   as the accuracy budget so users trade $\chi$ for precision explicitly.
4. Expose a pseudo-stabilizer rank $\tilde\xi$ = number of non-zero `nu` coefficients.
5. Add focused tests (`tests/`), keep optional deps optional
   (`pytest.importorskip("stim")`), update `docs/`/`__all__` if you add public symbols
   (repo Public API Rules).

## Validation Checklist
- [ ] Clifford-only circuits: `nu` stays $\chi=1$; STN statevector == stim.
- [ ] $|T\rangle^{\otimes n}$ builds with $\chi=1$ (Clifford H layer + free T rotations).
- [ ] Single non-Clifford rotation matches dense statevector (all three axes).
- [ ] Corollary-2.1 local rotations do not grow $\chi$.
- [ ] Measurement expectation, sampling probabilities, and collapsed state match dense.
- [ ] Norm stays 1 after every step (`pepsy.tn_norm(nu) ≈ 1`).
- [ ] Optional-dependency guards + focused tests pass (`pytest -q tests/test_...`).

## Common Pitfalls
- **Sign/phase bugs** dominate. Track the $\alpha$ phase from the basis decomposition and
  the $\delta_1\cdot\sigma_2$ sign in Lemma 2; validate every phase against dense output.
- **Basis vs $|\nu\rangle$ confusion.** Clifford → basis only. Non-Clifford → $|\nu\rangle$
  only. Measurement → both. Never apply a non-Clifford gate to the tableau.
- **Qubit ordering / MPS site order** must be identical between the stim tableau and the
  pepsy MPS. Fix one convention (`OneDMap`) and assert it.
- **Long-range rotations are the worst case** ($\le 16\chi$ via SWAPs). Keep the CNOT
  cascade centered on the innermost affected qubit (paper Fig. 4) to limit $\chi$.
- Do not vendor stim/quimb internals; isolate any workaround behind a small adapter and
  test it against the closest public API (repo `AGENTS.md`).

## References
- Paper: arXiv:2403.08724 (PRL 133, 230601). Reference implementation:
  `github.com/bsc-quantic/stabilizer-TN`.
- [references/method.md](./references/method.md) — equations, update rules, tableau rules,
  worked example.
- [references/pepsy_stim_api.md](./references/pepsy_stim_api.md) — pepsy + stim call surface.

---
name: stabilizer-tensor-networks
description: 'Learn and implement the Stabilizer Tensor Network (STN) universal quantum simulator from Masot-Llima & Garcia-Saez (PRL 133, 230601, 2024 / arXiv:2403.08724) inside pepsy. Use when the user asks to implement, prototype, port, study, or extend stabilizer tensor networks, the generalized tableau formalism, a stabilizer-basis MPS, nu/coefficient-state simulation, Clifford plus non-Clifford circuit simulation with an amplitude MPS, or stim-tableau plus quimb/pepsy-MPS hybrid simulation. Also use for questions about STN cooling, magic-state injection including deferred MAST projection, the STN update rules (Clifford, non-Clifford rotation, measurement), the stabilizer basis B(S,D), pseudo-stabilizer rank, or bond-dimension growth bounds from that paper.'
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

## Scope boundaries
- Use stim directly for plain stabilizer-only (Clifford) simulation.
- Use Pepsy's ordinary gate/optimizer paths for MPS/PEPS simulation with no stabilizer basis.

## Substrate (decided for this repo)
- **Tableau / basis $\mathcal B(\mathcal S,\mathcal D)$** → `stim` (`stim.TableauSimulator`,
  `stim.Tableau`, `stim.PauliString`). Never hand-roll the $O(n^2)$ tableau updates
  unless stim genuinely cannot express the step; the paper's own outlook recommends stim.
- **Coefficient MPS $|p\rangle$ (the paper's $|\nu\rangle$)** → quimb MPS owned by
  `STNState.p`; `.nu` is a compatibility alias. Construct and evolve it through
  `MpsStabOptimizer`, not through a nested `MpsOptimizer`.
  - Single-support frame Paulis use `p.gate_(..., contract=True)` and do not grow a bond.
  - Multi-support rotations/projectors use `pauli_combo_submpo` on the true contiguous
    support window followed by `p.gate_with_submpo_`. This is the live replacement for the
    paper/reference implementation's CNOT-cascade execution.
  - General few-qubit matrices are Pauli-decomposed, frame-mapped branch by branch, summed,
    and compressed with a balanced streaming reduction. The fallback defaults to at most
    two qubits via `max_pauli_decomposition_qubits=2`; larger values explicitly opt into
    the `4**k` cost. A `("submpo", mpo, where)` event instead acts directly in the
    coefficient frame.
- **Exact vs approximate (bounded-$\chi$) mode** is selected on `MpsStabOptimizer`:
  - *Exact*: `MpsStabOptimizer(..., chi=None)`. The SVD `cutoff` still removes exact
    numerical redundancy so repeated bond-dimension-2 MPOs do not double bonds forever.
  - *Approximate*: `MpsStabOptimizer(..., chi=cap, track_infidelity=True)`. Each compressed
  unitary update can record cumulative norm loss `1 - ||p||^2`; bounded evolution does
  not renormalize `p`, so compression loss remains visible.
- **Coefficient-MPS compression methods** use the canonical `quimb-<method>`
  mode family, with `quimb` as the direct alias. The historical `mpo-<method>`
  and `mpo` spellings remain supported. Every Quimb direct, density-matrix,
  zip-up, SRC, SRCMPS, and `fit-*` method is forwarded to
  `gate_with_submpo_`; randomized methods accept the separate
  `compression_seed` control. DMRG modes use a disposable
  `fit_init_strategy="guess-<method>"` (default `guess-zipup`) before active
  bonds reach their cap, while the exact target remains built from the live
  unmodified coefficient MPS. The underscore `guess_<method>` spelling remains
  a compatibility alias. Native Symmray and fermionic paths retain their
  sector-aware direct FIT initialization.
- **Canonical-centre discipline** → preserve the simulator's `cur_orthog` info through
  quimb operations. Canonicalize explicitly before local projection, evaluate local
  expectations and unitary norm loss at the tracked centre, and renormalize the centre
  tensor only after projection. Never renormalize bounded unitary evolution: its lost norm
  is the diagnostic. Full arbitrary non-unitary matrix entries are also intentionally not
  renormalized, but invalidate the norm-loss proxy because their norm change is physical.
- Keep the wrapper thin and Pepsy-idiomatic (see repo `AGENTS.md`): prefer upstream
  quimb/autoray/cotengra behavior over reimplementation.

## Diagnostic contract

- Treat `track_infidelity` as the canonical ``infidelity`` API. For unitary
  updates it is the normalized norm-loss proxy; for compressed dense
  non-unitary matrices the retained norm ratio is measured against the local
  physical ``G^dagger G`` target. It is not a general ideal-circuit overlap
  calculation.
- Read `1 - ||p||^2` from the tracked one-site canonical centre, including Quimb's MPS
  `exponent`. Do not build an uncapped target, contract `<target|p>`, or renormalize after
  a unitary update.
- `.infidelities` is a sparse historical trace. Clifford gates, bond-preserving one-site
  unitaries, measurements/projectors, arbitrary non-unitary matrices, and `submpo` events
  emit no sample. Never sum this list; each emitted value is already cumulative within its
  normalized unitary segment.
- An unnormalized non-unitary matrix invalidates the current unitary norm
  segment after recording its ``G^dagger G``-normalized compression loss.
  A coefficient-frame `submpo` has no certified physical target norm and
  invalidates the unitary proxy without adding a sample. A normalized
  projective collapse starts a fresh segment at zero but does not itself append
  a unitary sample.
- Projective measurement/reset boundaries are recorded separately in `.norm_events`.
  Each event snapshots the pre-collapse unitary segment norm, the physical Born branch
  probability, the actual projected norm before renormalization, and the post-normalized
  norm. Compare `projected_norm_sq` to `pre_norm_sq * branch_probability` to get the
  projector-compression proxy `projector_infidelity`. Use `norm_diagnostics()` for
  product/geometric summaries that multiply unitary and projector compression-survival
  factors, but never measurement probabilities. Prefer `local_fidelity`,
  `local_infidelity`, `cumulative_fidelity`, `cumulative_infidelity`,
  `norm_survival`, and `norm`; the older `total_*_proxy` keys are compatibility
  aliases. `geometric_mean_norm`
  is only the per-segment geometric mean, not the
  total norm summary.
- A selected `TrajectoryEvent` Kraus outcome is a normalized trajectory
  boundary. Before applying its non-unitary matrix, snapshot the current
  segment with its branch probability; normalize the selected branch at the
  canonical centre, reset the proxy, and commit a `"trajectory_kraus"` norm
  event. This lets later unitary steps track a fresh segment without treating
  the Born probability as compression loss. STN progress reports the shared
  `infidelity` field plus a compact stream `part` label. Fidelity and
  infidelity names describe compression measured from norms; live `norm`
  remains separate.
- Treat stream-local stochastic entries as the primary Pepsy noise design.
  `("x_error", p, q)`, `("depolarize1", p, q)`,
  `("depolarize2", p, q0, q1)`, `("pauli_channel1", probs, q)`,
  `("pauli_channel2", probs, q0, q1)`, and
  `("amplitude_damping", gamma, q)` lower to trajectory events. Use
  `PauliErrorModel` only as a macro for clean deterministic streams.
- Keep `.bond_history` independent from `.infidelities`; their indices are not aligned.
- Remember the limitation: norm loss detects compression that removes norm, but it is not
  a proof of state fidelity and cannot detect every norm-preserving directional error.

## Reference implementation (for cross-checking, not vendoring)
`github.com/bsc-quantic/stabilizer-TN` — `v1.1` is a single `stabilizers.py`; `v1.2` (latest)
is packaged under `src/` and adds disentangling experiments. Its main class `gen_clifford`
inherits **Qiskit's** `Clifford` for the tableau and a **quimb** MPS for the complex
coefficient vector; `gen_clifford.compose(...)` accepts non-Clifford unitaries, decomposed by
their own methods. **We deliberately differ:** use **stim** (not Qiskit `Clifford`) for the
tableau and direct quimb MPS operations orchestrated by **`MpsStabOptimizer`** for the
coefficient side. Consult their `stabilizers_example.ipynb` and `.compose` logic to validate
the decomposition/update math, but do not copy internals (repo `AGENTS.md`).

## Read the method first
The dense equations, the three update rules, the tableau→basis Pauli decomposition, the
paper's CNOT-cascade construction, and a worked 5-qubit example live in
[references/method.md](./references/method.md). Read it before writing update-rule code.
The concrete pepsy + stim call surface is in
[references/pepsy_stim_api.md](./references/pepsy_stim_api.md).

## Implementation status (already built in `src/pepsy/optimizers/stabilizer_tn/`)
The simulator is mature and validated against dense/stim (`tests/test_stabilizer_tn.py`):
`STNState` (tableau + `|p>`), Clifford and
non-Clifford evolution, constructive exact cooling, explicit greedy Clifford cooling,
immediate and deferred-MAST magic-state injection, fixed/basis-updating measurement, reset,
perfect computational-basis sampling, and optional Torch/JAX/CuPy coefficient backends
covered by focused tests when the dependencies/runtimes are available.
It is exposed as `pepsy.StabilizerMpsSimulator` / `pepsy.MpsStabOptimizer` and
`pepsy.STNState`, with typed measurement/projection/norm/injection diagnostic records.

**Key verified shortcut (use this, not the CNOT-cascade masks).** Because
$|\psi\rangle = C|\nu\rangle$ with $C$ the tableau Clifford, a physical Pauli operator
$O$ acts on $|\nu\rangle$ as $M = C^\dagger O C$ — a signed Pauli obtained directly from
stim by conjugating through the tableau:

```python
M = state.frame_pauli(P)   # == sim.current_inverse_tableau()(P); sign is real ±1
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
  `with_injection` measures and recycles an ancilla immediately. The separate
  `with_deferred_injection` MAST path assigns one fresh ancilla to each injectable gate,
  performs the Clifford gadget and branch correction during replay, then defers the physical
  basis-updating ancilla projections to a chosen final order. Stream entries:
  `("measure", pauli, where[, outcome[, absorb_basis]])` and `("reset", where)`.

Both $\exp(-i\theta/2\,M)$ and $\tfrac{I\pm M}{2}$ are exact **bond-dim-2 MPOs**
(`c·I + coef·P`) built on the real support window by
`pepsy.optimizers.stabilizer_tn.operators.pauli_combo_submpo`; apply with
`gate_with_submpo_` and the configured `chi`/`cutoff`. Single-support $M$ is applied as a
bond-preserving 2×2 gate. Explicit matrices: first verify unitarity, then try
`stim.Tableau.from_unitary_matrix` for Clifford; use ZYZ rotations for a 1q non-Clifford
unitary; Pauli-decompose every other matrix. Rotations whose angle is a multiple of
$\pi/2$ are Clifford and route to the tableau (free, $\chi$ unchanged).

## Live execution paths
- **Named Clifford** → `STNState.apply_clifford`; invalidate the cached inverse tableau;
  leave `p` and its canonical centre unchanged.
- **Named Pauli rotation** → build physical `P`, compute `M=C^dagger P C` with
  `frame_pauli`, extract its real sign/support, then use a local gate or windowed sub-MPO.
  For Clifford angles, synthesize the tableau directly with local basis changes,
  a CNOT parity network, and `S`/`Z`/`S_DAG`; never form the exponentially large
  Pauli rotation matrix.
- **Fixed-basis measurement** → compute the Born probability only for sampled outcomes;
  apply `(I + mM)/2`, reject zero-probability forced outcomes, and normalize at the tracked
  centre. The tableau is unchanged.
- **Basis-updating measurement** → localize `M` to signed `Z_k` with a median-pivot
  Clifford `V`, apply `V` to `p`, absorb `V^dagger` into `C`, project site `k`, and normalize
  there. This is the reset/injection primitive.
- **Explicit matrix** → check true unitarity *before* asking stim whether it is Clifford;
  stim does not reject all non-unitary near-Clifford matrices. Non-Clifford 1q unitaries use
  ZYZ; remaining matrices within `max_pauli_decomposition_qubits` use a balanced
  Pauli-branch operator sum without normalization. Only a matrix verified unitary can emit
  a norm-loss sample. For larger physical-frame operators, prefer supported gate/rotation
  decompositions; `submpo` is coefficient-frame only and carries no unitarity declaration.
- **Backend conversion** → stim/tableau classification remains NumPy/CPU; coefficient MPS,
  local gates, and MPO arrays use `to_backend`. Convert user matrix entries to NumPy for
  classification before converting coefficient-side operations back to the chosen backend.

## Cooling and magic-injection policy

There are two deliberately different ways to control coefficient-MPS bond dimension:

- **Constructive exact cooling (`exact_cooling=True`, default):** before a multi-site
  non-Clifford Pauli rotation, inspect its frame image. If an isolated product coefficient
  site has a Pauli stabilizer that anticommutes with the local rotation axis, replace the
  multi-site MPO by one local rotation and absorb the controlled-Pauli remainder into the
  tableau. The represented state is exact and that update does not increase `|nu>` bond.
  This is a deterministic, cheap applicability check, not a global optimization. Leave it
  enabled in ordinary simulation; use `exact_cooling=False` only to test or benchmark the
  normal MPO fallback.
- **Greedy Clifford cooling (`disentangle_cliffords`):** score local two-qubit Clifford
  candidates with Schmidt/SVD data, apply an improving candidate, and absorb its inverse
  into the tableau. It can reduce an already-grown bond, but costs local SVD work over each
  selected bond. Invoke it explicitly at a few meaningful circuit checkpoints; never hide it
  in every T gate or every non-Clifford rotation.

For magic injection, choose the schedule rather than treating deferred MAST as an automatic
replacement:

- `with_injection(n_data, gates, n_ancilla=1, ...)` is the immediate path. It measures each
  gadget immediately and reuses a clean ancilla. Prefer it when ancilla count and steady
  throughput matter most.
- `with_deferred_injection(n_data, gates, n_ancilla=None, projection_order="middle_out", ...)`
  is the MAST path. It reserves one clean ancilla per injectable `t`, `tdg`, or non-Clifford
  `rz(k*pi/4)` gate, keeps those projections until after circuit replay, and reports replay
  and projection costs separately. Prefer it when a low replay-phase peak bond is valuable
  and a final projection phase plus `t` extra ancillas are acceptable. The circuit itself
  must not touch the reserved ancillas. `middle_out` is the default static order; `input`
  preserves injection order; `min_span` greedily selects the current shortest frame span.

Use `MpsStabOptimizer.analyze_stream(gates, ...)` and
`MpsStabOptimizer.recommend_settings(gates, ...)` before selecting settings. These are
Pepsy-stream-first APIs: they inspect Clifford, T-family, other non-Clifford, dense matrix,
sub-MPO, measurement/reset/cap, and qubit-use features, then return typed,
mapping-compatible advice records. `recommend_settings` delegates only the
direct/immediate/deferred part to `MpsStabOptimizer.recommend_magic_strategy(gates, ...)`,
which remains available when the caller wants just the injection schedule. On an unrun
`from_stim` simulator, `sim.queued_stream_analysis()` and
`sim.queued_recommend_settings()` analyze the sampled and optionally transformed Pepsy
stream. This is deliberately advice, not an `auto` executor: `apply()` must retain direct
semantics unless the caller explicitly selects `with_injection` or
`with_deferred_injection`.
For correctness smoke runs before benchmarking, use
`run_stabilizer_mps_stream(gates, mode=...)`, which returns a typed
`StabilizerMpsRunResult` recording the actual mode/settings, replay/projection timings,
bond data, norm diagnostics, measurements, and injection reports. Its default mode is
direct; `mode="recommended"` is an explicit opt-in to execute the advisor's mode.
`sim.run_queued_stream(...)` applies that runner to an unrun converted queue without
mutating the source simulator.

Compare `direct`, `immediate`, and `deferred` execution through the public
runner APIs. Read peak `|nu>` bond together with projection bond and projection
cost; final bond alone does not show the deferred projection cost. Keep timing
harnesses outside the Pepsy package.

## Roadmap / future improvements
The prioritized roadmap (Clifford disentangling, dynamic layout, future PEPS/decoder work,
and completed injection/measurement/sampling milestones) lives in
`docs/development/plans/stabilizer_tn.md`, with citations from the PRL-133-230601 citation scan.
R1 supports `inject_rz` for $\phi$ a multiple of $\pi/4$, `inject_t`/`inject_tdg`, immediate
nearest-clean-ancilla recycling through `run_with_injection` / `with_injection`, and deferred
MAST projection through `run_with_deferred_injection` / `with_deferred_injection`.
Arbitrary-angle injection is intentionally excluded because preparing its resource state has
the same non-Clifford cost; use direct rotation or compile to Clifford+T. R2 also includes the
default constructive exact-cooling pre-check and the explicit greedy disentangling sweep.
When extending the simulator, update that PLAN and add a dense-validated test.

## Extension workflow

The simulator already exists. Extend it incrementally and preserve the representation
invariant at every intermediate step.

1. **Internalize the invariant.** Read [references/method.md](./references/method.md), then
   reason from $|\psi\rangle=C|p\rangle$. Clifford gates change `C`; physical
   non-Clifford operators act on `p` through $C^\dagger O C$; basis-updating measurement
   changes both in a coordinated way that preserves the physical state before projection.
2. **Choose the correct frame.** Named physical gates and dense `(matrix, where)` entries
   must be frame-mapped. A `submpo` event is deliberately coefficient-frame and must not be
   conjugated through `C` a second time.
3. **Choose the narrowest coefficient update.** Let constructive exact cooling attempt its
   deterministic local-pivot identity before a multi-site rotation. If it does not apply,
   use a local 2x2 gate for one support site, `pauli_combo_submpo` for a two-branch Pauli
   operator, and the branch-sum path only for a genuine multi-term matrix. Label every
   sub-MPO with its real MPS sites.
4. **Maintain numerical state.** Thread the `info` orthogonality-centre tracker through
   quimb splitting/canonicalization calls. Preserve the configured backend for MPS-side
   arrays. Normalize projective collapse, but do not normalize unitary evolution or
   arbitrary non-unitary gates. Report norm loss only across normalized unitary segments;
   if a projector is compressed before normalization, record that projector-survival
   factor separately from the Born probability.
5. **Validate the physical state.** For small `n`, compare `C @ p_dense` with an independent
   dense circuit up to global phase. Compare Clifford-only behavior with stim. Test exact
   `chi=None` first, then a bounded-`chi` path and backend variants when touched.
6. **Schedule expensive optimizations deliberately.** Keep exact cooling on in ordinary
   replay. Use the greedy disentangler only at explicit checkpoints. For magic circuits,
   choose immediate recycled injection or deferred MAST based on ancilla budget and whether
   final projection cost is acceptable; do not mix their ancilla-lifetime assumptions.
7. **Keep APIs synchronized.** If a public symbol or stream form changes, update the owning
   `__all__`, top-level lazy exports, `docs/api/`, `tests/test_public_api.py`, and the PLAN.

## Validation checklist
- [ ] Clifford-only circuits leave `p` at $\chi=1$ and match stim.
- [ ] X/Y/Z and long-range Pauli rotations match dense evolution with `chi=None`.
- [ ] Local frame images use the bond-preserving fast path; multi-site images stay on the
  true support window.
- [ ] Fixed-basis and basis-updating measurements match dense probabilities and collapsed
  states; repeated measurements are deterministic; impossible forced outcomes raise.
- [ ] Reset disentangles its targets; direct and injected T/T-dagger/pi/4-multiple Rz paths
  agree, including ancilla recycling.
- [ ] Exact cooling agrees with the ordinary MPO path when a stabilizer pivot exists and
  correctly falls back when it does not. Greedy cooling is tested only when explicitly called.
- [ ] Deferred MAST matches direct replay for forced outcomes and every supported projection
  order; an odd-sized `middle_out` register projects every fresh ancilla exactly once.
- [ ] `sample_bits`/`probability_bits` match dense probabilities without mutating the source.
- [ ] Clifford matrices, 1q non-Clifford unitaries, general unitary matrices, and
  non-unitary matrices all follow their intended dispatch paths.
- [ ] Normalized unitary/measurement paths keep norm one; general non-unitary matrices keep
  the dense reference norm instead of being normalized.
- [ ] Bounded `chi` is respected and the unitary norm-loss proxy decreases toward zero as
  `chi` increases; its latest emitted value equals `1 - norm()**2`; non-unitary and
  coefficient-frame sub-MPO paths emit no sample.
- [ ] Optional backends keep `p` and applied MPS-side arrays on that backend.

## Common pitfalls
- **Sign/phase bugs dominate.** Preserve the real sign returned by
  `hermitian_pauli_terms`; validate physical states up to global phase because stim's dense
  tableau unitary is single precision and has no global phase.
- **Physical frame vs coefficient frame.** Clifford → tableau only; physical non-Clifford
  → frame-map then `p`; `submpo` → `p` directly; basis-updating measurement → coordinated
  `p -> Vp`, `C -> CV^dagger`, then projection.
- **Stim's unitary parser is not a unitary validator.** Call the explicit unitarity check
  before `stim.Tableau.from_unitary_matrix`, or weighted non-unitary coins can be silently
  misrouted as Cliffords.
- **Sub-MPO labels control where the operator acts.** `where` controls the compression
  region, but quimb aligns the MPO from its own site labels. Building a local MPO on sites
  `0..w-1` and passing a different `where` applies it to the wrong qubits.
- **Canonical-centre metadata is part of the algorithm.** Blind rescans/full-network
  normalization erase the measurement performance gain; stale metadata produces incorrect
  local norms. Update or deliberately invalidate `state.info["cur_orthog"]` after rebuilds.
- **Do not reinterpret diagnostics.** Quimb does not expose one uniform discarded-SVD
  record across sub-MPO, branch-sum, and other compression paths. Do not synthesize one or
  sum `.infidelities`; preserve the unitary norm-loss contract above.
- **Long-range cost is dynamic.** A local physical gate can have a spread frame image after
  Clifford evolution. Static `MpsOptimizer` layout analysis of the physical stream does not
  optimize these evolving coefficient supports.
- **Do not conflate cooling schedules.** Exact cooling prevents a specific next update from
  growing; the greedy sweep tries to reduce a bond that already exists. The latter is useful
  but should not run after every T gate.
- **Deferred MAST does not recycle.** Its ancillas remain reserved until final projection;
  ordinary circuit entries must not act on them. `min_span` is a greedy *projection-order*
  planner, not the greedy Clifford cooler.
- **Qubit ordering must agree.** Stim, dense reconstruction, and MPS sites use the same
  big-endian convention with qubit 0 first.
- Do not vendor stim/quimb internals; isolate any workaround behind a small adapter and test
  it against the closest public API (repo `AGENTS.md`).

## Follow-up ideas (related work)
- **arXiv:2607.08396** (Deger, Koutsioumpas, Webster, Sayginel, Roffe, Browne, 2026),
  *code-compiled tensor networks* — a **different, complementary** route (plain physical
  MPS + CSS-code-compiled circuits), **not** a tableau + $|\nu\rangle$ STN. Two portable
  techniques worth trying here:
  - **Classical permutation tracking** (a qubit-relabel register) instead of SWAP
    networks — directly attacks the "long-range rotations / non-local Clifford
    localizers are the worst case ($\le 16\chi$ via SWAPs)" pitfall above and the slow
    `cycle=2` STN measurements: track permutations classically and relabel MPS sites
    lazily rather than applying SWAP/localizer gates that span the chain.
  - **Phase-polynomial backend** for diagonal (phase) magic — an exact backend whose
    cost is set by phase-polynomial degree, not entanglement; a companion to the
    coefficient MPS $|\nu\rangle$ when the non-Clifford part is diagonal-heavy.
  - Framing takeaway: entanglement, magic, and non-Gaussianity are each individually
    insufficient to indicate simulation hardness — which supports the STN premise.

## References
- Paper: arXiv:2403.08724 (PRL 133, 230601). Reference implementation:
  `github.com/bsc-quantic/stabilizer-TN`.
- [references/method.md](./references/method.md) — equations, update rules, tableau rules,
  worked example.
- [references/pepsy_stim_api.md](./references/pepsy_stim_api.md) — pepsy + stim call surface.

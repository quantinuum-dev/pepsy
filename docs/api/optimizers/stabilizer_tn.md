# `pepsy.optimizers.stabilizer_tn`

The Stabilizer Tensor Network (STN) simulator (Masot-Llima & Garcia-Saez, PRL
133, 230601, 2024; arXiv:2403.08724). A state is stored as `|psi> = C |nu>`: a
stim tableau Clifford `C` (the stabilizer basis `B(S, D)`) times a coefficient
MPS `|nu>` (the paper's coefficient state, exposed as `.p`). Clifford gates
update only the tableau (free, `|nu>` unchanged); non-Clifford gates and
measurements update `|nu>`.

General physical matrices use the exact coefficient-frame Pauli mapping when
their explicit budget is within the default three qubits. Set
`max_pauli_decomposition_qubits=4` only when accepting the 256-term cost of a
four-qubit dense gate; such gates emit a warning and are rejected by default.
Set `max_pauli_terms` separately to bound the retained decomposition. `chi=None`
remains exact up to the configured cutoff, while a finite `chi` compresses the
mapped operator with the normal diagnostics. The coefficient-MPS compression
backend uses the same bare method names as `MpsOptimizer`: `mode="direct"`
(the default), `"dm"`, `"zipup"`, `"src"`, their `*-first` and
`*-oversample` variants, and the `fit-*` variants. `"dmrg"`/`"dmrg1"`/
`"dmrg2"`/`"dmrg3"`, `"svd"`, `"swap"`, `"perm"`, and `"exact"` remain
available. Historical `"quimb-*"` and `"mpo-*"` spellings are accepted only
as deprecated aliases. `mode="exact"` forces `chi=None`; Clifford tableau
updates remain free in every mode.

For DMRG modes, `fit_init_strategy="guess-<method>"` selects an isolated
native-compressed FIT guess before active bonds reach their `chi` ceilings;
`"guess-src"` is the default SRC warm-up and continues into the fixed-rank
one-site phase after expansion. `"auto"` resolves to `"guess-src"`.
`"direct"`, `"random"`,
`"random_expand"`, and `"svd_guess"` are also supported. The underscore
spelling, for example `"guess_zipup"`, remains a compatibility alias. Set
`compression_seed` separately from the STN measurement `seed` for randomized
native methods. The STN DMRG wrapper grows with a two-site FIT target for
`dmrg`, `dmrg1`, and `dmrg2`, and a three-site target for `dmrg3` when the
active frame window spans at least three sites; `dmrg3` falls back to two-site
FIT for the common two-qubit window. Each update finishes with a one-site
refinement on multi-site windows, while the SRC guess is kept isolated from
the exact target construction. On dense backends, the exact coefficient
sub-MPO is retained as a tagged lazy FIT target layer: the active MPS window
is canonicalized first, then FIT contracts the MPS and sub-MPO tensors without
absorbing the operator into an intermediate target MPS. Symmray and fermionic
routes retain the materialized backend-safe target fallback.
`get_fit_diagnostics()` reports the selected block size, growth/refinement
sweeps, SRC guess method, target representation, and the DMRG1 one-site latch.

`StabilizerMpsSimulator` is the descriptive public name for the simulator, with
`MpsStabOptimizer` kept as the long-standing compatibility alias. Both are
available at top level as `pepsy.StabilizerMpsSimulator` and
`pepsy.MpsStabOptimizer` (state container: `pepsy.STNState`). For new code,
prefer the descriptive class name and the explicit bitstring helpers
(`sample_bitstrings`, `bitstring_probability`, and
`bitstring_probabilities`); the shorter historical names remain supported.
Internal measurement and projection paths keep the coefficient-MPS
`state.info["cur_orthog"]` metadata synchronized. Ordinary diagnostic
expectations and samples use non-collapsing paths; if lower-level code directly
canonicalizes `sim.state.p`, call `sim.sync_canonicalization()` before
continuing the simulation.
Supported
gate-stream entries include Clifford gates,
non-Clifford Pauli rotations, `("t", q)` / `("tdg", q)`, explicit `(matrix,
where)` gates, `("submpo", mpo, where)` events, `("measure", pauli, where[,
outcome[, absorb_basis]])`, `("reset", where[, basis])`,
`("measure_reset", basis, where[, outcome[, absorb_basis]])`, and guarded
physical cap events `("cap", where, vec[, absorb])`. Stochastic/error entries
such as `("depolarize1", p, q)`, `("pauli_channel1", probs, q)`, and
PECOS-style stateful leakage entries such as `("leakage", p, q)` and
`("measure_leaked", q)` are also Pepsy stream entries, but they are sampled by
the trajectory runners rather than plain `sim.run()`.

For an end-to-end choice between direct simulation, immediate injection,
deferred MAST, and the two cooling mechanisms, see the
[STN magic and cooling how-to](../../howto/stabilizer_tn_magic.md).

Before choosing settings, start with the Pepsy-native stream advisor:
`MpsStabOptimizer.analyze_stream(gates, n_qubits=...)` returns a typed
`StreamAnalysisRecord` with counts for Clifford entries, injectable T-family
rotations, other non-Clifford rotations, dense matrices, coefficient-frame
sub-MPOs, measurements, resets, caps, touched qubits, and warnings. Then
`MpsStabOptimizer.recommend_settings(gates, goal="run" | "validate" |
"benchmark", ...)` returns a typed `StabilizerMpsSettingsAdvice` containing
constructor settings (`chi`, `cutoff`, `exact_cooling`, `stabilize_unitary`),
an explicit execution method (`apply`, `with_injection`, or
`with_deferred_injection`), ancilla requirements, warnings, and a
human-readable `message`.

`recommend_settings` calls the narrower
`MpsStabOptimizer.recommend_magic_strategy(gates, ...)` internally for the
`direct` / `immediate` / `deferred` decision. On an unrun simulator,
`queued_stream_analysis()` and `queued_recommend_settings()` read its queue,
including a `from_stim(..., stream_transform=...)` result. All of these APIs are
advisory only: `apply()` continues to use direct STN execution unless the caller
explicitly selects an injection constructor.

For a small correctness or smoke run, use the explicit validation runner:
`StabilizerMpsSimulator.run_stream(gates, n_qubits=..., mode="direct" |
"immediate" | "deferred" | "recommended", ...)` returns a typed
`StabilizerMpsRunResult`. `StabilizerMpsSimulator.simulate(...)` is an alias,
and the historical module-level `run_stabilizer_mps_stream(...)` remains
available. The default `mode` is `direct`; `mode="recommended"` is an explicit
opt-in to the mode selected by `recommend_settings`. The result records the simulator,
actual mode, settings used, replay/projection timing, final and peak
coefficient-MPS bond, norm diagnostics, measurements, projection events,
injection report, and remaining queue length. On a `from_stim` simulator,
`sim.run_queued_stream(...)` replays the already sampled Pepsy queue through the
same runner without mutating the original queued simulator.

You can initialize from an ordinary computational-basis qubit MPS directly:
`MpsStabOptimizer(p)` or `MpsStabOptimizer.from_mps(p)` wraps `p` with the
identity tableau, so initially `C = I` and `|psi> = |p>`. Use this for an
already-prepared MPS ground state. Pass `inplace=False` to copy the supplied
MPS before evolution.

For one sampled Stim trajectory, use
`MpsStabOptimizer.from_stim(circuit, seed=...)`. It compiles the Stim circuit,
infers its qubit count, samples native Pauli noise once, and queues the resulting
stream. The seed also deterministically initializes its later measurement sampling.
The returned simulator retains `.stim_plan` and `.stim_sample`, including
the selected faults and herald bits. `stream_transform=` receives the immutable
sampled stream and can insert ordinary physical Pepsy gates or remove a terminal
readout before replay; this keeps the Stim-to-Pepsy parsing in one public path.
Stim parsing is a convenience adapter, not a requirement for the advisor: the
same `analyze_stream` and `recommend_settings` APIs operate directly on any
Pepsy stream.

```python
sim = pepsy.MpsStabOptimizer.from_stim(circuit, chi=32, seed=7)
sim.run(progbar=True)
print(sim.stim_sample.faults)
```

With `progbar=True`, the STN progress bar reports the current stream
`part` (`clifford`, `T`, `measurement`, `reset`, `nonclifford`, ...) and the
MPS-compatible `infidelity` field. It denotes the retained-norm compression
proxy; it is not a target-state overlap.

The STN norm diagnostics use the same naming contract as ordinary MPS
compression: `current_fidelity` / `current_infidelity` describe the active
normalized coefficient segment, while
`cumulative_fidelity` / `cumulative_infidelity` are accumulated in log space.
These are compression fidelities measured from retained norms for `|nu>`, not direct
overlaps with the physical target state `C|nu>`. A direct target overlap is a
separate diagnostic and is only available when an explicit reference state is
contracted, such as the final FIT-target check in ordinary MPS DMRG.
In `norm_diagnostics()`, `norm`/`state_norm` are the live coefficient-state
norm, while `cumulative_norm` is the square-root retained-compression proxy.

For physical readout, keep the two representations explicit:
`sim.to_basis_statevector()` returns the dense coefficient vector `|nu>` in
tableau order, while `sim.to_statevector()` returns the
computational-basis vector `|psi> = C|nu>`. Both materialize only a length
`2**n` vector; physical readout applies `stim.Tableau.to_circuit()` and never
constructs a dense `2**n`-by-`2**n` Clifford matrix. `to_physical_statevector()`
remains a compatibility alias for `to_statevector()`. For a non-dense physical
representation, use `sim.to_mps(mode="exact")`. This replays the tableau gate
stream into a new ordinary MPS with unlimited bond and zero cutoff. For
controlled approximation, use a bare native mode such as `mode="src"` or
`mode="dmrg"` with `chi=...`
and `cutoff=...`; `logical_order=True` (the default) returns sites in logical
qubit order, even when a static STN coefficient layout is installed.
`to_physical_mps()` remains a compatibility alias for `to_mps()`.

## Tableau inspection

The STN basis Clifford is available as a read-only Stim tableau.  It is the
`C` in `|psi> = C|nu>`; `x_output(i)` gives destabilizer `d_i` and
`z_output(i)` gives stabilizer `s_i`:

```python
tableau = sim.tableau()
print(tableau)
```

For a compact Pepsy-style summary, use `ascii_tableau()` to obtain text or
`show()` to print it.  The default sparse generator format reports only the
non-identity support, while `compact=False` uses Stim's full-width Pauli
strings.  `max_generators=` is useful for large registers:

```python
sim.show(max_generators=12, color=True)
text = sim.ascii_tableau(compact=False, color=False)
```

`draw()` returns a Stim circuit diagram of the current Clifford frame without
materializing the dense Clifford matrix.  The default is a text timeline;
the default text form is returned as a string.  Stim formats such as
`timeline-svg` are forwarded as Stim diagram helper objects, and
`format="circuit"` returns the underlying `stim.Circuit`:

```python
print(sim.draw())
svg = str(sim.draw(format="timeline-svg"))
circuit = sim.draw(format="circuit")
```

These views show the tableau/frame `C` only.  Non-Clifford evolution remains
in the coefficient MPS `|nu>` and is summarized by the bond and norm fields in
`show()`.

Shot ensembles can use the same optimizer-level MPI API as ordinary MPS
optimization:

```python
result = sim.run(
    shots=1_000_000,
    mpi=True,
    workers="auto",
    progress="auto",
    seed=7,
    retain="none",
)
```

`mpi=True` uses the communicator created by `mpiexec`; it does not launch MPI
processes. Each shot is rebuilt from the simulator's initial snapshot, and
`workers="auto"` divides the available CPU allowance among ranks on the same
host. `progress="auto"` emits one rank-zero aggregate bar only in interactive
terminals.

`exact_cooling=True` is the default constructive pre-check for multi-site
non-Clifford Pauli rotations. If the frame image has an isolated product
stabilizer pivot, the optimizer performs one local coefficient rotation and
absorbs the controlled-Pauli remainder into the tableau. The update is exact,
deterministic, and does not grow the coefficient-MPS bond. Successful uses are
recorded in `exact_cooling_events`. Set `exact_cooling=False` only when testing
or benchmarking the ordinary MPO fallback.

`disentangle_cliffords(sweeps=1, *, bonds=None, tol=None)` is the separate,
optional greedy Clifford-gauge optimization: it scores local two-qubit Clifford
candidates from Schmidt/SVD data, applies an improving coefficient-frame
Clifford `D`, then absorbs `D^dagger` into the tableau, so `(C D^dagger)(D
|nu>) = C |nu>`. It can also be placed at an exact point in a stream as
`("disentangle",)` or `("disentangle", {"sweeps": 1, "bonds": ..., "tol": ...})`.
It can lower a bond that already exists, but it is not free. Schedule it at a
few explicit checkpoints instead of after every T gate or rotation.

`apply_layout("auto")` installs a **static STN frame layout** before replay:
it dry-runs the queued Clifford/basis-update skeleton, collects the expensive
coefficient-frame supports `C^dagger O C`, and chooses an MPS site order that
keeps those supports short. This is intentionally different from
`MpsOptimizer`'s physical gate-stream layout. The operation is exact and cheap
only while the coefficient MPS is product (`state.max_bond() == 1`); apply it
before non-Clifford evolution entangles `|nu>`. The constructor accepts
`layout="auto"` / `layout_kwargs={...}`, and `current_frame_layout(...)`
returns the plan without mutating the simulator. Immediate and deferred
injection runners also accept `layout="auto"`; they build a synthetic layout
stream from the magic-ancilla gadgets and final projections, rather than using
the original data-only stream.

For measurement/feed-forward circuits, use
`("if", record, bit, action)`. `record=-1` means the latest measurement,
negative records are Stim-style offsets, and `bit` is the computational
measurement bit (`+1 -> 0`, `-1 -> 1`). The action is exactly one gate entry,
for example `("if", -1, 1, ("x", q))`. This form is supported by both STN
frontends and by the trajectory runners. `compile_stim_circuit` lowers
`CX/CY/CZ rec[k] q` to the same event; general classical arithmetic remains
outside the quantum replay contract.

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
  them has no scaling benefit; compile to Clifford+T to inject). The reserved
  ancilla pool is checked before replay: indices must be unique, in range, and
  initially clean physical `|0>` qubits, and ordinary stream entries must not
  touch the pool. Pass `layout="auto"` to choose a coefficient-MPS order from
  the rewritten injection-gadget stream before replay.
- `run_with_deferred_injection(gates, *, ancillas, projection_order=...)` and
  `with_deferred_injection(n_data, gates, *, n_ancilla=None, ...)` implement
  deferred MAST injection. Each injectable gate gets one distinct fresh ancilla;
  the magic gadgets replay first and their basis-updating `Z` projections occur
  at the end. `middle_out` is the default projection order; `input` retains
  injection order and `min_span` chooses the current shortest frame span.
  Deferred mode cannot recycle ancillas. The same unique/in-range/clean and
  ordinary-entry isolation contracts are enforced before replay. Inspect
  `deferred_projection_events` and `last_deferred_injection_report` for the
  per-projection support/bond data and replay versus projection timing. Pass
  `layout="auto"` to include the magic preparation, branch corrections, and
  final projection supports in the static layout prepass.
  Immediate injection analogously exposes `immediate_projection_events` and
  `last_immediate_injection_report`.

Measurement and diagnostic logs use typed records while preserving older access
patterns: `measurements` contains tuple-compatible `MeasurementRecord` objects;
`norm_events`, projection events, and injection reports contain mapping-like
records such as `NormEventRecord`, `ImmediateProjectionRecord`,
`DeferredProjectionRecord`, `ImmediateInjectionReport`, and
`DeferredInjectionReport`. Both `event.field` and `event["field"]` work.

## Scalable sampling

- `sample_bitstrings(shots, *, seed=None, order=None, shuffle=True,
  packed=False)` (`sample_bits` compatibility alias) —
  computational-basis bitstrings by **perfect (tree) sampling**: shots sharing a
  measured prefix share the collapsed state, so the collapse work is done once
  per distinct prefix, not per shot (a large saving for structured/low-rank
  `|nu>`). No `O(2**n)` statevector is formed. `order="physical"` preserves the
  historical `Z_0...Z_{n-1}` readout, `order="mps"` follows the current static
  coefficient-MPS layout, `order="auto"` uses the layout order only after a
  nontrivial layout has been installed, and an explicit qubit permutation is
  also accepted. Set `shuffle=False` to keep prefix-grouped rows and skip the
  final row permutation. Set `packed=True` to return `np.packbits` output with
  `ceil(n / 8)` byte columns.
- `iter_sample_bitstrings(shots, *, chunk_size, seed=None, order=None,
  shuffle=True, packed=False)` (`iter_sample_bits` compatibility alias) —
  chunked sample generation with a shared RNG across chunks, for large shot
  counts where materializing one full `(shots, n)` array is inconvenient.
- `bitstring_probability(bits, *, order=None)` (`probability_bits`
  compatibility alias) — `|<bits|psi>|**2` as a product of conditional Born
  probabilities.
- `bitstring_probabilities(bitstrings, *, order=None)` (`probability_bits_many`
  compatibility alias) — batched computational bitstring probabilities using
  the same prefix-sharing readout tree as bitstring sampling, so duplicate or
  prefix-clustered bitstrings avoid independent full readout passes.

## Correctness and failure semantics

- Bitstrings contain exactly binary integer values, Pauli supports use distinct
  in-range sites, and forced measurement outcomes are exactly `+1` or `-1`.
- Impossible postselection raises `ValueError` before changing the tableau,
  coefficient MPS, measurement log, or diagnostics.
- Immediate and deferred magic-injection schedules validate their reserved
  ancillas before replay, so duplicate/out-of-range/dirty pools or ordinary
  stream entries touching the pool fail without partially applying the stream.
- `run()` removes successfully applied entries from its queue. If a later entry
  fails, that entry and the remaining suffix stay queued, so retrying does not
  replay the successful prefix.
- `norm()` and normalized measurements account for Quimb's separate MPS
  `exponent` scale as well as the tensor data.
- Dense-operator coefficient pruning is controlled by `operator_tol`, never by
  the MPS SVD `cutoff`. With `operator_tol=None`, the threshold is relative to
  the matrix scale and input dtype; an explicit value is an absolute tolerance.
- Fallback Pauli decomposition is limited by
  `max_pauli_decomposition_qubits=3` before its `4**k` enumeration begins.
  Four-qubit dense attempts warn and require an explicit limit of at least `4`.
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
Fallback dense matrices use a `4**k` Pauli decomposition and are limited to
three qubits by default. Four-qubit attempts warn and are rejected unless the
limit is raised explicitly. Sparse results such as `I + XX`, `I + YY`, `I + ZZ`, or
small mixtures of those terms are applied as a single exact sub-MPO; dense
results are combined with a balanced, streaming MPS sum. This improves
reduction depth but does not remove the exponential number of candidate Paulis.
For larger physical-frame operators, decompose into named gates or Pauli
rotations. A `submpo` event is appropriate only when the MPO is already
expressed in the coefficient frame. Clifford-angle Pauli rotations are
synthesized directly as linear-size Stim basis-change and parity circuits, then
cached; they do not form a `2**k x 2**k` dense matrix. Prefer structured
coefficient-frame sub-MPOs where applicable. Fidelity tracking is automatic and
performs no reference-state copy or overlap contraction. For normalized unitary
evolution it records the cumulative proxy `1 - ||nu||**2` after compressed
coefficient-MPS updates, reading the norm from the tracked one-site canonical
centre. `get_compression_norm_events()` exposes each update's local retained
norm ratio, while `norm_diagnostics()["local_fidelity"]` is the latest such
ratio and `cumulative_fidelity` is the stable cumulative proxy. By default
unitary updates are not renormalized, so lost norm remains visible;
`stabilize_unitary=True` restores the pre-compression working norm after recording
the same local and cumulative ledger.
For dense multi-qubit non-unitary matrices, the target norm is measured from
the local physical `G†G` expectation and the retained norm ratio is reported as
`infidelity`. Coefficient-frame sub-MPOs and arbitrary physical maps without a
certified target norm do not emit such samples; an unnormalized non-unitary
map also suspends later unitary samples until projection restores a normalized
baseline. The public `infidelities` trace is therefore sparse and
historical, is not index-aligned with `bond_history`, and must not be summed.
Projective measurement/reset boundaries do preserve the current segment before
normalization in `norm_events`: the event records the pre-collapse norm,
`1 - ||nu||**2` for that segment, the Born branch probability separately, and
the actual post-projector norm before renormalization. For fixed-basis projection,
comparing the actual post-projector norm with
`pre_norm**2 * branch_probability` gives a separate `projector_infidelity` proxy
for compression done while applying the projector. Basis-updating projection
computes the physical `branch_probability` before its localizer; if the localizer
is compressed at finite `chi`, `projector_branch_probability` records the
post-localizer conditional probability used to isolate the final one-site
projector loss. The post-collapse state is then normalized. Use
`norm_diagnostics()` to form
product/geometric-mean survival summaries across completed segments plus the
current open segment; these summaries multiply unitary- and projector-
compression survival factors, but not measurement probabilities. The preferred
summary keys are `infidelity`, `fidelity`, `norm_survival`, and `norm`; the
older `total_*_proxy` keys remain compatibility aliases.
The proxy is not exact overlap fidelity or a discarded-SVD-weight report;
validate physical accuracy independently when that distinction matters.

The disentangling sweep scores the local Schmidt spectrum for the 20 two-qubit
Clifford classes modulo output-local Cliffords, rather than brute-forcing all
11,520 two-qubit Cliffords or copying the full MPS per candidate. Its `tol`
controls both the relative numerical-rank decision and the local SVD split:
use `tol=0` to retain every numerical singular value, or the normal MPS cutoff
to remove round-off-sized values and expose the lower bond dimension.

## Backends (Torch / JAX / CuPy)

Pass `to_backend=` (e.g. `pepsy.backend_torch(dtype=torch.complex128,
device="cuda")`, `pepsy.backend_cupy(...)`, `pepsy.backend_jax(...)`) to the
constructor or `with_injection`. The coefficient MPS `|nu>` is placed on that
backend, and user gates/MPOs must be prepared with the same converter before
they are queued, so the heavy MPS contractions
(SVD, `swap+split`, sub-MPO application) run on that array backend.  The stim tableau
(classical Clifford tracking) stays on the CPU.  Constant gate matrices are
cached per backend; expectation/fidelity scalars are converted back to Python
floats.  `to_basis_statevector()` returns the coefficient vector `|nu>` in
tableau-basis order without applying the tableau. `to_statevector()` returns
the physical computational-basis vector `C|nu>` and applies a tableau circuit
without constructing a dense `2**n x 2**n` Clifford matrix.
`to_physical_statevector()` is its compatibility alias. Both methods still materialize
a length-`2**n` vector and are for small-`n` validation only. The focused STN
tests exercise NumPy, Torch, JAX, and CuPy paths; optional JAX/CuPy tests skip
only when the dependency or CUDA runtime is unavailable.

When an existing coefficient MPS is supplied, the stabilizer optimizer infers
its common `backend`, `dtype`, and `device` automatically, even when
`to_backend` is omitted. `backend_info()` returns the live mapping and refreshes
the public `backend`, `backend_dtype`, and `backend_device` attributes. Explicit
matrix gates and every tensor in coefficient-frame sub-MPOs are checked against
that backend and device at the stream boundary; non-NumPy payloads must also
match dtype, while NumPy-to-NumPy dtype promotion is compatible. A foreign payload raises
`TypeError`; prepare it explicitly with the same converter used for the
coefficient state. Stim gate classification still uses a temporary NumPy view,
while coefficient contractions remain on the inferred backend. Stim and
trajectory-generated matrices are converted by the library before they enter
this user-stream boundary.


> API details are maintained as handwritten Markdown in this page.

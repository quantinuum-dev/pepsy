# `pepsy.optimizers.tree_stabilizer`

`StabilizerTreeSimulator` is the first tree-backed Stabilizer Tensor Network
milestone. `TreeStabOptimizer` remains a deprecated compatibility alias. It
represents the state as

```text
|psi> = C |p>
```

where `C` is a Stim tableau Clifford and `|p>` is a dense two-level
`TreeTensorNetwork` evolved by `TreeOptimizer`.

TreeStab forwards `cutoff` and `cutoff_mode` to that same coefficient
optimizer. The defaults are `cutoff=1e-10` and `cutoff_mode="rsum2"`, matching
Quimb's open-boundary `gate_with_submpo` compression convention. TreeStab's
canonical coefficient-update modes are `mode="tree_mpo_direct"` and
`mode="tree_mpo_dm"` (the default is the former). Both build a true
`TreeMPO` over the active TreePlan span and apply it with
`TreeOptimizer.apply_subtreempo`; the suffix selects direct SVD or
density-matrix compression. Hyphenated spellings such as
`mode="tree-mpo-dm"`, and the compatibility alias `mode="tree_mpo_dem"`, are
accepted.

TreeStab also forwards `compression_mode` to its coefficient
`TreeOptimizer`. DM mode selects the density-matrix-equivalent local
`svd:eig` decomposition after the complete coefficient operator and tree
state have been fused. It does not canonicalize or compress the operator
separately, and it does not build a global dense state. `mode="dm"` remains a
compatibility shorthand for TreeMPO-DM routing.

Other TreeOptimizer algorithms are available through the same coefficient
engine: `mode="dmrg"` (alias `"fit"`), `"dmrg2"`, `"dmrg3"`, `"mix"`,
`"src"`, `"sdc"`, `"sdcr"`, `"zipup"`, and their supported oversampling
variants. `dmrg1` remains the ordinary tree's deprecated alias of `dmrg`.
The default remains `tree_mpo_direct`. TreeStab maps physical operators into
the coefficient frame before applying the selected tree algorithm; Clifford
frame updates still leave the coefficient state untouched.

DMRG uses TreeOptimizer's exact layered target and separate disposable guess,
not a precompressed target or a direct-compression substitute. All `fit_*`
constructor controls have the same defaults and validation as TreeOptimizer.
Generic `dmrg`/`fit` requires `fit_block_size=1`; choose `dmrg2` or `dmrg3`
for block growth. With the default four iterations, their block schedules
are `(2, 2, 1, 1)` and `(3, 3, 2, 1)`. Each iteration has an inward and an
outward pass. The default `fit_init_strategy="auto"` selects `guess-src`
for these dense coefficient trees. The guess method and local
`compression_mode` are independent. `get_fit_diagnostics()` delegates to
the engine's latest FIT record and returns `None` outside FIT.

TreeStab exposes `compression_seed`, `max_bond_oversample`,
`cutoff_oversample`, `cutoff_mode_oversample`, and `stabilize_unitary` with
the ordinary tree semantics. Copies and shot templates preserve the live
settings; frame-layout replacement and dense physical cap reconstruction
also retain the FIT schedule, compression controls and diagnostic policies.
Coefficient localizers use `TreeOptimizer.apply_gate`, including its compact
SubTreeMPO construction, validation and factorization cache. Explicit full
TreeMPO and compact SubTreeMPO events use `apply_sub_mpotree` directly.
Physical caps first construct the reduced tree losslessly, then compress it
through the same dispatcher using a bond-one identity over the complete tree.
This honors the selected algorithm and cutoff convention without constructing
a dense identity matrix. Dense physical-state reconstruction remains guarded.
TreeStab also forwards `max_intermediate_bond`, `subtree_workers`,
`record_history`, `profile`, `profile_sync` and `track_bond_diagnostics`.
FIT settings are constructor controls, not `run()` overrides.
Public threaded shot replay initializes Stim's small NumPy matrix classifier
before dispatching workers, avoiding a first-use import deadlock reproduced
with the installed Stim build. This does not change the simulated state.

```python
from pepsy.optimizers import StabilizerTreeSimulator

sim = StabilizerTreeSimulator(
    6, mode="dmrg3", chi=32, fit_n_iter=4,
    fit_init_strategy="guess-src", fit_init_seed=7,
    fit_rtol="auto", exact_cooling=False,
)
sim.apply([("rot", 0.37, "XYZ", (0, 2, 5))])
print(sim.get_fit_diagnostics()["block_size_trace"])
```

Dense non-Clifford gates are Pauli-decomposed through ``C† G C`` in the
coefficient frame, then compiled into compact Tree-native ``TreeMPO``/TTNO
operators. The same applies to one-site, two-site, and wider coefficient-frame
Pauli sums: branches use Tree virtual channels only on the union of their
minimal Steiner subtrees, and TreeOptimizer performs the final canonical
compression. Separated supports therefore never become a fictitious
contiguous MPS window. Explicit ``mode="submpo"`` and caller-supplied
MPS-style ``submpo`` events remain compatibility-only interfaces; new TreeStab
code should use `tree_mpo_direct` or `tree_mpo_dm` and
`subtreempo_event(...)`.

Canonical and compression state has the same single owner as ordinary tree
simulation: local isometry proofs live on the coefficient tensors'
``left_inds`` and are interpreted by ``TreeTensorNetwork``. TreeStab delegates
``isometry_direction()``, ``isometry_map()``, ``can_skip_canonize()``, and
``validate_isometry_metadata()`` to its coefficient ``TreeOptimizer``; it does
not keep another map. TreeMPO coefficient updates reuse proven path/subtree Q
tensors and select one-sided SVD compression only when the live proof is
valid. Backend conversion and dense cap reconstruction preserve or install
those proofs rather than forcing a second canonicalization sweep.
State-evolution measurements use the coefficient tree's state-owned canonical
centre. After lower-level direct access to the coefficient TTN,
``sync_canonicalization()`` explicitly rebuilds that centre before replay
continues; post-run diagnostics should normally use ``copy()``.

The first milestone supports:

- named and matrix-valued Clifford gates, which update only the tableau;
- bounded dense one- and few-qubit matrices, including non-Clifford and
  non-unitary matrices, through coefficient-frame Pauli decomposition;
- physical Pauli rotations, frame-mapped with `C† P C` and applied through the
  tree-native Pauli rotation/sub-MPO path;
- fixed-basis Pauli expectation, measurement, and projection;
- basis-updating Pauli measurement, reset, and measure-reset;
- immediate/recycled magic-state injection for `T`, `T†`, and non-Clifford
  `Rz(k*pi/4)` entries;
- deferred/MAST injection with one fresh ancilla per injectable gate and
  configurable final projection order;
- constructive exact cooling for multi-site Pauli rotations when a
  tree-isolated product-stabilizer pivot exists;
- explicit, caller-scheduled greedy two-qubit Clifford disentangling over
  selected logical qubit pairs;
- metadata-only tree-span scheduling for independent measurement/reset batches;
- dynamic frame-layout planning from the queued ``C† P C`` supports;
- conditional/batched product-Pauli sampling without a ``2**n`` statevector;
- ``chi=None`` for exact evolution up to the requested singular-value cutoff;
- ``from_stim`` and non-consuming ``analyze_stream`` compatibility with the MPS
  STN frontend.
- coefficient-frame ``submpo`` stream events for arbitrary unitary or
  non-unitary MPO operators.
- complete Tree-native ``TreeMPO``/TTNO stream events via
  ``subtreempo_event`` (with ``subttno_event`` as an alias); these contract
  internal TreeMPO bonds on the coefficient TreePlan rather than lowering to
  a chain MPO.
- bounded physical ``cap`` events, with dense state reconstruction guarded by
  ``max_dense_cap_qubits`` and an identity-frame rebuild on ``n - 1`` qubits;
 ``amplitude`` and ``probability`` provide matching small-state readouts.
- optional Torch/JAX/CuPy-compatible coefficient backends through
  ``to_backend`` and ``backend_info``;
- ``ghz``, ``pseudo_stabilizer_rank``, ``norm_diagnostics``, and the shared
  stream-advice/runner surface: ``recommend_magic_strategy``,
  ``recommend_settings``, ``run_stream``, and ``simulate``.

```python
import pepsy

sim = pepsy.StabilizerTreeSimulator(4, gates=[
    ("h", 0),
    ("cnot", 0, 1),
    ("rz", 0.2, 0),
])
sim.run()
print(sim.to_statevector())
outcome, probability = sim.measure_pauli("Z", 0)
```

The tree plan is fixed before replay. Pass `tree=` or `layout=` when a
specific geometry is required; otherwise the initial stream supports are sent
to `TreeLayoutFinder`. For STN streams whose physical supports spread under
Cliffords, pass `frame_layout="auto"` (or call `current_frame_layout`) to
plan from the pre-replay coefficient-frame supports ``C† P C``. An entangled
coefficient TTN must retain its existing plan, following the same state/layout
contract as `TreeOptimizer`.

Automatic tree construction accepts `layout_kwargs`, including the hybrid
path/congestion weights and bounded greedy/nevergrad refinement options from
`TreeLayoutFinder`. Static `frame_layout="auto"` is intentionally rejected for
branch-dependent feed-forward streams unless an explicit plan is supplied.
When `with_injection` or `with_deferred_injection` builds an automatic plan, the
synthetic magic preparation, data--ancilla CNOT, and projection supports are
included in a magic-aware hybrid/auto layout by default; pass `layout_kwargs` to
increase the search or refinement budget.

Greedy Clifford disentangling is separate from ordinary replay and exact
cooling. Call `sim.disentangle_cliffords(...)` at sparse checkpoints, or place
`("disentangle", {"sweeps": 1, "bonds": ((q0, q1),)})` in a stream. `bonds`
contains logical qubit pairs; `None` visits all unordered pairs in increasing
tree-distance order. The 20 two-qubit Clifford classes are scored by the
aggregate numerical rank and entropy of the affected tree-geodesic edges.
Accepted moves apply `D` to `|p>` and absorb `D†` into the tableau, preserving
the represented physical state while reducing coefficient-tree entanglement.

`measure_many([("Z", 1), ("X", 2)])` and `reset_many((1, 2))` use
`order="min_span"` by default. The scheduler reads each observable's current
`C† O C` support and scores its minimal TreePlan Steiner subtree, with the
tree-geodesic localizer cost as a tie-breaker. It executes the selected
operation once, then recomputes the remaining metadata costs; it never runs
trial TreeMPO contractions or truncations merely to choose an order. Results
from `measure_many` and `measure_reset` remain aligned with the supplied input
order, while `last_measurement_schedule` exposes the chronological order and
costs. Use `order="input"` or an explicit permutation to override it. A joint
multi-qubit `measure(...)` remains one indivisible operation.

Constructive exact cooling is enabled by default for multi-site Pauli rotations: it chooses a
tree-distance-aware product-stabilizer pivot, applies one local coefficient
rotation, and absorbs the controlled-Pauli remainder into the tableau. Set
`exact_cooling=False` to force the ordinary tree Pauli-rotation path. The
shared trajectory runner also accepts
TreeStab factories for random-unitary mixtures, depolarizing channels, and
state-dependent Kraus channels; selected branches are normalized at their
trajectory boundary. Dense matrix
  decomposition is bounded by ``max_operator_qubits=2`` by default (the MPS-compatible alias
`max_pauli_decomposition_qubits` is also accepted) and uses the compact
tree-native Pauli-sum TreeMPO path. `max_pauli_terms` (default 256) is a second guard for
explicit larger-matrix opt-ins; it fails before constructing an oversized
coefficient-frame MPO. Immediate injection is explicit: use
`prepare_magic`/`inject_rz` for one gadget or `run_with_injection` for a stream
and reserved ancilla pool. The runner chooses a nearest clean ancilla in the
fixed tree geometry, recycles it through `reset`, and reports projection cost
plus peak coefficient-tree bond. Deferred injection is explicit through
`run_with_deferred_injection` or `with_deferred_injection`: it applies branch
corrections at their original stream locations, then performs final
basis-updating projections in `input`, `middle_out`, explicit, or greedy
`min_span` order. Basis-updating measurement uses a tree-distance-aware
Clifford localizer before absorbing the inverse Clifford into the tableau;
`disentangle=True` selects the basis-updating measurement path; the legacy
`absorb_basis=True` keyword remains accepted as an alias and is never silently
treated as a fixed-basis projector.

The safe MPS naming compatibility surface includes `from_mps`,
`from_tableau_and_nu`, `run(..., progbar=True)`, `apply(..., progbar=True)`,
`expectation_pauli_sum`, and tree truncation accessors. MPS-only
`apply_layout` remains chain-specific. Tree also exposes `from_stim`,
`analyze_stream`, `queued_stream_analysis`, `current_frame_layout`,
`apply_frame_layout`, `sample_basis`, `sample_bits`, `sample_bitstrings`,
`probability_bits`, `probability_bits_many`, `iter_sample_bits`, and
`iter_sample_bitstrings`, plus `amplitude`, `probability`, and `cap`.
Sampling accepts `basis="X"`, `"Y"`, or `"Z"`, a per-qubit pattern such as
`"XYZ"`, or `basis="random"`; returned columns remain physical-qubit
indexed. The conditional projections behind all of these methods stay
Tree-native and share collapsed TTN prefixes, rather than copying the MPS
chain sampler. The same event constructors are available on TreeStab as on
MpsStab: `submpo_event`, `submpo_event_parts`, `is_submpo_event`,
`subtreempo_event`, `subttno_event`,
`measure_event`, `reset_event`, `measure_reset_event`, and `cap_event`.
Sampling is not bounded by `max_dense_sample_qubits`; that constructor
argument remains accepted for compatibility with older callers.

`StabilizerTreeSimulator.run` also accepts the shared shot and MPI options:

```python
result = sim.run(
    shots=1_000_000,
    seed=7,
    mpi=True,
    workers="auto",
    progress="auto",
    retain="none",
)
```

Each shot starts from an independent copy of the current tableau and
coefficient tree; the caller's state and queued stream are not consumed.
Local replay supports `strategy="independent"`, `"coalesced"`, and `"auto"`,
including state-dependent Kraus channels. MPI `"auto"` resolves to independent
replay to preserve stable shot seeds across rank counts.

Trajectory events and native stochastic entries are compiled when they enter
the TreeStab queue, so they can also be replayed through `run(shots=...)`.
Do not combine stream-local trajectory events with the separate
`error_model=` convenience macro.

`cap(where, vec)` contracts one physical qubit with a length-two vector and
compacts the remaining labels, matching the MPS physical-cap semantics. It is
a correctness-first dense-state fallback with hierarchical TTN factorization,
so replay rejects states larger than `max_dense_cap_qubits`; it does not build a
full `2**(n-1) x 2**(n-1)` replacement operator. Scalable DEM-style capping
should still use structured weighted-XOR or coin streams. A cap resets the
stabilizer frame to identity and cannot be combined with a static STN frame
layout.
Its factorization does not truncate; the rebuilt TreeOptimizer performs
compression with the configured `chi`, `cutoff`, `cutoff_mode` and algorithm.

An entry such as ``("submpo", mpo, where)`` acts directly on the coefficient
state ``|p>`` in the same way as the MPS STN API; it is not conjugated through
the physical Clifford frame. The payload must expose a usable MPO interface,
or TreeOptimizer may lower it to a bounded dense operator. A complete
``TreeMPO``/TTNO can instead be scheduled with
``StabilizerTreeSimulator.subtreempo_event(tree_operator)`` (or ``subttno_event``). Its
TreePlan must match the coefficient tree and its declared support must include
all TreePlan physical sites, or exactly the complete operator's explicit
``operator_support`` (for example, one produced by ``TreeMPO.from_gate``); the
coefficient backend then contracts the
operator's internal virtual bonds with Tree-native QR routing and one final
configured compression sweep. Set ``track_norm=False`` for a non-unitary
operator, since its physical norm change is not a compression-fidelity loss.
For an arbitrary physical operator, pass ``(matrix, where)`` instead so
TreeStab can Pauli-map it through ``C† G C``; large dense physical matrices
remain subject to ``max_operator_qubits``.

Feed-forward uses `("if", record, bit, action)`, with Stim-style negative
measurement offsets and computational bits (`+1 -> 0`, `-1 -> 1`). The action
must be one gate entry. Stim `CX/CY/CZ rec[k] q` instructions are lowered to
this same form by `compile_stim_circuit`.

TreeStab derives `backend`, `dtype`, and `device` from every live coefficient
TTN tensor, including a caller-supplied Torch, JAX, or CuPy tree when
`to_backend` is omitted. `backend_info()` refreshes the same public
`backend`, `backend_dtype`, `backend_device`, and `array_backend` attributes.
Explicit matrix gates, every tensor in coefficient-frame sub-MPOs, and native
TreeMPO tensors are checked at the stream boundary. A foreign backend or device
payload raises; non-NumPy dtype mismatches also raise, while NumPy-to-NumPy
dtype promotion remains compatible.
`TypeError`; prepare it explicitly with the same converter used for the
coefficient TTN. Stim gate classification remains a NumPy-side operation,
while TreeOptimizer applies the coefficient update on the inferred backend.
Stim and trajectory-generated matrices are converted by the library before
they enter this user-stream boundary.

Named coefficient rotations and Pauli sums are constructed on the state
backend through Autoray. Exact cooling also keeps the leaf matrix on that
backend: one thin SVD supplies its candidate vector and rank test, followed
by backend Bloch reductions. Only the Boolean rank decision and three Bloch
values cross to the CPU tableau. The cooling algorithm remains enabled by
default and preserves its existing tolerance and canonicalization rules.

The coefficient tree retains norm/log-fidelity scalars on-device during
ordinary replay. Public diagnostics/readout and explicit extracted-exponent
bookkeeping materialize host scalars. Profiling, FIT finite/overlap checks,
and spectrum/bond diagnostics remain off by default. MPI shot replay now
also defaults to `collect_diagnostics=False`. Measurement decisions, dense
physical gate classification, native QR safety, and upstream rank selection
can still synchronize; explicit reporting/injection runners retain their
existing timing reports.

TreeStab's `norm_diagnostics()` keeps two coefficient-state diagnostics
separate. `local_fidelity` and `cumulative_fidelity` are cheap
canonical-centre compression metrics measured from retained norms, controlled by
`track_infidelity` and available independently of `track_truncation`.
`norm`/`state_norm` report the live represented coefficient-Tree norm;
`cumulative_norm` is the square-root retained-compression proxy, and
`norm_survival` is the explicit norm-derived alias of `cumulative_fidelity`.
The `current_segment_*` and `truncation_*` fields remain the optional
spectrum/discarded-weight diagnostics and should not be compared directly with
the norm-derived `current_*` fidelity fields.
`track_truncation=True` additionally enables the expensive
per-edge singular-spectrum and discarded-weight records returned by
`truncation_report()` and `get_infidelity_samples()`. Neither metric is a
physical target-state overlap; the tableau frame does not turn a norm proxy
into directional fidelity.

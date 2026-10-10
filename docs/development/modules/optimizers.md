# pepsy.optimizers

This package contains the high-level optimizers that sit above Pepsy's tensor,
operator, boundary, and solver layers. The core package remains `pepsy`;
`pepsy_examples` is for examples and external testing, and `tc_gauge` is an
important downstream time-compression consumer that depends on Pepsy behavior.

## Layout

- `_stream_events.py`: shared control, conditional, and sub-MPO parsing.
  Geometry and tree replay use this module directly, without importing MPS
  replay. Existing parser imports from `mps.optimizer` remain available.
- `mps/`: MPS gate-stream optimization.
  - `__init__.py`: lazy public exports; selecting layout helpers does not load
    replay, Gibbs preparation, or MPO optimization.
  - `optimizer.py`: `MpsOptimizer`, including public replay orchestration,
    option validation, live-state canonical metadata, FIT targets, and rollback.
    `run()` prepares the layout, resolves mode options, then executes the
    prepared stream; shot routing and the empty-stream path stay at the entry
    point.
    `_run_dmrg` owns stream counters, normalization cadence, quality checks,
    and progress. `_run_dmrg_single_window` and `_run_dmrg_batch_window` own
    exact-target/guess preparation, FIT, rollback, and the norm/center commit
    for their respective windows. They retain calls through the optimizer's
    existing hooks.
  - `_execution.py`: mode dispatch, control-delimited gate replay, and the
    execution scope that restores temporary layouts on success or failure.
    Its operations are bound onto `MpsOptimizer` and call the live instance's
    hooks, so the class retains state ownership and subclass dispatch.
  - `_controls.py`: measurement, reset, cap, and conditional control execution.
  - `_norm.py`: represented scale, local normalization, and norm diagnostics.
    Both receive the live optimizer and call its hooks; canonical state remains
    on the optimizer. Public methods retain their signatures and documentation.
  - `_streams.py`: immutable stream snapshots, symbolic gate resolution, and
    queue normalization. State/backend validation stays on the optimizer;
    trajectory grammar stays in the shared noise implementation.
  - `_exact_batch.py`, `_exact_structured.py`: bounded dense gate fusion and
    structured exact replay on supported arrays.
  - `layout.py`: gate-stream layout search and `MpsGateStreamSchedule`.
  - `_layout_execution.py`: optimizer layout installation, logical/physical
    mapping, replay-queue preparation, reordering, schedule installation, and
    logical readout. Queue preparation resolves plans and event order before
    the live MPS is reordered. Functions
    receive the live optimizer and call its canonicalization, native-swap,
    stream-validation, and replay hooks. Public methods retain their signatures
    and documentation on `MpsOptimizer`; private method aliases preserve
    descriptor binding and subclass overrides without adding a base class.
  - `gibbs.py`: purified finite-temperature `GibbsMps` preparation.
  - `diagnostics.py`: private, dependency-free layout formatting and FIT timing
    summaries. Timing collection and numerical diagnostics remain on the optimizer.
  - `compression.py`: stateless Quimb compression adapters, method option
    groups, and disposable `guess` / `svd_guess` construction. These helpers
    can load without initializing MPS replay or FIT. Historical imports from
    `optimizer.py` remain aliases to the same objects.
  - `normalization.py`: empty compatibility import path with no numerical
    imports; internal normalization execution is in `_norm.py`.
- `mpo/`: MPO gate-stream optimization.
  - `optimizer.py`: `MpoOptimizer`. Public `run()` prepares options and the
    replay snapshot; `_run_dmrg_replay` owns FIT scheduling, failure records,
    and fallback; `_run_compression_replay` shares direct/SVD success and
    rollback handling. Numerical kernels and canonical state stay on the
    optimizer. `_replay_policy` resolves the finite-check alias and scopes
    temporary caches before entering `run()`.
    Within `_run_dmrg`, local callbacks own FIT schedules and per-update
    recovery. `_run_dmrg_single_window` and `_run_dmrg_batch_window` prepare
    targets and guesses, invoke that transaction, and record successful FIT
    norms and diagnostics. Channel events, batching, sample points, and
    progress remain in the driver. These helpers are separate from the MPS
    helpers because operator-layer norms and recovery have different contracts.
  - `targets.py`: extraction target for gate-pair and DMRG target builders.
  - `compression.py`: extraction target for compression backends.
- `tree/`: `TreeOptimizer` gate replay and controls, `TreeTensorNetwork` state
  and canonical metadata, `TreePlan` / `TreeLayoutFinder` geometry and search,
  and `TreeMPO` operators. Private helpers have explicit state/value inputs:
  `_policy.py` resolves modes and lists retained configuration settings;
  `_application.py` validates operator regions and prepares local targets;
  `_readout.py` computes projected-amplitude Born weights; `_diagnostics.py`
  constructs records without contracting tensors. `compression.py` owns
  SRC/SDC environments; variational sweeps use the shared `pepsy.fitting.TreeFIT`.
  See the [consolidation record](../plans/tree_optimizer_consolidation.md) for
  ownership, compatibility boundaries, and validation.
- `peps/`: PEPS/PEPO gate-stream optimization.
  - `optimizer.py`: `PepsOptimizer`.
  - `gates.py`: extraction target for gate routing and target application.
  - `warmstart.py`: extraction target for warm-start construction.
  - `routing.py`: extraction target for sweep/global backend routing.
  - `diagnostics.py`: extraction target for infidelity/progress records.
- `sweep/`: local PEPS slice optimization.
  - `optimizer.py`: `SweepOptimizer`.
  - `_als.py`: fixed-rank one-site ALS within each row/column, using the
    existing boundary stores and directional strip cursor, with default
    explicit Hermitian metrics and iterative CG; matrix-free CG, direct solves,
    and spectral pseudoinverse are alternatives.
  - `_als_cg.py`: native Hermitian environment actions, balanced environment
    halves, Jacobi-preconditioned CG, residual and curvature checks.
  - `_als_lbfgs.py`: one-site quadratic L-BFGS with analytic complex gradients,
    SciPy host vectors, native environment actions, and residual diagnostics.
  - `environments.py`: Quimb MPS boundary store and engine selection helpers.
  - `local_objective.py`: extraction target for local objective assembly.
  - `traces.py`: extraction target for sweep traces and progress summaries.
- `stabilizer_tn/`: `StabilizerMpsSimulator` / `MpsStabOptimizer`, `STNState`,
  and typed STN diagnostic records for the Stim-tableau plus coefficient-MPS
  simulator. `_advice.py` owns stream analysis and recommendations; `_layout.py`
  owns coefficient-frame layout tracing and installation; `_stream_helpers.py`
  owns shared parsing and Clifford localizers. The simulator retains execution
  state and method contracts, including classmethod/subclass dispatch.
  See `../plans/stabilizer_tn.md` for its implementation record and
  `docs/howto/stabilizer_tn_magic.md` for exact cooling, greedy checkpoints,
  and immediate versus deferred MAST injection.
- `planning.py`: non-executing physical-versus-stabilizer and
  MPS-versus-tree circuit advice using measured frame supports and explicit
  chi-scaled work proxies.
- `qmera/`: schedule-first qMERA local-energy objectives, parameter
  dictionaries, compiled lightcone contractions, schematics, and
  Symmray-native fermion helpers. The main spin 1D builder uses retained
  registers, boundary disentanglers, brickwall or cyclic ladder pair rounds,
  and explicit Z₂ or unrestricted Pauli pair templates selected by
  `pair_ansatz=`. `system_size=N` names the 1D
  site count; 2D and explicit-mode fermions retain their existing schedules.
- `global_opt.py`: whole-network variational optimization helpers.
- `noise.py`: public trajectory records, native channel parsing, independent
  and coalesced replay, importance weights, and result aggregation.
- `_stim_compile.py`: private Stim instruction parsing, small Clifford matrix
  caching, and plan construction. `noise.compile_stim_circuit` is the public
  entry point; record classes stay in `noise` for import/pickle compatibility.
  Record constructors are imported at call time to avoid a module cycle.

MPS layout and replay both use `_stream_events.py` for event-name and sub-MPO
parsing. MPO timing reuses the dependency-free MPS diagnostics summary.

Entries described as extraction targets are proposals, not implemented
subsystems or instructions to begin a refactor. Inspect their source before
placing changes; retain existing public paths until a compatibility review.

## PEPS optimizer stack

`PepsOptimizer` is the outer gate-stream driver. It builds exact two-site
targets, compresses warm starts to the requested PEPS bond dimension, and then
optionally refines with `SweepOptimizer` or `GlobalOptimizer`.
It exposes `boundary_engine` and `boundary_options` so sweep cleanup can use
the same boundary implementation choices as `SweepOptimizer` directly.

`SweepOptimizer` is the local PEPS slice optimizer. It keeps two environment
stores:

- `bdy` for the trial norm `<state|state>`.
- `bdy_overlap` for the overlap `<target|state>`.

The current default store is `pepsy.boundary.states.BdyMPS`, whose `mps_b`
dictionary contains reusable boundary MPS entries keyed as `Y{i}_l`,
`Y{i}_r`, `X{i}_l`, and `X{i}_r`. `SweepOptimizer` selects a row or column,
attaches the needed left/right environments, optimizes the packed local slice,
and then advances the boundary for the next slice with
`pepsy.boundary.sweeps.CompBdy`.

## Boundary engines

The default dense Pepsy boundary engine is:

```text
build_bra_ket(...) -> BdyMPS(...) -> CompBdy.move_bdy/move_step_bdy(...)
```

That path uses local FIT/DMRG-style boundary updates and works well for the
dense backends it was designed around. It is less suitable for Symmray-backed
networks, where Quimb's native boundary contraction and environment routines
can preserve backend semantics better.

`SweepOptimizer` also supports `boundary_engine="quimb-mps"` (or `"auto"` for
Symmray-looking inputs). This builds local row/column environments with
Quimb's `compute_x_environments(...)` and `compute_y_environments(...)`, while
scalar sweep-time normalization and infidelity use Quimb's
`contract_boundary(...)` through Pepsy's public `method="mps"` metric helpers.
`PepsOptimizer(boundary_engine="auto")` keeps this same policy when it delegates
to sweep cleanup.

## MPS gate-stream optimizer

`MpsOptimizer` defaults to `direct` compression. Other replay modes include
`dmrg`, `swap`, `perm`, `svd`, `mix`, `exact`, and `exact-batch`; `batch-exact`
normalizes to `exact-batch`, while `mpo` remains a compatibility alias for
`direct`. For repeated evolution on a graph
with a useful one-dimensional layout, call `opt.apply_layout("quality")` once.
The MPS then stays in the selected physical order across `run()` calls and
logical readout goes through `opt.logical_order`, `opt.remap_sample(...)`, or
`opt.to_dense()`. During `mode="perm"`, `opt.qubits` is the synchronized
Quimb-compatible view of that same physical-position mapping.

The persistent reorder is free only for a product MPS (`p.max_bond() == 1`).
For an entangled initial state, the default is to raise; an explicit
`allow_lossy_reorder=True` opts into one reorder using the caller's cutoff.
The deprecated `run(use_layout_finder=True)` compatibility path still performs
the old temporary reorder and swap-back and should not be used for iterated
time evolution.

Canonical metadata in `opt.info_c["cur_orthog"]` is part of the numerical state.
Local expectation and norm diagnostics should move from this tracked range,
not rescan or contract the full MPS. Any target MPS copy needs isolated
metadata. Exact mode intentionally has no canonical cache; switching back to
an MPS mode rebuilds and canonicalizes the state.

`exact-batch` is opt-in fully contracted replay with automatic one-/two-qubit
fusion and compact diagonal broadcasting. Large mixed Z/ZZ diagonal runs
compact repeated supports before a one- or two-value graph-phase pass.
Same-pair parity-preserving gates have separate bounded-memory NumPy/Numba
and CuPy kernels; a pass-count and state-size check limits two-value runs
and GPU runs containing one-qubit factors. Other gate streams retain the original
fusion. It shares exact-mode restrictions, does not restore full-state
storage order between gates, and
falls back to the reference kernel for unsupported array/state types. See the
[implementation and upstream audit](../notes/mps_exact_batch.md).

## Import style

The `mps`, `mpo`, `peps`, `sweep`, `tree`, `tree_peps`, `energy`, and `qmera`
entry packages resolve exports lazily. Importing or listing one of these
namespaces does not load its numerical implementations. Each export resolves
directly to its owning implementation; historical child-module imports remain
available through compatibility aliases. Accessing an implementation loads its
actual dependencies.

Geometry-only callers can import `QMeraGeometry` from `qmera` or `TreePepsPlan`
from `tree_peps` without initializing NumPy, Quimb, or an optimizer. This does
not imply that every layout helper is dependency-free: for example, tree
layout currently uses MPS control parsing and numerical helpers.

Use clean class imports at API boundaries:

```python
from pepsy.optimizers import MpsOptimizer, PepsOptimizer, QMeraBuilder, SweepOptimizer
```

Use implementation leaves when a test or internal change needs module globals:

```python
from pepsy.optimizers.sweep.optimizer import SweepOptimizer
```

## Editing notes

- Prefer package namespaces such as `pepsy.boundary`, `pepsy.optimizers`, and
  `pepsy.tensors`; do not revive removed root-level flat modules.
- Add focused tests near the optimizer or boundary behavior being changed.
- For Symmray behavior, keep optional dependencies optional with
  `pytest.importorskip(...)`.

# pepsy.optimizers

This package contains the high-level optimizers that sit above Pepsy's tensor,
operator, boundary, and solver layers. The core package remains `pepsy`;
`pepsy_examples` is for examples and external testing, and `tc_gauge` is an
important downstream time-compression consumer that depends on Pepsy behavior.

## Layout

- `mps/`: MPS gate-stream optimization.
  - `__init__.py`: lazy public exports; selecting layout helpers does not load
    replay, Gibbs preparation, or MPO optimization.
  - `optimizer.py`: `MpsOptimizer`.
  - `_exact_batch.py`, `_exact_structured.py`: bounded dense gate fusion and
    structured exact replay on supported arrays.
  - `layout.py`: gate-stream layout search and `MpsGateStreamSchedule`.
  - `gibbs.py`: purified finite-temperature `GibbsMps` preparation.
  - `compression.py`, `normalization.py`, `diagnostics.py`: empty reserved
    import paths. Their proposed extractions have not happened; these
    responsibilities currently remain on `MpsOptimizer` in `optimizer.py`.
- `mpo/`: MPO gate-stream optimization.
  - `optimizer.py`: `MpoOptimizer`.
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
  - `environments.py`: Quimb MPS boundary store and engine selection helpers.
  - `local_objective.py`: extraction target for local objective assembly.
  - `traces.py`: extraction target for sweep traces and progress summaries.
- `stabilizer_tn/`: `StabilizerMpsSimulator` / `MpsStabOptimizer`, `STNState`,
  and typed STN diagnostic records for the Stim-tableau plus coefficient-MPS
  simulator. See `../plans/stabilizer_tn.md` for its implementation record and
  `docs/howto/stabilizer_tn_magic.md` for exact cooling, greedy checkpoints,
  and immediate versus deferred MAST injection.
- `planning.py`: non-executing physical-versus-stabilizer and
  MPS-versus-tree circuit advice using measured frame supports and explicit
  chi-scaled work proxies.
- `qmera/`: schedule-first qMERA local-energy objectives, parameter
  dictionaries, compiled lightcone contractions, schematics, and
  Symmray-native fermion helpers.
- `global_opt.py`: whole-network variational optimization helpers.

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
`dmrg`, `swap`, `perm`, `svd`, `mix`, `exact`, and `exact-batch`; `mpo` remains a
compatibility alias for `direct`. For repeated evolution on a graph
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
fusion and compact diagonal broadcasting. Large ZZ layers with one or
two distinct value pairs and same-pair parity-preserving gates have
bounded-memory NumPy/Numba and CuPy kernels; a pass-count and state-size
check limits the two-value route. Other gate streams retain the original
fusion. It shares exact-mode restrictions, does not restore full-state
storage order between gates, and
falls back to the reference kernel for unsupported array/state types. See the
[implementation and upstream audit](../notes/mps_exact_batch.md).

## Import style

The `mps`, `mpo`, `peps`, `sweep`, `tree`, `tree_peps`, `energy`, and `qmera`
entry packages resolve exports lazily. Importing or listing one of these
namespaces does not load its numerical implementations. Each export still
comes from its original implementation module; direct child-module imports
remain available. Accessing an implementation loads its actual dependencies.

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

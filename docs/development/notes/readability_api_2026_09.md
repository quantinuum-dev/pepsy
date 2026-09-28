# Readability and API review — 2026-09-28

Baseline: `develop` at `4e398e4`. Scope: identify concrete readability,
comment, and API improvements, then complete a first bounded improvement.

## Review coverage

Parsed all **211 Python modules** under `src/pepsy` with Python's AST:
**245,762 physical lines**, including comments and docstrings, before this
batch. Measured function bodies separately from their leading docstrings,
inspected signatures, and searched for identical function bodies.
These are review aids, not code-quality scores or performance measurements.

The inventory covered every namespace. Manual inspection concentrated on
MPS/MPO/tree replay entry points, FIT gate updates, noise result records,
NetKet packing/amplitude entry points, sampler exports and vector sampling,
and the existing module/API guides. This is not a line-by-line review of
every module or a new full functional audit. Bare counts of undocumented
functions include nested callbacks and property setters, so they do not
measure missing public API documentation.

## Ranked improvements

This is the initial ranking. Implementation and validation of the subsequently
approved items are recorded in [the follow-up below](#authorized-follow-up-implementation).

| Priority | Concrete finding | Action and acceptance criterion | Status |
| --- | --- | --- | --- |
| 1 | `VecSampler` documented a constructor `basis` argument that does not exist; `refresh()` claimed its backend was fixed, although it re-infers it. Query and batch methods have different conversion defaults. | Correct the contracts, show result shapes, normalization, source/cache lifetime, gradients, and basis/weight meanings. Keep signatures and executable bodies unchanged. | Completed in this batch |
| 2 | `_submpo_event_parts` is duplicated in `optimizers/mps/layout.py` and `optimizers/_stream_events.py`; FIT timing summaries are duplicated in MPS diagnostics and MPO replay. | Reuse the existing owner after checking parser dependencies, private monkeypatch users, exception text, and lazy imports. Existing layout/control and timing checks must pass. Add no generic utility layer. | Proposed |
| 3 | `MpoOptimizer.run()` has a 403-line body after its docstring and 39 explicit arguments excluding `self`. It mixes option preparation, snapshots, dispatch, fallback, and result bookkeeping. | Separate preparation from the replay transaction. Keep atomic restoration, fallback replay, external-reference semantics, subclass hooks, and the public signature. Run MPO replay, alignment, and channel suites. | Proposed |
| 4 | MPS `run()` has 72 explicit arguments excluding `self`; tree construction has 57. Many settings already have sound documentation but are hard to discover together. | Add concise option groups and minimal examples to existing guides: ordinary replay, compression/FIT, layout, diagnostics, and shots. Document persistent versus per-call settings. Keep current names/defaults; do not add a second configuration API simply to shorten signatures. | Proposed |
| 5 | `optimizers/noise.py` combines records, native stream compilation, Stim translation, and independent/coalesced runners across 6,754 lines. Similar result methods operate on different units: shots versus leaves. | First document result cardinality and estimator denominators consistently. Then consider extracting Stim compilation with preserved imports/serialized paths. Validate serial/parallel seeds, counts, measurements, and importance weights. Do not merge estimators merely because their formulas look similar. | Proposed |
| 6 | `vmc/netket.py` combines packing, amplitude construction, monitoring, and driver setup across 6,141 lines. Some public builders have one-line docstrings; two nested `evaluate_one` bodies are identical. | Document config/site ordering and parameter ownership first. Evaluate one amplitude helper with explicit captured inputs; preserve JAX tracing and optional imports. Run dense/native amplitude and VMC API checks. | Proposed |

The MPS refactor already completed is recorded in
[its handoff](../../../history/2026-09-28-mps-execution-extraction.md).
It does not mean all MPS kernels were simplified: `_run_dmrg` still has an
843-line body. `FIT.run_gate` has 604 lines and
`bp.series._local_expectation_open_loop_series` has 551. Their numerical
state and cache lifetimes need dedicated review before further extraction.
The earlier [organization review](module_organization_2026_09.md) also explains
why large cohesive MPO algebra and symmetry modules were retained.

## Comments and API conventions to apply during each change

- Explain why tensor order, normalization, cache invalidation, or rollback
  matters at the point where it is enforced. Preserve mathematical names in
  kernels; use descriptive names for orchestration and user-facing concepts.
- State shapes, ordering, mutation, return types, and optional backend needs
  on public entry points. Put one executable small example beside the API.
- Keep one recommended owning-namespace import. A compatibility alias is
  not a duplicate implementation; removal needs a downstream usage review.
- Consolidate identical bodies only after inspecting their dependencies and
  state. Identical nested closures can capture different environments.
- Preserve algorithm-specific meanings. A vector sampler's `weights` are
  joint probabilities; trajectory weights are likelihood ratios. Renaming
  either field would require a separate compatibility decision.

For example, MPO's `_replay_policy` resolves `fit_finite_check` before `run()`
executes. Reading only the assignment inside `run()` misleadingly suggests
that the alias is ignored. This is a navigation concern, not an established
bug; existing alignment tests cover alias conflicts.

## Completed first batch

Updated `sampling/vector.py` docstrings and the
[sampling API guide](../../api/sampling/samplers.md). Added a sampler chooser,
owning-namespace imports, a vector result table, and guidance on querying a
sampled batch with its resolved basis. No runtime dependency, numerical
algorithm, signature, return type, or alias changed.

Validation and commit status are recorded in the
[session handoff](../../../history/2026-09-28-readability-api-review.md).

## Authorized follow-up implementation

The user subsequently approved the parser, MPO, option-guide, and noise/VMC
priorities. The following bounded changes are implemented in this batch:

- MPS layout now imports the identical event-name, sub-MPO support, and
  sub-MPO event parsers from `_stream_events.py`. The local private names remain
  available. State/layout algorithms were not moved.
- MPO `run()` now has a 176-line implementation body, down from 403.
  `_run_dmrg_replay` retains the original FIT schedule/failure/fallback body;
  `_run_compression_replay` consolidates the three direct/SVD transaction
  blocks. Preparation, empty replay, option validation order, and snapshot
  timing stay at the public entry point. Timing summaries reuse the existing
  dependency-free helper rather than a second function and phase list.
- Existing MPS, MPO, and tree guides now group options by purpose and explain
  which settings persist. The MPO example and recovery table distinguish run
  atomicity, local FIT transactions, fallback, and layout installation.
- `_stim_compile.py` owns Stim instruction translation and its matrix cache.
  `noise.py` retains public record definitions and the compiler wrapper, so
  imports and serialized record paths remain valid. Compilation imports those
  constructors at call time; module import does not require Stim.
- NetKet's two identical eager physical-configuration evaluators now share
  one factory. Configuration mapping and fermion phases stay in their original
  adapters; JIT/vmap paths are unchanged. Noise result cardinality/weights and
  the VMC build/sample/measure/optimize workflow are explained in the guides.
  Packing and the general amplitude factory now document configuration order,
  parameter ownership, output shapes/scaling, and eager versus JIT execution.

These changes implement the selected boundaries. They do not claim a complete
rewrite of noise/NetKet or a redesign of all optimizer signatures. Broader
kernel and public-builder documentation work remains a future review topic.
See the [implementation handoff](../../../history/2026-09-28-parser-mpo-noise-vmc.md)
for checks and remaining environment limitations.

### Upstream compatibility review

Reviewed the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The Symmray array-documentation page was unavailable; installed code remains
the validation target. Installed versions: Quimb `1.15.1.dev66+ge927f06e1`,
Autoray `0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.4.1.dev8+gc45f91457`, JAX `0.8.2`, NetKet `3.22.4`, Stim `1.16.0`.
Inspected the installed `pack`, `unpack`, MPO sandwich, and Pepsy nonlocal-gate
signatures. No upstream call signature or compression policy changed.

- **Adopt:** existing shared parsers/timing summary and one eager evaluator.
- **Compatibility shim:** keep `noise.compile_stim_circuit` as the public
  wrapper and keep public record classes in their original module.
- **Defer:** dependency upgrades and upstream numerical/default changes. The
  unreleased Quimb changes do not authorize modifying this refactor's cutoff
  defaults or operator conventions.

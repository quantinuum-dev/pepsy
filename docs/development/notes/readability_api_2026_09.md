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

## Remaining-code review and FIT pass — baseline `f410fa0`

The earlier parser/MPO/noise/NetKet batch was committed and pushed as
`f410fa0`. A fresh AST inventory covers all **212 Python source modules**.
Manual follow-up examined FIT execution, BP observable support parsing and
open-series orchestration, Torch measurement/sampling closures, MPO product
term normalization, PEPO construction, and the published alias contracts.
This remains structural coverage plus selected manual review, rather than a
claim that every implementation line has been reviewed.

### Ranked findings

Body lengths below exclude leading docstrings and refer to this baseline.

| Rank | Concrete problem | Decision and check |
| --- | --- | --- |
| 1 | `FIT.run_gate` mixes validation, scheduling, convergence, failure timing, and final polish across 604 lines; `run_eff` mixes three execution routes across 446. | Implemented this pass: named execution methods and one shared native-kernel dispatcher. Keep numerical kernels, validation order, wrappers, and public signatures. Check existing schedule, cache, gradient, and fermion regressions. |
| 2 | FIT's gate docstring says fixed sweeps by default despite `rtol="auto"`; the guide describes `run_eff` as exclusively fixed despite its explicit stopping controls. Return/mutation contracts are hard to find. | Implemented: correct stopping descriptions, document `fit.p` ownership and returns, provide a small executable example, and compare chain/tree controls. |
| 3 | BP `series._term_sites` and `observables._term_sites` have identical bodies; the open expectation and partial-trace orchestrators are 551 and 437 lines. | Next BP pass: choose a small existing owner for support parsing after checking import boundaries; separate route/budget preparation from execution. Preserve observable ordering, normalization, cache keys, and budget outcomes. |
| 4 | Torch `measure_samples` has a 422-line body combining sample provenance, weights, connections, distributed work, and result assembly; two `run_sweeps` closures duplicate sampling/profiling counters. | Separate input/provenance preparation first. Share sampling bookkeeping only with explicit state ownership; validate saved Markov versus importance samples, gradients, chain shapes, and distributed reductions. |
| 5 | `MPOAutomaton.from_product_terms` combines term parsing, exact channel sharing, and emission across 409 lines. | Extract term normalization before channel assembly. Preserve exact backend-aware fingerprints and coefficient-slot ordering; compare dense operators and parameterized bases. |
| 6 | Dense PEPO plan `build` has a one-line docstring and 525-line body covering distinct cluster families and report assembly. | Document return/materialization contracts, then separate cluster-family assembly with shared allocator ownership explicit. Validate topology, residuals, symmetry, and dense references. |

MPS `_run_dmrg` (843 lines) and MPO `_run_dmrg` (553) remain large numerical
kernels. The completed public MPS/MPO orchestration refactors do not imply that
these kernels were reviewed line by line. Their numerical changes should be
reviewed separately from the FIT orchestration pass.

### Implemented boundaries and API decisions

- `run_gate` now has a **164-line implementation body** and `run_eff` has
  **98**, including their preserved validation. Gate scheduling and polish
  have separate owners. Full-chain block, cached one-site, and fermionic
  compatibility execution each have a named method.
- The common dispatcher performs no tensor conversion and adds no policy.
  Existing one-/two-/three-site kernels are called through `self`, preserving
  overrides and instrumentation. Convergence and rank/cache phase transitions
  remain with their original schedules.
- Removed dead commented-out norm calculations and replaced a misleading
  "normalize" comment with the reason scale must be retained. Added explicit
  input index/backend, mutation, return, and stopping contracts to FIT docs.
- Chain/tree argument differences are documented rather than renamed in this
  compatibility-preserving pass: inclusive chain intervals versus connected
  tree regions, one directional sweep versus paired passes, and `None` versus
  `self` returns. Both expose the fitted network as `fit.p`.
- Updated fitting examples to `from pepsy.fitting import FIT`. Reviewed root
  alias exports, API tests, and the migration guide. Searches of the available
  Pepsy examples, Gaugy, Gaugy examples, Tensy, and Tensy examples Python/notebook
  files found no selected FIT/legacy FIT-option aliases; that is not evidence
  that external users have migrated. Existing root aliases and named deprecated
  aliases retain their documented compatibility window. No public alias was
  removed and no replacement facade was added.
- Identical `flush`/`run_sweeps` closures capture mutable replay/sampling state.
  Their text alone is insufficient reason to introduce a general runner.

The upstream sources linked above were rechecked. Installed versions remain
the same; Symmray's array page is still unavailable. Installed `Tensor.split`,
`MatrixProductState.canonize`, and `Tensor.modify` signatures were inspected.
**Adopt:** existing native kernels and shared dispatch. **Defer:** upstream
upgrades/default changes and algorithm changes. No compatibility shim is needed
for this extraction; explicit FIT cutoff settings remain unchanged.

Validation and publication status for this pass are recorded in the
[FIT readability handoff](../../../history/2026-09-28-fit-readability.md).

## BP, Torch VMC, MPO, and PEPO follow-up

The user explicitly approved the four remaining domains from the ranking.
This pass builds on the uncommitted FIT work at `f410fa0` and implements all
four proposed extraction boundaries:

| Domain | Implemented boundary | Entry-point body before → after |
| --- | --- | --- |
| BP | One support parser in `_series_geometry`; separate native-cluster and explicit-edge executors for reduced densities and scalar expectations | Reduced density 437 → 150; scalar expectation 551 → 141 |
| Torch VMC | Retained-sample provenance, chain configuration, and amplitude preparation helpers; one sweep-counter implementation for both estimators | `measure_samples` 422 → 335 |
| MPO | Product-term validation, prefix construction, suffix equivalence, and transition/slot emission | `from_product_terms` 409 → 84 |
| Dense PEPO | Pair, star, path, and plaquette helpers with one shared allocator; separate report assembly | `build` 525 → 185 |

These lengths exclude leading docstrings and include orchestration calls;
they describe navigation improvements, not a runtime speedup. Helpers remain
in their owning modules. No dependency, configuration class, public alias,
test, or module was added to implement the extraction.

BP route checks, cache keys, budget outcomes, and graded cyclic contractions
retain their original owners and order. Torch distinguishes stale Markov
provenance from reusable importance proposals and preserves chain axes,
weighting, and distributed reductions. MPO keeps exact backend-aware sharing
and input-order coefficient slots. PEPO shares one allocator across cluster
families so residual subtraction sees the same previously assembled blocks.

The API guides now explain saved-sample ownership, MPO product-term shapes
and slot semantics, and PEPO returns, NumPy construction, report mutation,
and materialization. A direct comparison also confirmed the existing C4
restriction for complex evolution; this is documented rather than relaxed.

The upstream sources above were rechecked without changing dependencies.
Torch is `2.9.1`; the listed Quimb/Autoray/Cotengra/Symmray versions remain
unchanged. Inspected installed `D2BP.normalize_message_pairs`,
`D2BP.normalize_tensors`, `TensorNetwork.contract`, and `Tensor.split`.
**Adopt:** shared parsing/bookkeeping and explicit construction stages.
**Defer:** dependency upgrades and numerical/default changes. No new shim.

Validation and commit status are recorded in the
[four-domain handoff](../../../history/2026-09-28-bp-vmc-mpo-pepo-readability.md).

## Shared FIT tolerance follow-up

Continuing the user's readability cleanup, another AST scan found identical
`_resolve_fit_rtol` implementations in ordinary MPS, tree, and stabilizer-MPS
optimizers. Their scalar policy now lives in the existing
`pepsy._internal.cutoff.resolve_fit_rtol`; each optimizer retains its method
as a dispatch hook and reads its backend dtype only for `"auto"`.
The helper preserves float conversion, finite/nonnegative validation, error
text, and the three precision thresholds. Truncation cutoffs retain their own
thresholds in `dtype_auto_cutoff`. No source module or dependency was added.

Reviewing the callers exposed stale documentation: MPS non-unitary DMRG
keeps its automatic tolerance, while tree replay disables automatic stopping
for non-unitary/unknown-target-norm updates. Existing regressions explicitly
cover both behaviors. The API guides now describe the distinction; the
execution policies are unchanged.

Other identical bodies found by this scan include the two tree thread-limit
contexts, MPS/stabilizer random FIT data adapters, and two NetKet contraction
closures. They remain candidates for separate ownership/backend review.
This pass does not consolidate them based on textual similarity alone.

Validation and commit status are recorded in the
[tolerance handoff](../../../history/2026-09-28-fit-tolerance-policy.md).

## Shared random initialization and JAX contraction dispatch

The next pass reviewed two of the duplicate bodies identified above. Dense
MPS and stabilizer-MPS random FIT draws now use `fit_random_array` in the
existing private random utility. Both optimizer hooks remain. Each optimizer
still owns seed creation, tensor visitation order, and disposable guess
construction, including the separate native Symmray/fermionic route.
Sharing a draw helper does not imply identical random guesses across domains
or backends.

NetKet's spin and fermion JIT factories now share
`_make_jax_scaled_contractor`. The extracted closure preserves exact, HOTRG,
CTMRG, and boundary dispatch and the real dtype of the exponent. Configuration
mapping, fermion signs, output formatting, and `vmap` remain in the adapters.
The eager path retains its separate implementation and Python index handling.

**Adopt:** remove these identical policy/dispatch bodies in their existing
owning modules. **Defer:** numerical changes, dependency upgrades, and the
tree thread-limit contexts pending a separate ownership review. There is no
new compatibility shim. The continuing-task upstream audit remains applicable;
installed Autoray generator and random-array signatures were rechecked.

Validation and commit status are recorded in the
[random/JAX handoff](../../../history/2026-09-28-random-jax-readability.md).

## DMRG replay stages and connected-amplitude contracts

Following the user's Quimb comparison, MPS and MPO now separate their DMRG
replay drivers from single-window and batch-window transactions. The existing
optimizer modules retain ownership; no source module, generic configuration
object, or public alias was added. Helpers have explicit inputs and call
existing instance hooks. MPS retains its local rollback and represented-norm
policy; MPO retains its transaction callbacks, channel handling, and
operator-layer norm diagnostics.

The MPS driver body is now 250 lines, previously 843; MPO is 300, previously
553. Measurements exclude leading docstrings and describe code organization,
not speed. Individual window helpers still contain substantial domain logic;
this is not a claim that every numerical kernel is now small.

The Torch connected-amplitude docstrings now state input and output shapes,
ordering, parent-amplitude ownership, device/dtype requirements, mutation,
chunking, and cache/gradient lifetime. The API guide includes a small CPU
example. Both executable method bodies are unchanged.

The organizing pattern follows Quimb's
[DMRG driver/sweep/local-update separation](https://github.com/jcmgray/quimb/blob/main/quimb/tensor/tn1d/dmrg.py)
and its explicit numerical contracts. **Adopt:** named private execution
stages and useful contract documentation. **Defer:** numerical/default changes,
dependency changes, and broad API redesign. The continuing-task upstream audit
is reused; these extractions introduce no upstream calls or new shim.

Validation and commit status are recorded in the
[DMRG stages handoff](../../../history/2026-09-28-dmrg-stages-readability.md).

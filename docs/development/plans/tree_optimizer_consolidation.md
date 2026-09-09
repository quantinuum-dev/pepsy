# TreeOptimizer consolidation

Status: consolidation implemented on 2026-09-09; see the implementation record
below for retained boundaries and validation. Algorithms and public defaults
are preserved. This work follows
the implemented [path execution plan](tree_fit_path_traversal.md) and the
[correctness and API review](../notes/tree_api_consistency.md).

## Assessment

The main problem is responsibility and lifecycle duplication around the
algorithms. The algorithm count alone is not a reason to remove functionality.

Before consolidation, `tree/optimizer.py` had 8,541 lines, 210 class method definitions
(including property accessors), 83 distinct non-private method/property names,
and 54 constructor parameters excluding `self`. The constructor is 410 lines;
`apply_sub_mpotree` is 338; `run` is 389; `copy` is 90. These are structural
measurements, not evidence that every method is necessary or incorrect.

Concrete sources of maintenance cost:

- Requested modes are represented by `mode`, `compression_mode`, named DMRG
  aliases, and FIT options. Dispatch and progress reporting interpret these
  separately. Constructor/run resolution is already shared; extend that work.
- Ordinary gates already lower to `TreeMPO`, but several wrappers repeat
  support validation and update ownership. Begin/finish/abort calls appear in
  nine methods each. Successful installation, scale handling, and diagnostics
  are distributed across algorithm branches.
- `copy` manually reconstructs the constructor settings and then restores
  histories, labels, and several internal flags. FIT guesses, approximate
  readout, and shot templates have separate construction/RNG requirements.
- Layout search, backend preparation, controls, replay/MPI adaptation, resource
  estimates, plotting, and compression all live in the optimizer class.
- Historical notes contain superseded algorithm descriptions. For example,
  `tree_modes_review.md` describes SDC before the complementary-environment
  implementation. That record must not serve as current algorithm guidance.

Useful boundaries already exist: `TreePlan`, `TreeTensorNetwork`, `TreeMPO`,
the 215-line `compression.py` SRC/SDC kernel, and `pepsy.fitting.TreeFIT`.
Preserve them. TreePeps also uses TreeFIT; noise and MPI have shared runners.

## Final execution shape

Keep `apply_gate` as a convenient frontend and `apply_sub_mpotree` as the
primary operator boundary:

```text
gate -> TreeMPO --+
                 +-> validate and plan -> selected algorithm -> finish update
explicit TreeMPO-+
```

Validation resolves the operator representation, backend capability, logical
and physical support, limits, and exterior identity proof before state changes.
Planning identifies the active region and freezes the incoming path orientation
once. The plan contains geometry and policy, not stale tensor IDs or numerical
environments. Obtain live indices after QR or rank changes.

Keep four algorithm families behind this boundary:

| Family | Modes | Work that must remain distinct |
| --- | --- | --- |
| Exact preparation, then compression | `direct`, `dm` | Lossless operator routing followed by a canonical SVD or density-matrix split sweep. |
| Successive complementary environments | `src`, `sdc` | Original layered target, randomized sketches or deterministic low-rank environments, then nested projections. Keep the existing shared kernel. |
| Streaming compression | `zipup` | Absorb and truncate outgoing messages immediately, with its own intermediate-cut semantics. |
| Variational fitting | `dmrg1`, `dmrg2`, `dmrg3`, configurable `dmrg` | Exact layered target, independent guess, existing TreeFIT engine and block-growth/refinement schedules. |

For auto FIT traversal, a path gets endpoint-consecutive sweeps; a branched
region gets depth-first sweeps. Explicit `depth` and `depth-first` remain
available on every FIT support. This is based on induced geometry, including
few-body collinear supports, not gate arity. An arbitrary TreeMPO without a
valid exterior identity proof may require the full tree despite small declared
physical support.

Share the region and orientation, not every sweep order. Direct/DM can prepare
from both endpoints before their directional truncation; SRC environments run
opposite the projections; zipup has immediate cuts. Internal path nodes retain
all physical and exterior virtual legs. There is no conversion to a chain MPS.

One owner finishes each logical update. Each engine reports its resulting
represented scale and canonical state through the same internal contract.
The finalizer must not blindly add the operator exponent: FIT's layered target
already carries it. Keep norm-survival, spectral truncation, and FIT residual/
overlap diagnostics distinct; they do not estimate the same quantity.

This does not require a full state copy before every gate. Preserve in-place
kernels and existing private-result installation where appropriate. Preflight
rejections must precede mutation. Numerical failures after mutation require
cache/canonical invalidation and honest failure semantics; the current
`_abort_update` clears diagnostic aggregation, not tensor changes. Do not label
a context manager an atomic state transaction without implementing rollback.

## Ownership and API

Use a few private modules with explicit inputs and outputs, introduced only
when there is code to move:

| Owner | Responsibility |
| --- | --- |
| `optimizer.py` | Public API, live state/configuration, delegation to application, controls, and shared replay runners. |
| `_policy.py` | Pure option normalization and resolution used by construction, replay, copies, and display. |
| `_application.py` | Validated active-region planning, private local target preparation, and shared peel geometry. Numerical dispatch and compatibility hooks remain on the optimizer. |
| Existing `compression.py` and `pepsy.fitting.tree` | SRC/SDC numerical environments and FIT numerical environments/sweeps respectively. |
| `_readout.py` | Projected-amplitude probability kernels with explicit state/backend inputs; control orchestration remains on the optimizer. |
| `_diagnostics.py` | Norm, spectral, and update record construction; no tensor access or independent canonical state. The optimizer owns accumulation and retention. |
| Existing TTN, TreeMPO, and layout owners | Tensor/canonical metadata, operator representation/proofs, and geometry/search respectively. |

Keep layout convenience methods on the optimizer, but put search orchestration
behind the layout finder. Pass a trial-scoring callback where state replay is
needed so the layout module need not import TreeOptimizer. Keep the tree shot
adapter thin and use the existing generic noise/MPI engines.

Avoid mixin classes, a new plugin/strategy framework, and helpers that accept
the entire optimizer and access arbitrary private attributes. Moving those
methods unchanged into multiple files would hide rather than reduce coupling.
Do not split the large TTN/operator modules just to reach a line-count target.

Keep the constructor and documented public settings compatible. Resolve the
effective method into one small per-call policy, with one source for public
configuration; do not retain a second mutable configuration that can drift.
Copies and workers should use one configuration snapshot mechanism. A new
public options object or renamed parameters would be a separate API proposal.

Document the eight short algorithm names first, configurable `dmrg` as an
advanced option, and legacy route aliases in a compatibility section. Preserve
existing combinations and override precedence. In particular, `direct` currently
permits a separately chosen `compression_mode`; switching the route to direct
does not unconditionally reset that compressor. Simplification must not silently
change this behavior, default traversal, seeds, cutoffs, or FIT budgets.

## Copies, caches, and compatibility

Separate public copying from private work construction. Public `copy()` retains
its independent state, histories, queue semantics, and derived child-seed draw.
Private FIT/readout work needs the state and numerical settings without growing
histories. FIT guesses already omit histories, but still construct an entire
optimizer; eventually call the same application engine on a private state.
Readout must preserve the parent's RNG even on failure. Preserve existing seed
draw counts when replacing these implementations.

Trajectory branches require their own explicit retention policy: do not drop
diagnostics merely because a private copy is faster. The shared noise runner
already recognizes `_copy_for_trajectory_branch`; use that contract only after
checking the tree results and retention requirements.

Keep cache ownership narrow:

- Immutable geometry may persist in bounded caches tied to the TreePlan.
- Gate factors also depend on payload identity and backend; preserve invalidation
  on queue/payload changes and state/layout replacement. No per-gate host hashing.
- SRC/SDC numerical messages and last-use counters belong to one application;
  retain reuse and release within that application.
- FIT overlap messages belong to one target/guess and use dependency-based
  invalidation after changes. Reuse across sweeps is essential.
- Norm caches and canonical metadata belong to the TTN. Exterior identity
  certificates belong to TreeMPO and its explicit mutation contract.

Retain explicit `apply_1q`, `apply_2q`, `apply_subtree_operator`, and chain
`apply_submpo` entry points during consolidation. Their specialized factorized
kernels can have different finite-cap behavior from ordinary TreeMPO replay.
Isolate those kernels behind compatibility adapters if necessary; do not remove
them just because ordinary `apply_gate` no longer selects them.

There are narrower cleanup candidates: `_apply_gate_dmrg_impl` just forwards
to TreeMPO application; `_gate_route` ignores its width argument; two-/multi-site
fallbacks after its explicit sub-MPO rejection are unreachable with the built-in
resolver. Remove these only after auditing callers/overrides. Preserve covered
compression hooks used by integrations, including replaceable
`_compress_edge_with_diagnostics`. The target-building helper also has test
callers; migrate those to the actual target boundary before removing it.

## Implementation sequence and acceptance

1. **Establish the baseline.** Preserve the current uncommitted correctness work
   as the starting point; do not mix it invisibly with a large reorganization.
   Capture mode/alias precedence, finite-cap outputs, center/scale metadata,
   diagnostics, RNG behavior, and representative timings from this exact tree.
2. **Consolidate policy and construction.** Extract pure option resolution and
   one settings snapshot for copies/workers. Preserve the public signature and
   mutable-setting behavior. Validate normal replay, copies, and shot overrides.
3. **Extract responsibilities without changing algorithms.** Move diagnostic
   accumulation and readout helpers; delegate layout search. Keep the numerical
   operations, ordering, and update boundaries unchanged for this step.
4. **Unify the application lifecycle.** Introduce the shared per-call plan and
   single successful-update finalizer. Reuse existing kernels; remove duplicate
   lowering/validation only behind the validated internal boundary. Then replace
   disposable FIT optimizer construction using the same engine and seed policy.
5. **Prune and document.** Remove proven unreachable routing, isolate remaining
   compatibility kernels, and maintain one current implementation map and one
   public API reference. Mark superseded notes as historical with links forward;
   do not rewrite old benchmark results as evidence about current algorithms.

Each step should be independently reviewable and validated before the next.
Do not add algorithm changes, native capabilities, a new default, one-site
QR-carry skipping, or a performance promise to these refactoring steps.

Acceptance uses the existing tree API, path, FIT-message, SRC/SDC, zipup,
representation, measurement, and compression-hook suites, with focused additions
only for uncovered invariants. Check dense reference states/probabilities at
sufficient rank and baseline equivalence at finite caps, including actual
isometries, extreme represented scales, mutable operators, caps/labels, failed
calls, and supported native Symmray paths. Preserve seeded replay on the same
backend; do not compare raw gauge-dependent tensor entries as state equality.
Validate TreeStab, TreePeps/TreeFIT, shared trajectories, MPI unit adapters,
public imports, and package layout at the affected boundaries.

Use matched pre/post workloads to check total time, copies, peak memory, center
travel, and environment rebuilds: repeated one-site gates, long two-site paths,
branched few-body gates, finite-cap FIT, and controls with long histories.
Keep performance harnesses outside the package and distinguish profiling from
uninstrumented timing. Upstream audits and backend checks remain mandatory when
implementation work touches numerical paths. GPU and live MPI results require
their own runs.

Completion means one authoritative resolution path, one ordinary operator
boundary, explicit algorithm/caching ownership, one update finalizer, and no
duplicated constructor reconstruction. A smaller file is a consequence, not
the acceptance criterion.

## Initial proposal review

The initial review used source/caller searches, AST inventory, relevant tests,
and the existing implementation notes. That proposal changed documentation
only; implementation and validation followed after user authorization.

## Implementation record

The optimizer is now 7,733 lines. The primary `apply_sub_mpotree` method shrank
from 338 to 186 lines, layout optimization from 252 to 100, and public `copy`
from 90 to 32. Public constructor parameters/default expressions are unchanged.
These reductions come from explicit responsibility boundaries and removal of
duplicate work, rather than mixins or helpers receiving the whole optimizer.

All nine update entry paths use `_update`, which owns exactly one successful
completion or diagnostic abort across nested calls. Non-FIT operator scale is
added in one shared successful completion block; FIT keeps the scale already
carried by its target. The width-independent route helper, unreachable ordinary
two-/multi-site fallbacks, and trivial DMRG forwarding wrapper were removed.
Explicit low-level kernels and covered compression hooks remain available.

Copies, layout pilots, and FIT guesses use `_configuration_snapshot` and
`_new_state_optimizer`. `_copy_state` preserves labels, scalar ledgers, policy,
and the derived RNG without copying histories or the queued stream. Public
`copy` adds their independent retained copies; approximate MPO readout uses
fresh histories and restores the parent RNG even if its state fork fails.
Profiling appends only the new private-work events.

One ownership fix accompanies the extraction: installing a selected layout
clears both path and gate-factor caches. The previous code cleared paths only,
which could leave a cached operator mounted on a different TreePlan.

Two boundaries deliberately remain narrower than the initial proposal:

- Private state creation still uses the validated constructor. Bypassing it
  would require separately reproducing backend checks and unknown-canonical-
  state recovery. Removing those scans is a future measured optimization,
  not needed to remove duplicated configuration or history copying.
- Numerical dispatch and low-level compatibility kernels remain methods on
  TreeOptimizer. Extracting them together with every callback would recreate
  an optimizer-sized context. The pure planning/target helpers, separate
  SRC/SDC kernel, and shared TreeFIT already provide useful ownership without
  adding such a context. The test-used target builder remains a thin adapter.

The pre-refactor source was saved outside the repository before editing.
Characterization compares finite-cap states for all eight short modes, all
three FIT traversals, represented exponents, actual isometries, final centers,
RNG states, block schedules, mode/alias precedence, and constructor signatures.
Tests also exercise the existing native and cross-consumer boundaries.

### Validation results

The saved pre-refactor source and final source produced **14 bitwise-identical
finite-cap statevectors** and **1,004 identical characterization records**.
These cover the eight short modes, FIT traversal choices, canonical centers and
actual isometries, represented scales, block schedules, RNG states, and option
precedence. Constructor argument/default ASTs are identical. No numerical
engine or shared backend dispatch changed in this consolidation.

The affected Tree/operator/FIT/measurement/path/compression/sampler/TreePeps/
TreeStab/trajectory/MPI-unit/public-API/package-layout checks total **1,181
passed, 4 skipped**. The broad run initially passed 1,180 tests and found one
stale TreeStab test that instrumented the compatibility alias rather than the
primary `apply_sub_mpotree` boundary. The saved baseline failed the same test.
After updating that instrumentation, the changed test passed independently;
no TreeStab implementation change was needed.

The full repository run stopped after **182 passed, 1 failed** at
`test_native_reduced_loop_compression_uses_graded_svd_adapter`: BP gauge
construction reports a Symmray bond/vector-size mismatch. An isolated run
against the saved pre-refactor source reproduced that failure. It remains
outside this change. Ruff over `src tests` and `git diff --check` pass. GPU
performance and live multi-rank MPI were not tested.

### Matched CPU timings

Baseline and final source ran sequentially in the same environment, using
fixed seeds and complex128 on NumPy and Torch CPU. Replay workloads use a
24-qubit random balanced TTN with input and retained bond dimension four;
Torch and optimizer thread limits are one. Timings are medians of five runs
after one warmup, with construction outside the timed replay. The readout
workload uses a four-qubit state with 10,000 retained update records and seven
timed runs after one warmup. Allocation tracing was a separate run.

| Workload | NumPy before / after (ms) | Torch before / after (ms) |
| --- | ---: | ---: |
| 64 one-site gates | 88.979 / 88.496 | 141.246 / 142.298 |
| 16 long-path two-site gates | 54.316 / 54.735 | 76.664 / 77.507 |
| 8 branched three-site gates | 36.032 / 36.484 | 50.479 / 50.623 |
| 6 DMRG2 path/branch updates | 247.928 / 249.477 | 259.653 / 260.297 |
| Approximate readout, 10,000 records | 76.268 / 2.021 | 78.742 / 2.429 |

Replay medians changed by at most 1.3%; this is evidence against a material
regression on these workloads, not a general replay speedup. Final norms and
bond dimensions matched exactly for every timed replay case. Readout values
also matched exactly. Avoiding history copies made this long-history readout
about 38 times faster on NumPy and 32 times faster on Torch. Traced Python peak
allocations fell from 5,438,395 to 70,197 bytes on NumPy and from 5,444,063 to
81,901 bytes on Torch. These are Python allocations, not total native/device
memory. Large-bond and GPU scaling require separate measurements.

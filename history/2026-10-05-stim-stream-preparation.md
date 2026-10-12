# 2026-10-05 — Stim translation and explicit stream preparation

- Scope: user requested public Stim-to-gate-stream APIs, separate stream
  analysis/preparation, and simulator-owned shot execution for the Helix notebook.
- Branch / baseline: `develop` / `dec6960`.
- Commit status: working-tree changes; no commit or push.

## Changes

- Added public `stim_plan_to_gate_stream` and `stim_readout_parities` under
  `pepsy.optimizers`, with root convenience exports, typing declarations and
  manifest entries. Translation consumes `StimCircuitPlan` and retains
  independent Pauli noise as unsampled stream entries. Readouts retain
  detector order and XOR-combine observable annotations by logical ID.
- `StabilizerMpsSimulator.compile(stream)` explicitly prepares/replaces its
  queue without execution. `set_gates` remains equivalent; already compiled
  trajectory plans are reused by identity and exposed via `compiled_stream`.
  Threaded shot dispatch and workers now reuse the prepared plan.
- Analysis counts stochastic channels and their support separately from opaque
  entries; known random Clifford channels retain Clifford-only classification.
- Stim parsing supports grouped record-controlled gates and empty readout
  annotations. Nonzero measurement readout noise, inverted measurement results
  and `MPAD` now reject rather than silently discarding classical semantics.
- Updated [noise guide](../docs/api/optimizers/noise.md) and
  [STN guide](../docs/api/optimizers/stabilizer_tn.md). The authorized external
  Helix notebook now imports the public functions, analyzes separately, prepares
  with `engine.compile`, and executes with `engine.run(shots, strategy, workers)`.
  Its local helper only loads builders and groups imports; saved current
  outputs and unrelated `.DS_Store` edits were preserved.

## Validation

- New [regressions](../tests/test_stim_gate_stream.py): 39 passed. Checks fresh
  faults, independent/coalesced/auto replay, repeatability, multiplicities,
  unchanged parent state, all 15 two-qubit Pauli outcomes against Stim,
  hidden-reset offsets, repeated observable IDs, grouped controls, empty
  annotations, explicit rejection, plan identity in threaded workers, pickle
  serialization and failed-preparation queue preservation.
- Existing trajectory/STN/public API/package/import-boundary selection:
  612 passed, one skipped. Available backend/compression/tree/sampling selection:
  365 passed, 26 skipped (missing accelerator hardware/CuPy or extra JAX CPU
  device). The selections overlap; these numbers are not a distinct-test sum.
- Ruff, whitespace, changed API guide file links and notebook schema/syntax
  checks passed. Exact notebook engine cells ran successfully with synthetic
  Stim fixtures for independent and coalesced strategies, without saving
  fixture results as Helix results.
- Full suite audit found two failures in `test_ham.py` and was interrupted
  after reproducing both on an isolated checkout of unchanged `dec6960`.
  Baseline run with `--maxfail=2`: 1017 passed, 12 skipped, two failed.
  Both tests pass when selected alone on both checkouts. In full test order,
  complex Torch delinearization sees a registered real-only SVD and raises
  `SVD_real requires a real Torch tensor`. This existing registration/order
  issue is outside Stim preparation; it was recorded, not masked or fixed.
  No full-suite success is claimed.

## Decisions and limits

- Adopt existing trajectory-plan and simulator backend-validation contracts;
  no compression, state-update algorithm, dependency or installed-library edits.
- Independent Pauli/depolarizing channels lower to the generic stream API.
  Correlated/heralded channels remain supported by native Stim runners and
  explicitly reject in this translator. New classical readout primitives
  are deferred; physical premeasurement Pauli noise is not advertised as
  equivalent to arbitrary readout errors with later feedback/reset.
- Readout arrays are raw parities, without Stim reference-sample subtraction
  or decoder correction. Coalesced averaging requires result counts.
- Reviewed the [official Stim gate reference](https://github.com/quantumlib/Stim/blob/main/doc/gates.md).
  Environment: Pepsy 0.5.0, Stim 1.16.0, Quimb 1.15.1.dev66+ge927f06e1,
  Autoray 0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f,
  Symmray 0.4.1.dev8+gc45f91457; same numerical dependencies as prior session.
- Original Helix builders remain unavailable; a full regenerated Helix circuit
  has not been run. No real multi-rank MPI or unavailable accelerator checks.

## Follow-up semantic review requested by the user

Review only; no runtime fixes applied. Read the current Stim compiler, native
shot runners, trajectory measurement metadata, annotation resolution, stream
preparation and advice alongside the official Stim gate reference. Fresh tiny
Stim/Pepsy probes confirmed these findings that existing passing tests missed:

1. `stim_readout_parities` drops all records with `reset=True`. Independent
   trajectory metadata uses that flag for user-visible `MR`, while coalesced
   metadata uses it only for hidden bare-reset collapses. For
   `R 0; X 0; MR 0; M 0; DETECTOR rec[-2]`, actual measurement bits are `[1,0]`,
   but independent readout parity returns zero and coalesced returns one.
   A detector immediately after MR instead raises an unavailable-measurement
   error independently. The current notebook uses each retained simulator's
   native measurements and avoids this flag ambiguity.
2. Native Stim runners do not insert herald bits into the measurement record
   used by detectors and feed-forward. For
   `R 0 1; M 0; HERALDED_PAULI_CHANNEL_1(1,0,0,0) 0; DETECTOR rec[-1];
   CX rec[-1] 1; M 1`, Stim gives `[0,1,1]`; both native Pepsy runners retain
   herald one separately, report detector zero and final measurement zero.
   A detector referring to `rec[-2]` after herald then M instead raises due to
   omitted herald slots. The unsampled translator correctly rejects heralds;
   its documented native fallback needs these semantics fixed.
3. Returned ideal matrices alias writable `_STIM_UNITARY_CACHE` arrays.
   Overwriting the matrix from `compile_stim_circuit('H 0')` with identity
   makes the next H plan use identity. The probe restored the original matrix
   immediately. Protect private cache payloads or isolate exported copies.
4. Classical control lowering accepts invalid `CX 0 rec[-1]` and rewrites it
   as a classical X on qubit zero; Stim rejects record editing. Valid reverse
   control `XCZ 0 rec[-1]` is rejected by Pepsy. Validate the classical wire's
   allowed position and lower supported reversed controls explicitly.

Preparation/advice improvements also reproduced:

- `engine.compile` accepts unknown named gates and sites outside the initial
  register. Matrix backend and noise probabilities are checked, but static
  event semantics should be validated before committing a prepared queue.
- A compiled `CX rec[-1] 1` action is classified as opaque; the advisor misses
  its target, reports no control entry, and warns that support is unknown.
  Analyze conditional actions recursively while preserving conservative
  classification for conditional non-Clifford operations.
- Twenty identical depolarizing events build twenty separate channel objects
  and trigger eighty matrix classifications in analysis. Reuse immutable
  channels and cached classification after protecting mutable payloads.
  This is an operation-count observation, not a measured runtime speedup.

Recommended order: distinguish visible measurements from hidden reset
collapses; implement a consistent classical record for measurement/herald/
readout events and feedback; protect compiler cache matrices; strengthen
static validation and conditional analysis; then deduplicate preparation work.
Keep the accepted translate/analyze/compile/run API.

## Notebook follow-up: deferrals and shared layout search

- Revised the Helix notebook and its helper to document the deferred classical
  record features and avoid recommending native herald fallback. Added a
  pretranslation check for unsupported classical wire directions. Generated
  cache-backed matrices are not modified.
- Added a layout cell before replay. A temporary independent-Pauli-free
  planning skeleton uses `current_frame_layout(order="quality")`, which already
  delegates weighted coefficient-frame supports to the shared
  `MpsGateStreamLayoutFinder`. Installs the returned plan with `apply_layout`
  while the coefficient MPS is product, preserving the prepared noisy stream.
  Feedback layout planning explicitly rejects; static locality metrics are
  not runtime/accuracy claims. Ordinary-MPS replay pilots remain deferred.
- New synthetic check found incomplete native measurement histories on some
  coalesced leaves, with and without layout: a visible MR record disappears
  from one retained simulator while the batch record remains complete.
  Consequently the notebook defaults to independent replay and explicitly
  defers auto/coalesced execution rather than mixing ambiguous record paths.
  This underlying history bug is not fixed by the notebook change.
- Existing static-frame layout tests: 13 passed. Helper Ruff passed. Full
  notebook-cell synthetic checks preserve saved outputs and validate schema,
  syntax, feedback guards, layout persistence and identical visible records
  with/without layout for independent replay with one and two workers.
  These are synthetic checks, not regenerated Helix results. Full
  Helix regeneration remains blocked by unavailable original builder sources.
  Changes remain local and uncommitted.

## Second notebook review

Review only; notebook/helper behavior was not changed. New exact Stim-reference
probes passed for X/Y/Z MR, supported MR-followed-by-feedback, and MPP/MXX/MZZ.
Executed the notebook's batch-summary and disabled-decoder cells on a synthetic
fixture; schema and code syntax passed. Helper Ruff and Pepsy diff checks passed.

Remaining notebook integration findings:

- The STN adapter feeds every frame record to the shared finder as `submpo`.
  Although `frame_events` retain their kinds, `site_usage` loses measurement
  and reset boundaries and `qubit_roles` is empty on the checked fixture.
  Thus it shares weighted quality search/refinement but does not inherit the
  MPS QEC lifetime/role improvements in full. Coefficient-frame role evidence
  needs deliberate integration; physical data/ancilla labels cannot simply be
  copied to an evolving coefficient basis. Compression objectives/pilots and
  ordinary gate scheduling remain explicitly deferred.
- The noise cell builds a DEM even when `setup_decoder=False`. A replayable
  `R 0; H 0; M 0; DETECTOR rec[-1]` fails there with nondeterministic detectors,
  while independent quantum replay succeeds. Defer DEM construction to the
  decoder branch rather than weakening Stim's detector validation. This is a
  synthetic failure mode, not a demonstrated property of the missing Helix
  builder's output.
- Separate analysis of the raw stream and engine compilation prepare channels
  twice (two compile calls in the instrumented probe). Inspect the installed
  plan through `queued_stream_analysis()` after compilation to reuse prepared
  channels. No runtime speedup was measured.

Original builder files are still unavailable, so none of these checks establishes
that a regenerated full Helix circuit runs successfully. No commit or push.

## Implemented fixes following the second review

- Shared a single lifetime/role metadata resolver within the MPS layout finder.
  STN frame planning now passes measure/reset/MR kinds to that resolver and
  lifetime candidate generation, while retaining sub-MPO cost types and
  weighted coefficient supports. Plans label these as coefficient-support
  reuse hints, not physical data/ancilla classification. Ordinary optimizer
  layout uses the same resolver. Replay pilots/scheduling remain deferred.
- Queued STN analysis and recommendations reuse the installed trajectory plan.
  Regression instrumentation verifies the same plan is returned by identity,
  avoiding reconstruction of noise channels.
- Notebook preparation precedes `engine.queued_stream_analysis()`. The noise
  cell no longer builds the DEM; optional decoder setup constructs it only
  when requested. Existing outputs remain preserved.
- Validation: 70 selected MPS/STN layout and queued-analysis tests passed;
  all 40 Stim-stream tests passed; 54 public API/package-layout tests passed.
  These selections overlap; counts are not additive. Ruff `src tests` passed.
  Synthetic exact notebook-cell replay verifies unchanged measurement records
  with/without layout for one and two workers, and reaches replay with a
  nondeterministic-detector fixture when decoding is disabled. Schema, AST,
  output preservation and cell ordering checks passed.
- No dependencies or numerical compression policies changed; reuse the same
  active-task upstream audit and recorded versions. No full-suite rerun; the
  earlier full-order baseline failures remain documented above. Original
  Helix builders remain missing. Fixes are local, uncommitted and unpushed.

## Missing-builder notebook error reported by the user

- User confirmed the active saved failure is missing `qecc2026Stim` and
  `circuit_utils`. Rechecked local file names and repository tracked files;
  neither original module is present. No original builder was reconstructed.
- Helper now respects `HELIX_MAIN_ROOT` and loads an exported `.stim` circuit
  through `load_stim_circuit`. Relative files resolve beside the helper. The
  notebook's explicit file-input path bypasses builders and noise insertion;
  its supplied circuit already defines the intended operations/noise/readouts.
- Missing inputs now give a clear unavailable status and guard all dependent
  simulation cells. Dependent state is cleared before input loading to prevent
  stale results after failed reruns. No replacement experiment is supplied.
- Fresh-process execution of all notebook code cells without builders reaches
  the unavailable status without cascading exceptions. Full code-cell replay
  with an exported synthetic noisy fixture verifies MR readouts and shot counts;
  import-only fixture verifies environment-variable builder discovery. These
  temporary fixtures are not actual Helix sources or saved Helix results.
  Notebook schema, syntax, saved-output preservation and helper Ruff passed.
- Full original Helix construction still requires the missing files, or an
  exported circuit from its original builder. Restart the kernel before rerun
  so cached helper imports refresh. Changes remain local and uncommitted.

## Auto replay and backend-aware local workers

- User requested removing the compiled-stream identity assertion, applying the
  layout plan, and enabling automatic shot strategy. Notebook now uses
  `shot_strategy="auto"` and `workers="auto"`, applies the plan before replay,
  prints actual independent/coalesced representation and backend/device, and
  treats retained counts as shot multiplicities. Native visible records are
  used for both representations. Saved outputs remain preserved.
- Coalesced branch cloning now copies native measurement-prefix lists even
  when public optimizer `copy()` intentionally resets history. Regression
  covers an MR prefix, later genuine branch, rec[-2] feedback and raw readout
  with auto/coalesced replay on one and two workers.
- Local StabilizerMpsSimulator dispatch defaults to `parallel_backend="auto"`.
  CPU auto workers use the existing host budget; live accelerator placement
  (including structured-array block backends) defaults to one worker to limit
  concurrent device allocations. Explicit worker/backend choices win. This
  neither transfers tensors nor implements a batched GPU kernel. Auto replay
  strategy still uses branch bounds/caps, separately from backend scheduling.
- Retained the ordinary one-pass run path when backend is auto/thread, and
  initializes Stim's small NumPy matrix bridge before threaded dispatch,
  matching the TreeStab protection. The broader check reproduced a first-use
  worker stall before this initialization; the corrected selected suite passes.
  Parallel strategy preflight reuses the prepared trajectory plan.
- Validation: 54 Stim-stream tests passed; 17 selected STN layout/copy/shot
  tests passed; trajectory, importance, public API/package selection reports
  200 passed / 3 skipped; backend selection reports 26 passed / 22 skipped.
  Counts overlap and do not represent a full-suite run. Signature-policy tests
  cover CUDA/MPS/JAX GPU/TPU-like placement without claiming unavailable device
  execution. Notebook synthetic checks force both auto strategies and verify
  visible MR records, multiplicity sums, applied layout, schema and syntax.
  Ruff `src tests`, helper Ruff and diff checks passed.
- Numerical dependencies/compression unchanged; reused active-task upstream
  audit. Full archived 52-qubit benchmark remains unverified as described in
  Helix's archive note. Changes are local/uncommitted; restart the notebook
  kernel to use the updated Pepsy classes and branch-copy behavior.

## Explicit Torch CPU notebook backend

- Helix notebook now builds an explicit converter with
  `py.build_backend(device="cpu", dtype=torch.complex128, set_default=False)`
  and passes it to the simulator. Coefficient state uses complex128; translated
  matrices, including conditional feedback payloads, are converted once before
  stream preparation. Original NumPy matrices remain available for static
  layout planning. Conversion copies cached Stim matrices before creating Torch
  tensors and leaves named noise and record parameters unchanged.
- Actual Torch CPU replay passed for independent, coalesced and auto strategies
  with two workers on a noisy MR/feedback fixture. A separate layout fixture
  applied its plan and retained Torch complex128 tensors during replay. The
  archived 52-qubit first-eight-instruction prefix passed two shots at chi=16;
  the full archived circuit remains unverified. Notebook schema, cell syntax,
  saved-output preservation, helper Ruff and Helix diff checks passed.
- Changes remain local and uncommitted. Restart the kernel before rerunning
  imports so the helper's new stream conversion function is available.

## Authorized publication

- User requested committing and pushing Pepsy and the Helix notebook. Pepsy
  changes were rebased onto origin/develop `2039fbc`; only the changelog
  conflicted, and both sets of entries were retained. Device-local policy and
  Helix's unrelated `.DS_Store` modification are excluded.
- After integration, Stim-stream/public API/package and upstream trajectory
  regression/auto selections passed: 179 passed, 8 optional tests skipped.
  Ruff passed for src/tests and Helix helper. The actual Torch CPU complex128
  notebook fixture applied automatic layout and replayed noisy shots correctly.
  These checks do not validate the complete archived Helix circuit or replace
  the earlier full-suite limitations. Commits are prepared for the authorized
  push; publication completion is reported separately after remote verification.

# 2026-09-28 — Shared parsers, MPO replay, and noise/VMC organization

- Scope: user-approved consolidation of duplicated parsers, MPO replay
  simplification, optimizer option guides, and noise/VMC organization.
- Branch / baseline commit: `develop` / `4e398e4`.
- Commit status at implementation handoff: uncommitted; no new commit or push. The previous vector
  docstrings/API review remains in the working tree and is preserved.

## Changes

- Consolidated three MPS-layout parser implementations into their existing
  shared owner and reused the MPS FIT timing summary for MPO.
- Separated MPO FIT replay and direct/SVD replay from the public orchestration
  method. Removed duplicate direct/SVD success and rollback handling.
- Extracted Stim syntax translation from `noise.py` into `_stim_compile.py`;
  public compiler and record class paths remain unchanged.
- Shared NetKet's identical eager configuration evaluators. Spin/fermion
  mapping, ordering phases, and traced JAX evaluation retain their owners.
- Expanded the public packing/amplitude factory docstrings with site order,
  parameter, output-shape, scaling, and execution-mode contracts.
- Added grouped option tables and clarified state lifetime, failure recovery,
  noise-result weights/cardinality, and VMC workflow stages in existing guides.
- Detailed decisions and upstream review:
  [readability/API review](../docs/development/notes/readability_api_2026_09.md).

## Validation

- Focused selection: **450 passed, 17 warnings in 44.77s** across MPS layout,
  timing diagnostics, MPO replay/alignment, trajectory noise, VMC/NetKet,
  public API, package layout, and import-boundary tests.
- Structural comparisons: retained numerical methods unchanged; MPO public
  signature/docstring unchanged; extracted FIT body unchanged; Stim compiler
  logic unchanged apart from deferred imports of its record constructors.
- Four injected direct/SVD failures verified atomic rollback versus retained
  partial mutation, failure/timing status, and replay-policy cache cleanup.
- Stim plan identity reuse and pickle round trip passed.
- The private compiler imports without loading Stim, noise, or replay engines.
- The MPO guide example reproduces the identity operator after replay.
- Both moved eager evaluators and their spin/fermion adapter bodies match the
  original AST; noise constants/exports and the shared timing phase list match.
- Ruff, the two focused CI mypy targets, and changed Markdown file-link checks
  passed (56 local file links).
- Full collection with `MPLBACKEND=Agg python -m pytest -q -ra -o addopts=''`:
  **5,168 passed, 129 skipped, 791 warnings in 510.31s (8m30s)**; exit status 0.
  No tests were removed or added for this refactor.
- Skip breakdown: 58 missing-CuPy cases, 43 unavailable-CUDA cases, one
  unavailable-Metal case, 25 tests requiring multiple MPI ranks, and two
  requiring two configured XLA host devices. Those paths remain unvalidated
  in this single-process environment.
- Final whitespace check passed. No commit, push, or hosted CI run performed.

## Limits

No runtime dependencies or new public configuration classes were added.
This refactor does not establish a speedup or reduce numerical workload.
The environment limitations above remain; the passing run establishes the
available local suite, not every possible accelerator/distributed configuration.

## Publication follow-up

The user authorized final review, a commit of this batch, and a push to
`develop`. The review includes the earlier vector/API documentation batch.
Fetching `origin` confirmed that the local baseline and `origin/develop`
both remained at `4e398e4`. Final review removed an orphaned timing comment;
no executable code changed after the recorded full-suite run. Publication
and hosted CI results are reported separately after the Git operations.

# 2026-11-05 — Experimental independent cluster batches

An opt-in prepared-replay prototype batches compatible disjoint patches,
preserving sequential results with a modest measured speed improvement.

- Scope: user authorized trying same-network parallel cluster gate updates.
- Branch / baseline: `develop`, `859fda0`.
- Commit status: implementation, tests, API/changelog/status updates and this
  handoff remain uncommitted. The preceding untracked
  [review note](2026-11-05-loop-cluster-rereview.md) was preserved.

## Changes

`PreparedLoopClusterUpdate(..., batch_size=4)` groups consecutive disjoint,
shape-compatible complete patches. The default stays sequential. One-site
gates, source apply boundaries, overlap and incompatible shapes stop a group;
there is no lookahead or reordering. Environment programs use `torch.vmap`;
initial SVD, ALS equations, guarded solves and inter-sweep QR/LQ are batched.
Members retain independent stopping/rank/fallback decisions. Eager is required.
Preparation and final gauge extraction remain sequential.

Writeback preserves each active endpoint/gauge and commits per-pair exponent
changes and callbacks in original order. The output-gauge merge retains
unrelated updates from earlier members rather than restoring a stale snapshot.
Callbacks must remain observational. Batch contraction preflight includes the
combined estimated workspace and FLOPs; it is not a total memory bound.

The [measurement and design note](../docs/development/notes/2026-11-05-independent-cluster-batches.md)
records upstream checks, methodology, independent dependencies, limitations,
memory and exact timing samples. Classification is **prototype**; no installed
library, dependency or global registration changed.

## Validation and measurement

- New batching suite: **17 passed**, including CPU/CUDA, both complex
  precisions, Hermitian/PSD metrics, reconstruction, source/one-site barriers,
  overlap/shape fallback, rank growth, independent stopping, workspace budgets
  and finite-difference gate/gauge derivatives. A mixed Cholesky/spectral
  gradient regression keeps failed Cholesky factors outside autograd.
- Final affected selection: **537 passed, 46 warnings in 80.29 s**. This
  includes the BP compression/reduced/prepared/patch/local-gauge/Torch suites,
  new batches, public API/package layout, PEPS/PEPO and paired full updates,
  plus the native graded reduced-compression regression.
- `python -m ruff check src tests`, documentation link targets and whitespace
  checks pass. The complete repository suite was not rerun; the previously
  recorded full-suite native fermionic readout failure remains separate.
- Four synthetic disjoint six-site patches: CUDA D4 **1.365x**, D6 **1.201x**,
  D8 **1.059x** faster than sequential replay, with matching iterations,
  normalized errors and local fidelity. CPU D2 gives **1.383x**. Plans are
  warm; timing includes input copies and final writeback. Peak memory rises.
- Read-only scans of real depth20 Eli schedules find about **19%** of pair
  actions in consecutive disjoint groups before shape checks. This limits
  trajectory gains; neither synthetic batch4 ratios nor those eligibility
  counts establish whole-run speed or physical accuracy.

The first broad run had one new test failure from an unnecessary cache-miss
expectation on active-rank growth. The final regression checks bond growth and
an independent physical matrix element instead. The note also explains why
the gradient check includes the RMS exponent. No numerical tolerances were
weakened to make either check pass.

Evidence: `/tmp/pepsy-cluster-batches-final-tests.log`,
`/tmp/pepsy-cluster-batch-bench-20261105-final.json` and the script/results
paths in the measurement note. No production run, frozen source snapshot,
saved state, sibling implementation or downstream runner was changed.

## Remaining scope

This is an opt-in library prototype. The example runner does not yet expose
the batch size. Full-system tree reduction, nonzero relative PSD floors and
native U1 sector alignment retain the defects from the preceding review;
they are outside this change. Wider dependency scheduling, batching final
gauge extraction, AOT batch capture and whole-trajectory benchmarking remain
separate work.

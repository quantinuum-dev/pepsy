# 2026-10-05 — Automatic MPS trajectory execution

## Implemented

Local `MpsOptimizer.run(strategy="auto", workers="auto")` makes scheduling
choices from circuit and initial-state metadata, without timing probes, state
sampling, proposal callbacks, or changes to numerical settings. The public
rules and diagnostics are in the [MPS API guide](../../api/optimizers/mps.md).

- Accelerator states use one automatic worker, including CuPy and Torch CUDA.
  MPI also uses one automatic local worker for accelerator states; CPU MPI
  rank budgeting and independent automatic representation remain unchanged.
- Coalesced and small CPU ensembles use one worker. Independent NumPy/Torch
  CPU jobs with at least four shots, eight entries and initial bond dimension
  64 use at most four workers, respecting CPU affinity and existing numerical
  thread counts. Unknown budgets and other CPU backends stay serial. Existing
  runtime thread settings are never modified by the scheduler.
- Stream-local coalescing uses structural history bounds. For fixed mixtures
  without importance sampling, it can additionally use
  `1 + shots * sum(non_dominant_probability_per_event)` as an upper bound on
  expected occupied histories, requiring at least half the branch cap as
  headroom. No rare outcomes are discarded. Proposal policies, Kraus channels
  and branching controls disable this probability heuristic.
- Total/per-event caps remain authoritative. Automatic overflow restarts
  independently with the original seed. The chosen worker budget, planned
  strategy, reason and fallback reason are optional result diagnostic fields.
- Per-event planning caps are now honored with an unlimited total cap.
  Conditional tuple-form reset/measure-reset support parsing also now passes
  the complete action to the existing parser; it previously dropped the name.

Classification: **adopt** existing compiled-stream, backend metadata and replay
facilities. No new array dispatch or decomposition shim was introduced. The
unchanged-environment upstream audit from the
[CUDA review](2026-10-05-mps-trajectory-cuda-review.md) and
[Kraus batching work](2026-10-05-mps-kraus-autoray-batching.md) is reused.

## Evidence and limits

The earlier [CPU review](2026-10-05-mps-trajectory-performance-review.md) and
[CUDA review](2026-10-05-mps-trajectory-cuda-review.md) motivate avoiding automatic
threading of small jobs and concurrent workers on one GPU. The new CPU size
thresholds and rare-mixture headroom are conservative heuristics, **not measured
universal crossover points**. No fastest-setting guarantee or new end-to-end
speedup is claimed. Product-state circuits which later grow large remain
serial under this initial-state heuristic.

Regression coverage includes real NumPy, Torch CUDA and CuPy replay with both
complex dtypes, explicit overrides, worker-invariant independent samples,
importance weights, conditional resets, and real cap overflow. Four repeated
1% bit-flip events with 16 shots and seed 1134 have estimated occupancy at most
1.64 but realize five histories, exceeding cap four. Both serial and threaded
automatic paths restart and match explicit independent replay for that seed.
GPU MPI dispatch is tested with a fake runner; multi-rank GPU performance is
unverified. Automatic CPU threading also has a bond-64 state replay reference.

Validation: 563 passed, one skipped (unavailable second JAX device), 30 slow
tests deselected in the combined trajectory, dynamic-control, fermion, MPI and
public API/layout selection. Eleven further conditional/layout/STN checks
passed. The final automatic-execution file passed all 43 tests, including one
additional CPU parallel replay test; this rerun overlaps the combined selection.
Ruff, relative API links and diff whitespace checks passed.

No full-suite claim: earlier GPU infidelity-ledger tolerance failures and
unrelated full-suite failures remain recorded in prior handoffs. This change
does not alter those algorithms or tolerances. Shared-prefix continuation after
overflow, frontier batching, and automatic memory budgeting remain proposed
future work. State dtype, cutoff, compression mode, backend and retention
choices remain the caller's numerical contract.

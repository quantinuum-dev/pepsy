# 2026-10-06 — PEPS phase and sweep timing

- Scope: user requested detailed timing of a live 5x6 D=4 PEPS sweep test,
  including the inner variational sweep, then authorized restarting that test
  with profiling while preserving the existing results.
- Branch / baseline: develop / 7cb1044. Changes are uncommitted.

## Implementation

`PepsOptimizer.run(timing=True, timing_sync_device=False)` now records outer
target, compression, normalization, fidelity, sweep/global phase totals and
per-batch deltas. `get_timing()` returns detached last-run totals, including
failed runs. Timers retain no tensor networks after the call. Timing off adds
no synchronization or extra numerical work. Existing SweepOptimizer boundary
and local-solve timers are retained as scalar per-slice data and totals in the
sweep summary. These are unsynchronized host wall times even when outer phase
barriers are enabled, and exclude setup/diagnostic overhead.
An optional detached-record `step_callback` streams each completed batch.
The downstream runner uses it with timing enabled to write per-interval JSONL
files, retaining progress before the complete interval JSON can be saved.

The downstream roughening runner forwards its existing timing flags and saves
the per-interval totals in `optimizer_step_*.json`; batch and slice diagnostics
travel with the existing records. No solver budgets or numerical defaults were
changed. Existing dependency audit was reused; current FIT synchronization and
SweepOptimizer timing hooks were inspected in installed/local source.

## Validation

- `tests/test_peps_timing.py tests/test_optimize_peps.py
  tests/test_peps_sweep_safeguards.py`: 162 passed, 38.26 s. Includes actual
  refinement/state parity, disabled barriers, reset behavior, and failure
  cleanup. Existing warning paths and sandbox CUDA warning remain visible.
- Downstream PEPS CLI timing serialization and direct API parity: 2 passed,
  16.26 s.
- After adding streaming: all 4 timing tests passed (including partial results
  before failure), and the 2 downstream checks passed again (15.66 s).
- A tiny Torch complex128 CUDA test with timing and synchronization enabled
  completed actual D=1 variational refinement on 2x2. Phase totals, inner slice
  records, and JSONL/full JSON equality were verified. This ran alongside the
  original GPU0 job, so its timings are functional evidence, not a benchmark.
  Output: `/tmp/pepsy_phase_timing_cuda_smoke_20261006`.
- compileall and `git diff --check` passed. Ruff unavailable in cloudspace.
  Full suite not run.

## Live test

The pre-instrumentation test uses Torch complex128 on GPU0, 5x6, D=4,
dt=0.1, depth=60, delta_theta=18*pi/88. Runtime root:
`/tmp/pepsy_examples_runs/peps_sweep_5x6_D4_gpu0_torch_c128_dt01_depth60_dtheta18pi88_20261006_193459`.
At 19:44 UTC it had completed t=0.4 and was working on t=0.5. Through t=0.4,
all 25 batches per interval were below the local refinement threshold;
cumulative local-infidelity proxy was 1.725838e-9. This is not exact global
fidelity. The monitor reconstructs the product across reset per-interval
traces; the CSV is a point-in-time snapshot, not a running background writer.

At t=0.5, 15 of 25 batches attempted refinement and all 15 were accepted.
The first refined batch improved local infidelity from 1.362477e-9 to
1.184512e-9, retaining D=4 against an exact target with maximum bond 8.
Cumulative local-fidelity product was 0.999999961946102. This interval took
about 637.77 seconds by checkpoint timestamps; phase timing is unavailable
for the original process. Some NLopt runtime warnings returned best-found
parameters; all 15 accepted refinements were nonworsening.

The user approved restart with detailed timings. The original parent/child
633661/633855 were stopped, with a stop record saved beside their results.
At 19:52 UTC, a fresh run was launched with the same settings plus
`--timing-sync-device`, parent 694652 and GPU0 worker 694816:
`/tmp/pepsy_examples_runs/peps_sweep_5x6_D4_gpu0_torch_c128_profile_dt01_depth60_dtheta18pi88_20261006_195210`.
Dry-run validation passed; the worker was verified on physical GPU0 and its
first per-batch JSONL timing records were observed. Other GPU jobs remained
active. The new run starts at t=0; it does not resume the original state.

Subsequently this profiled run failed at 20:20 UTC on a small negative metric
while advancing toward t=0.6. The user requested tolerant continuation and
Autoray clipping. Follow-up implementation, tests, and replacement run are
recorded in [the fidelity-tolerance journal](2026-10-06-peps-negative-fidelity-tolerance.md).

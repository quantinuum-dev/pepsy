# 2026-10-06 — Small negative PEPS fidelity estimates

- Scope: user asked to continue through small negative approximate infidelity
  and use Autoray clipping for fidelity, including sweep diagnostics.
- Branch/baseline: develop / 7cb1044. These changes and the earlier phase
  profiling changes remain uncommitted. No commit or push performed.

## Failure evidence

The profiled 5x6 D=4 Torch complex128 run failed at 20:20 UTC, after completing
t=0.5 and 24/25 batches toward t=0.6. Its final metric was -6.86e-10 at caps
(16,20). A supplied unit target norm disabled retries and the old double
roundoff allowance was 1e-12. This was not a GPU OOM. The completed t=0.5
interval took 628.07 seconds (545.89 in sweeps), with 15/15 accepted refinements
and cumulative local-fidelity product 0.999999961946133. This is a proxy, not
exact global fidelity. Outputs contain diagnostics, not a restartable PEPS.

## Implementation

- PepsOptimizer adds configurable `evaluation_negative_tol=1e-8`. Approximate
  negative values within this allowance warn and become zero without retries.
  The dtype roundoff scale is a lower bound. Zero restores roundoff-only
  behavior. Exact metrics, nonfinite values, and larger errors keep their
  existing guards. Raw attempts and clipping metadata are saved per batch.
- SweepOptimizer receives the same tolerance (overridable in sweep_kwargs),
  with its existing 1e-10 roundoff floor. Valid diagnostic losses are bounded
  using backend-native Autoray real/clip. Raw local loss fields and histories
  are retained, including clipped local records in the PepsOptimizer summary.
  The differentiable objective is deliberately unclipped, so its gradients
  are unchanged. Larger invalid/nonfinite candidates retain rollback guards.
- Updated API documentation and changelog. No contraction caps, solver budgets,
  gate ordering, target construction, or normalization algorithms changed.

## Validation

- Final domain selection: 221 passed in 39.10 s across test_optimize_peps,
  test_peps_sweep_safeguards, test_peps_optimizer_batching, test_peps_timing.
  Includes continuation after the observed discrepancy, raw diagnostics,
  strict overrides, NumPy/Torch clipping and local accepted-state behavior.
- Downstream CLI/default parity selection: 2 passed in 15.35 s.
- Actual CUDA0 Torch and CuPy clipping preserved dtype/device, and the raw
  Torch complex128 objective retained a nonzero gradient above fidelity 1.
- An intermediate run exposed 14 mock-backend failures caused by coupling
  the outer clamp to a SweepOptimizer private helper. Removed that coupling;
  the final domain selection above passed without changing those tests.
- compileall and git diff --check passed. Ruff unavailable in cloudspace;
  full suite not run. Existing dependency audit reused, environment unchanged.

## Live run

Restarted from t=0 on physical CUDA0 with all prior settings and synchronized
phase timings after dry-run validation. Parent PID 782163. Output:
`/tmp/pepsy_examples_runs/peps_sweep_5x6_D4_gpu0_torch_c128_profile_tol1e8_dt01_depth60_dtheta18pi88_20261006_210708`.
First batch diagnostics observed. Previous output directories preserved;
other GPU jobs not stopped. Production passage beyond t=0.6 remains unverified
at this handoff. See the phase-timing journal for earlier instrumentation.

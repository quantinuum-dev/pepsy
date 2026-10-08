# 2026-10-07 — Gate-by-gate sweep performance review

- Scope: stop CUDA:0/2 PEPS runs, retain CUDA:1 gate-by-gate run, inspect
  performance while retaining accuracy checks.
- Branch / baseline: develop / 0ba2a05; pre-existing adaptive-boundary edits
  remain uncommitted. This review changes no numerical implementation.
- Commit status: uncommitted; no commit or push requested.

## Operations and validation

- Stopped CUDA:0 parent/child 935190/935590 and CUDA:2 935218/935588 with
  SIGTERM. Verified all four exited; results preserved with stop markers.
- Verified CUDA:1 parent/child 935204/935589 and unrelated CUDA:3 MPS 89592
  remained running. No restart or settings change on CUDA:1.
- Read production JSON records and current optimizer implementation; no
  numerical tests were run because no numerical behavior changed.

## Measured findings

Source: `/tmp/pepsy_examples_runs/peps_D4_sweep_dmrg_normtol_gpu1_20261007_221424/theta_01/checkpoints/optimizer_step_000003.json`.

- The t=0.2 to 0.3 step took 1854.535 seconds for 49 gate batches:
  boundary calibration 1021.411s, sweep 760.035s, normalization 39.433s,
  warm-start compression 33.083s, target construction 0.412s.
- All 49 calibrations passed. Each selected chi=(48,48), after caps
  16,32,48 and independent fresh confirmation at 48 in both axes.
  Each cap/direction evaluates two norms and one overlap: 24 scalar
  contractions per checked gate, with warm boundaries reused between caps.
- All 49 final-confirmation warm-start loss estimates were below the
  configured 1e-9 optimizer tolerance (maximum across both axes ranged
  from 5.1e-15 to 3.79e-12). Nevertheless 1274 local updates ran (26/gate).
- `measure_infidelity=False` skips the calibration-derived precheck block
  in PepsOptimizer.run, so the below-tolerance shortcut cannot fire.
  This is current objective-only behavior, not a newly applied fix.
- At inspection, CUDA:1 had completed t=0.3 plus 24/49 batches of the next
  step; 73/73 saved calibrations had passed. Earlier batches within D need
  no variational calibration, hence batch count differs from check count.

## Potential improvements, not implemented

1. Reuse calibration for an explicitly reliable early-exit decision without
   adding separate overlap contractions. Require physical validity,
   convergence and observed cross-cap/cross-axis/fresh variation appropriate
   to the decision tolerance. Passing rtol=1e-5 alone cannot certify a
   1e-9 loss, and empirical agreement is not a rigorous error bound.
   Removing all sweep cost would save at most about 41% on this recorded
   step before extra safeguards; this is not a measured new speedup.
2. Transfer same-state calibrated boundary environments into the fitter,
   with correct normalization scaling, indices, ownership and cap policy.
   Existing code already reuses scalar norms, but discards these handles.
3. Reuse unaffected partial environments across single local gates, with
   explicit invalidation for changed tensors, normalization and every
   fitted update; retain fresh confirmation. Never reuse stale scalars.
4. Explore fitting the affected row/column first with a full-sweep fallback.
   This changes the optimization schedule and requires reference tests.

These findings concern the first angle at early times; near-unit local
fidelity and runtime savings must not be extrapolated to the whole sweep.

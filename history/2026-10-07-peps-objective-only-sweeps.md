# 2026-10-07 — Objective-only PEPS sweeps

- Scope: launch the requested 5×6 D=4 Torch complex128 sweep on cuda:0,
  with automatic batches, norm/overlap caps 32/56, two round trips per axis,
  50 SciPy iterations per local fit, no independent overlap diagnostics,
  and normalized outputs. User explicitly confirmed these interpretations.
- Branch / baseline: develop at `01c0d5e`; changes remain uncommitted.
- Added `compute_initial_loss=False` to SweepOptimizer's run controls. With
  `compute_final_loss=False`, whole-state initial/final diagnostics are skipped.
  Local objectives and safeguards remain active. An explicitly supplied
  `initial_loss` retains precedence. Defaults are unchanged.
- Companion examples changes map SciPy's CLI maxeval limit to `n_steps`
  (ultimately SciPy maxiter), and expose the opt-out overlap-check flag.
  The old maxeval-only forwarding was ignored by SciPy, leaving maxiter=30.
  Regression reproduced that failure before the fix.
- Validation: 265 PEPS performance/batching/safeguard/timing/optimizer tests
  passed, five warnings, 54.48 s. Logs: `/tmp/pepsy_sweep_no_checks_tests.log`.
  Public API/layout checks: 54 passed, one known environment failure
  (`test_package_version_matches_installed_distribution`: installed metadata
  0.4.0 vs project 0.5.0). This predates these edits; environment unchanged.
  Examples focused launch/forwarding checks: 10 passed, including real maxiter
  values 4 and 50 and normalized dense output with independent checks forbidden.
  Full examples benchmark: 627 passed, 5 skipped, 4 known bubble entrypoint
  failures (306.76 s); companion note records failure names and baseline evidence.
  Ruff is not installed; `git diff --check` passes.
- Upstream audit reused from the same active maintenance task, unchanged
  environment; no backend registration or contraction algorithm changed.

The new sweep launched at 01:05 UTC with parent PID 1410678 and first child
1410904. The child's checkpoint and log confirm Torch complex128 on cuda:0,
caps 32/56, normalization cap 32, maxeval request 50, overlap checks false,
auto batching, two round trips, no samples, and no inherited thread limits.
The first angle reached t=0.2 before its first refinement was inspected.
Exact commands, source patches, and output live under the root linked in the
companion examples note `2026-10-07-peps-scipy-launch-limit.md`.

## Earlier global diagnostic remains unresolved

The bounded first-fit probe saved its input states under
`/tmp/pepsy_global_first_fit_20261007/first_fit.pkl` and diagnostics in that
directory; log `/tmp/pepsy_global_first_fit_probe.log`. This was current code,
5×6 D=4 Torch complex128, auto batches, direct boundary caps 64/64, delta theta
18*pi/88. The first refinement was at t=0.3 with target D=8. Global NLopt
LD_VAR2 still stopped with runtime_error (12 evaluations despite budget 8),
restored its best vector, and the outer comparison rejected the candidate.
The last trial loss was about 0.999999 from an initial 2.14e-11. Rank-deficient
QR warnings accompanied the probe. This is not a global-mode fix.
The probe finished and cuda:0 was free before preparing the requested sweep.
Further gradient investigation must instrument TNOptimizer.handler.value_and_grad:
NLopt's closure bypasses vectorized_value_and_grad. The latter wrapper produced
an empty evals.json in this diagnostic. Do not infer finite gradients from it.

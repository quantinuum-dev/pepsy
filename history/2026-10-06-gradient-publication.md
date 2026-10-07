# 2026-10-06 — Remaining solver corrections and publication

- Scope: review remaining Pepsy/Gaugy optimizer issues, fix confirmed bugs,
  commit and push the task changes as explicitly requested by the user.
- Starting Pepsy baseline: `develop`, `2abb1e8`, with the earlier uncommitted
  [eight fixes](2026-10-06-gradient-fixes.md). Remote fetch found `6e9cef6`.
- Publication status and post-merge checks are recorded below as completed.

## Additional confirmed fixes

- Preserve autograd during terminal Torch evaluation: derivative-penalty
  objectives otherwise return NaN or discard a successful last update.
- Reject nonfinite native Torch/JAX gradients; skip JAX parameter/state updates
  when the measured loss is nonfinite.
- NLopt and FD-NLopt propagate scalar/real loss contract errors, report
  all-invalid evaluations explicitly, retain Pepsy forced-stop labels, expose
  actual successful termination reasons, and evaluate the returned parameters
  when the best-loss cache does not describe them. These extra evaluations
  are included in `n_evals`.
- Reject NLopt `max_step`. Its previous callback clipping returned a different
  point's value/gradient from those requested by NLopt, violating the
  [upstream objective contract](https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#objective-function).
  Bounds remain supported. No installed libraries were changed.
- Gaugy's lower-level window sweeps now inherit the signed/absolute reduction's
  best-loss policy, matching its compiled adapter; explicit overrides win.
  See the Gaugy `2026-10-06-gradient-solver-followup.md` handoff.

## Validation before merging remote changes

- Added solver regressions initially reproduced **16 failures**; the two
  later termination-reason regressions also pass after correction.
- Final Pepsy solver, PEPS/qMERA domain and API/layout selection:
  **394 passed**, two existing deprecation warnings, 92.21 seconds on CPU.
  Same eight test modules as the prior 376-test validation, with 18 new cases.
- Ruff (`src tests`), whitespace, and local document-link checks pass.
- Gaugy's new signed-sweep regressions initially produced **2 failed, 4 passed**;
  sweep API and scoped/reflection checks after correction: **90 passed**.
- The earlier full-suite attempt's 24 BP failures were reproduced at unchanged
  HEAD; see the prior handoff. They are not changed by this optimizer task.

## Scope preserved

Only task-owned implementation, tests, API documentation and review journals
are included. Existing AGENTS, MPS fusion edits and their changelog entry remain
local. No notebook outputs or simulation datasets are staged or removed.
The installed dependency audit from the same task is reused; the NLopt Python
reference and installed status constants were additionally checked.

## Publication / post-merge validation

Pending remote integration and final checks.

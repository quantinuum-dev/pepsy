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

- Optimizer fixes committed as `f459f5a`.
- Integrated remote `6e9cef6` (including `01c0d5e`) with both changelog sections
  and both sweep imports preserved. The unrelated MPS changelog entry was
  backed up locally, temporarily set aside for the merge, and restored
  unstaged. Other unrelated tracked edits were verified byte-for-byte.
- Post-merge solver/domain/API plus PEPS batching and timing selection:
  **481 passed**, 51 warnings, 109.46 seconds. No new test failures.
- Post-merge full-suite attempt with `-x`: **82 passed, 2 skipped, 1 failed**
  at `test_bp_compression.py::test_sequential_loop_series_compression_reuses_projected_messages`,
  the same BP convergence failure previously reproduced on the baseline.
- Gaugy follow-up: **90 sweep/reflection checks + 70 compiled solver/objective
  checks passed**; changed-file Pyflakes and whitespace checks pass.
- Final Ruff, whitespace, staged-file scope, and unrelated-edit preservation
  checks pass.
- Published to `origin/develop`: Pepsy code/integration merge **`9eb7ff1`**,
  containing optimizer fix commit **`f459f5a`** and remote `6e9cef6`.
  The push completed successfully. Gaugy's corresponding code fix
  **`7b03053`** was also pushed successfully to its `origin/develop`.
  This journal update records those verified code publications; no release
  tag or unrelated working-tree changes were published.

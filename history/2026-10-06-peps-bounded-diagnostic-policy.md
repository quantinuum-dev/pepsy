# 2026-10-06 — Continue through bounded approximate fidelity excursions

User requested clipping fidelity to [0,1] in global/sweep mode and avoiding
hard stops unless the estimate is far outside the physical interval.
Baseline: develop 4f8935e (already pushed). This change remains uncommitted.
Maintainer/fitting guidance and the same-session upstream audit were reused;
no contraction algorithm, dependency, or dispatch changes are introduced.

## Policy

- Default `evaluation_negative_tol` increases from 1e-8 to 1e-3 for both PEPS
  and sweep optimizers. Its historical name now covers both ends of [0,1].
  The chosen allowance (0.1 percentage point) was disclosed to the user.
- Approximate diagnostic estimates within the allowance are clipped with
  Autoray and warned; raw values are retained. Finite-cap estimates beyond
  that allowance keep retry/error guards. Exact contractions retain their
  roundoff-only allowance. Setting the option to zero requests strict mode.
- The global/sweep return path previously called the roundoff-only cleaner,
  which could still abort the driver after an otherwise recoverable inner
  result. It now uses the same approximate tolerance and saves raw/bounded
  values in optimizer summaries before outer acceptance proceeds.
- Sweep initial/local guards reject nonfinite or grossly out-of-range values
  and retain the current state. Both upper and lower bounds are checked.
- Differentiable objectives, solver histories, convergence budgets, target
  construction, caps, and output normalization are unchanged. This does not
  repair NLopt convergence or certify the clipped fidelity as an exact overlap.

## Validation

Regressions cover the previously observed -3.267e-6 error, -5e-4 excursions,
both ends of the interval, two-batch continuation through global/sweep return
paths, preserved outer acceptance, raw objectives on NumPy/Torch, gross-error
and nonfinite rollback, strict/exact overrides, and an actual coarse-boundary
identity circuit compared against its normalized dense overlap.

The first selection had 247 passes and two failures in tests expecting the
previous strict default for small coarse-boundary errors. Those tests now
explicitly select strict mode; their accuracy/retry assertions are preserved.
Separate default-policy tests exercise continuation on the same real inputs.
Final selection: 251 passed, 5 warnings (46.36 s), across batching, sweep
safeguards, PEPS optimization, timing, and performance suites. Compileall and
git diff --check passed. Ruff is unavailable in cloudspace. Full repository
suite not run.

## Running job

The global CUDA0 process launched at 22:08 uses pre-change code and an explicit
1e-6 allowance. It was still running, with t=.8 completed, during this edit.
Asked whether to restart from t=0 to apply the policy, preserving old output.
No restart has been performed without an answer; running processes cannot
load the changed policy automatically.

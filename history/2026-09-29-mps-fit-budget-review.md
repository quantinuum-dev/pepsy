# 2026-09-29 — Review of FIT budgets and roughening integration

- Scope: user requested another implementation review and report.
- Baseline: Pepsy `develop` / `a233a40`; examples `main` / `efb7696`.
- Status: review only; no implementation changes, commit, or push. Temporary
  probes and generated outputs stayed under `/tmp`; production data/jobs and
  unrelated working-tree edits were preserved.

## Findings

1. **P2 — Resume cannot distinguish the earlier meaning of an explicit None
   pair cap.** Shared configuration records retain
   `fit_single_pair_n_iter=None` without an algorithm-policy version.
   `roughening_sweep._is_complete` compares this configuration and archive
   completeness, but not the policy or recorded sweep counts. In the earlier
   intermediate implementation, an explicit None selected the legacy DMRG2
   adjacent-pair one-update shortcut; now it inherits `n_iter`.

   Reproduced using a temporary real 2x2 roughening run, with only the option
   resolver monkeypatched to emulate that earlier None policy. With DMRG2,
   `n_iter=8`, `fit_rtol=None`, and pair cap None, its saved counts were
   `[1, 8, 1, 1]`. After restoring the current implementation, the current
   configuration exactly matched and `_is_complete` returned True. Current
   fixed-budget replay instead requests eight sweeps for those pair windows.
   This does not affect roughening's explicit default pair cap 5. No claim is
   made that an existing production output used the earlier None policy.

   Suggested correction: include a solver-policy version in resume identities
   or explicitly reject the affected older None-policy records. A numerical
   configuration match alone cannot establish compatibility across this
   semantic change. Reproducer: `/tmp/review_resume_none_20260929.py`; log:
   `/tmp/roughening-review-resume-none-20260929.log`.

2. **P3 — Internal skill guidance still describes automatic DMRG2 one-update
   behavior.** `.github/skills/mps-optimizer/SKILL.md:82` and
   `.github/skills/tensor-fitting/references/fit-architecture.md:107` disagree
   with the current public implementation. Public API docs are current.
   Synchronize those references through the skill-maintenance workflow.

## Verified behavior

Pepsy's omitted/None pair cap inherits each replay's `n_iter`; explicit
positive caps use `min(n_iter, pair_cap)`. Named DMRG modes, ordinary gates,
batched windows, measurement windows, shot overrides, early stopping, and
explicit one-update fast paths remain consistent in the tested cases.
Roughening explicitly retains pair cap 5 and general cap 8. Per-window local
caps do not leak into later longer windows or subsequent calls.

## New validation

Shared Python 3.12 environment, local Pepsy source, one numerical CPU thread,
CUDA hidden. No dependency changes; the same-task upstream audit was reused.

- Focused package suites (`test_mps_fit_window_budget`, `test_mps_fit_kernels`,
  `test_mps_controls`, `test_mps_dynamic_controls`, `test_mps_audit_fixes`):
  **283 passed** in 22.03 s.
- Downstream suites (`test_roughening`, `test_roughening_sweep`,
  `test_entrypoints`, `test_provenance`, `test_runner_layout`, `test_run`):
  **298 passed, 2 CUDA skips** in 43.16 s.
- **36 additional cases** checked mixed adjacent/long-range gate streams,
  caps None/1/2/5, general limits 3/8, all four DMRG modes, repeated calls,
  and dense exact-state agreement. Temporary probe:
  `/tmp/review_fit_budget_20260929.py`.
- Pepsy `ruff check src tests` and `git diff --check` pass.

The full domain/GPU suites were not rerun in this review. The prior
[inheritance handoff](2026-09-29-mps-fit-budget-inheritance.md) records the
existing installed-version mismatch and unrelated examples lint findings;
these were not repaired as part of a review request.

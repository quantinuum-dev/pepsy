# 2026-09-29 — Adjacent FIT budget inherits n_iter by default

- Scope: user clarified that Pepsy's `fit_single_pair_n_iter` must follow
  `n_iter` unless explicitly overridden. Roughening retains its explicit 5/8
  policy.
- Branch / baseline: `develop` / `a233a40`.
- Commit status: working-tree edits only; no commit or push. Unrelated cluster
  work and examples notebook changes were preserved.

## Change

`MpsOptimizer.run` now defaults `fit_single_pair_n_iter` to `None`. The replay
option resolver substitutes that run's `n_iter`, including after shot-level
`run_kwargs` overrides. This also prevents named DMRG2 from silently selecting
its legacy one-update shortcut. Explicit positive pair caps remain limited by
`n_iter`; explicit `fit_single_pair_fast_path=True` still selects one update.
Convergence-based early stopping is unchanged.

Updated public API docs, changelog, numerical regressions, and examples CLI
help/README. Roughening's explicit `n_iter=8` and pair cap 5 are unchanged.
Passing `--fit-single-pair-n-iter none` now inherits `--n-iter`.

This supersedes the default-5/legacy-None policy in the earlier
[adjacent-budget session](2026-09-29-mps-adjacent-fit-budget.md), whose test
results remain historical evidence for that earlier implementation.
Its upstream audit is reused; this follow-up changes option resolution only,
with no environment, tensor-kernel, or dependency changes.

## Validation

Used `~/envs/py312`, local Pepsy first on `PYTHONPATH`, one numerical CPU
thread, and CUDA hidden. No production jobs or datasets were modified.

- `python -m pytest -q -ra -o addopts='' tests/test_mps_fit_window_budget.py
  tests/test_mps_fit_kernels.py tests/test_mps_controls.py
  tests/test_mps_dynamic_controls.py tests/test_mps_audit_fixes.py
  tests/test_public_api.py tests/test_package_layout.py`: **335 passed,
  1 existing failure** in 17.65 s. The failure remains
  `test_package_version_matches_installed_distribution`: installed metadata
  0.4.0 versus source version 0.5.0. No shared-environment reinstall.
  Log: `/tmp/pepsy-pair-inherit-20260929.log`.
- Downstream roughening, sweep, and entrypoint suites: **214 passed,
  1 CUDA skip** in 13.95 s. Log:
  `/tmp/roughening-pair-inherit-20260929.log`.
- Pepsy `ruff check src tests` and changed downstream files pass. Broad
  downstream Ruff still reports the six existing effective-model findings.
- Both repositories pass `git diff --check`. The earlier expanded 1288-test
  package suite and full examples suite were not rerun for this correction.

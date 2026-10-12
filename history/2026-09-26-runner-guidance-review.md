# 2026-09-26 — Review runner organization and clarify agent guidance

- Scope: user requested a second review, clearer agent Markdown files, and
  a summary of completed work and running sweeps.
- Branch / baseline: Pepsy `develop` / `80f451a`; examples `main` / `301159a`.
- Commit status: all task changes remain unstaged and uncommitted; nothing
  published. Existing unrelated changes were preserved.
- Prior implementation and validation:
  [runner reorganization](2026-09-26-magnetization-runner-organization.md).

## Changes and findings

- Updated examples root `AGENTS.md`, the magnetization experiment guide,
  and its benchmark guide. Guidance now names canonical module owners,
  single-job versus sweep commands, MPS/tree/exact selection, compatibility
  entrypoints, extension responsibilities, background-job checks, and
  focused versus full validation commands.
- Corrected stale references to `benchmark_engine.py` and `helper` as
  implementation owners; entropy now points to `engines/mps.py`. Corrected
  the documented bubble defaults after checking the parser: J=-1, hx=1.5,
  hz=0.2. No physical model or runtime defaults changed.
- Combined-suite review found two failures in the new subprocess tests:
  child runs exited successfully, but the parent decoded UTF-8 output using
  its ASCII locale. Set explicit UTF-8 output/decoding in the test harness;
  production code was unchanged.

## Validation

- Initial combined run: 379 passed, 2 failed with UnicodeDecodeError;
  `/tmp/roughening-layout-review-tests.log`. Earlier separate test runs did
  not expose this locale interaction.
- Full rerun after the encoding correction: 381 passed, 65 warnings;
  `/tmp/roughening-layout-review-tests-fixed.log`. This is a new combined
  run of the entire benchmark test directory, including launcher regressions.
- Ruff passed for implementation, tests, compatibility launchers, and Pepsy
  `src tests`. Both repositories' `git diff --check` passed.
- Updated guide links resolve; documented modules import, tree/sweep flags
  parse, and stated bubble/roughening defaults match current parsers.
- No Pepsy numerical suite rerun: this review changed examples guidance and
  the test harness only. No jobs stopped, restarted, or newly launched.

## Status at 09:24 MDT

- All three 5×5 exact sweeps (dt=0.05, 0.25, 0.4) are complete, 12 angles
  each through t=10.
- 5×6 exact: two angles complete, third at saved depth 19/25.
- 5×6 CUDA DMRG: one angle complete, second at saved depth 21/25.
- Process checks confirmed both sweep parents and their children alive;
  progress is point-in-time evidence, not a completion claim.

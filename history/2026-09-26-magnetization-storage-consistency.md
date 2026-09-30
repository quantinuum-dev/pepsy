# 2026-09-26 — Consistent optional cash/store runtime directories

- Scope: user-requested consistency review and clear cache/data separation;
  user explicitly selected `store/` only when requested.
- Branch / baseline: Pepsy `develop` / `80f451a`; examples `main` / `301159a`.
- Commit status: working-tree edits only, unstaged and uncommitted; nothing
  published. Existing data, notebook outputs, and unrelated edits preserved.

## Changes

- Created ignored `benchmark/cash/` and `benchmark/store/` runtime folders.
  Local result storage is opt-in with `--out store/<run-name>`; existing
  frontend defaults and explicit output paths were not changed or migrated.
- Added `--cotengra-cache-dir` to the shared-engine CLI for the existing
  MPO-observable optimizer. Default remains `<out>/cache`; custom paths
  support reuse across runs. Recorded the resolved path in checkpoints and
  final metadata, or null when no MPO observable optimizer is used.
- Sample-only roughening creates no such cache. MPS/FIT remains `auto-hq`;
  KZ/effective paths are unchanged and do not expose this cache option.
- Sweep children receive an absolute cache path resolved in the parent's
  launch directory. Corrected frontend `--out` help to show actual defaults.
- Synchronized root, experiment, benchmark, and package READMEs plus the
  three relevant AGENTS.md files. Fixed the experiment README's stale
  implementation-owner descriptions from before the runner move.
- Evidence and installed API audit:
  [runtime storage note](../../pepsy_examples/experiments/mps_magnetization/benchmark/docs/development/notes/runtime_storage.md).

## New checks

- Focused storage/sweep tests: 10 passed. Real on-disk cache creation/reuse
  preserved MPO energies and metadata; sample-only runs created no cache;
  path forwarding preserved the parent's working directory.
- Full benchmark directory: **384 passed, 65 warnings**, 84.31 seconds;
  `/tmp/magnetization-storage-full.log`. An initial focused test used a
  roughening-only flag with the shared parser; that test setup was corrected
  before the passing focused and full runs.
- Ruff passed for the implementation, benchmark tests, compatibility files,
  and Pepsy `src tests`. Both repositories' `git diff --check` passed.
- README/AGENTS local links, storage examples, CLI help, and Git ignore
  behavior were verified. The Git index remains empty.
- Pepsy numerical code and dependencies were not changed. No package-wide
  numerical suite was claimed; the full result above is the benchmark suite.

## Existing production runs, checked 09:34 MDT

- Exact 5×6: two angles complete; angle 3 at saved depth 21/25.
- CUDA MPS 5×6: three angles complete; angle 4 at saved depth 16/25.
- Both parent processes remain alive. No production jobs stopped, restarted,
  or newly launched by this task.

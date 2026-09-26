# 2026-09-26 — Organize magnetization runners and engine helpers

- Scope: user-requested runner/helper organization, with clearer MPS/tree
  ownership and room for additional engines. Work is in `pepsy_examples`;
  no Pepsy implementation changes were needed.
- Branch / baseline: Pepsy `develop` / `80f451a`; examples `main` / `301159a`.
- Commit status: all task edits are unstaged and uncommitted; nothing published.
  Pre-existing source, notebook, and documentation edits were retained.

## Changes

- Moved the current implementations into the benchmark's local
  `magnetization` package: `runners/`, `engines/`, `diagnostics/`,
  `effective/`, and `physics.py`. Used git moves, then removed only this
  task's staging so the index remains empty.
- Extracted existing MPS measurement helpers and tree layout/bond helpers
  into `engines/mps.py` and `engines/tree.py`. Shared evolution, engine
  construction, sampling, and checkpoints remain in `engines/shared.py`.
  This is an organization change, not a new engine plugin system.
- Original `run_*.py` scripts delegate to the new modules. Legacy Python
  imports alias the canonical implementations. The experiment's old
  `helper.py` forwards to `physics.py`; plotting helpers stay in `plots/`.
- Kept output roots and the sweep's child command path anchored to the
  benchmark directory, including for already-running sweep parents.
- Added command/import compatibility regressions and updated README maps
  and the closest benchmark instructions. No notebook files were edited.
- Detailed layout and extension guidance:
  [runner map](../../pepsy_examples/experiments/mps_magnetization/benchmark/magnetization/README.md).

## New validation

- Before edits: entrypoint, sweep, and shared-engine tests: 106 passed.
- After relocation: entire benchmark test directory: 377 passed, 65 warnings
  (including upstream deprecations); log `/tmp/roughening-layout-tests.log`.
- New regression file: 4 passed, covering import identity/output roots,
  fresh legacy MPS command, canonical tree command, and a two-angle exact
  sweep from a separate process; `/tmp/roughening-layout-launch-tests.log`.
- AST comparison against pre-move working files confirmed all 321 existing
  top-level function/class bodies are unchanged, normalizing import paths.
  Snapshot: `/tmp/magnetization-layout-before`; move script and patch are
  also under `/tmp`. These snapshots include earlier uncommitted work.
- Ruff passed for moved implementation, compatibility scripts, and new tests;
  Pepsy `python -m ruff check src tests` also passed. Both repositories'
  `git diff --check` passed. Updated README relative links resolve.
- Existing production CUDA sweep parent 4165168 completed angle 1 and
  launched angle 2 (child 4172556) through the compatibility launcher.
  CPU exact sweep parent 3923668 remained running, angle 3 at saved depth 18.
  These are point-in-time checks, not claims that the sweeps completed.

## Limits

- No numerical behavior, optimizer configuration, or dependency change was
  intended; no numerical upstream upgrade/adoption audit was needed.
- The Pepsy numerical suite was not rerun because package code was unchanged.
  Benchmark coverage and launch checks above apply to this reorganization.
- Production sweeps continue independently; no jobs stopped or restarted.

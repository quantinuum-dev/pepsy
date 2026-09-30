# 2026-09-25 — Current repository status review

- Scope: inspect recent changes and assess package status; no implementation fixes.
- Branch / baseline: `develop` / `9c164671ff49ea0f030779aa105d390fcff7a86a`.
- Commit status: this review record is uncommitted; no commit, push, merge,
  release, workflow dispatch, or environment modification was performed.

## Current evidence

- The checkout was clean at startup. Live GitHub branch queries confirmed
  `develop` matches HEAD and `main` remains at `46eb601`, four commits behind.
- The local `v0.5.0` tag points to `e5bb212`, seven commits behind HEAD.
  Current metadata still declares 0.5.0. GitHub's releases API returned no
  Release entries; current automation builds GitHub Actions artifacts only.
- Recent work extracts MPS reporting, organizes tree documentation, removes
  internal compatibility imports, and simplifies CI with separate caches and
  wheel-from-source-archive validation. These are distinct from the broader
  numerical changes already recorded in the 0.5.0 changelog.
- [Routine CI on HEAD](https://github.com/quantinuum-dev/pepsy/actions/runs/36197720452)
  passed all three jobs: core, package, docs. Core reports **1,381 passed,
  319 skipped, 2,188 deselected**. Focused mypy passed on the hosted runner.
- [Full workflow on HEAD](https://github.com/quantinuum-dev/pepsy/actions/runs/36198302737)
  passed: **4,526 passed, 177 skipped**, **72.86% coverage**, above the 60%
  gate. MPI reports **57 unit tests passed**, plus **25 integration tests
  passed per rank** under both two- and three-rank execution; benchmark
  steps also succeeded. Skipped paths remain unvalidated.
- These live results resolve the pending hosted/full-workflow status in
  [the preceding CI review](2026-09-25-light-ci-review.md). The updated
  scheduled configuration still requires promotion to the default `main`
  branch; the successful develop run does not change default-branch scheduling.

## Fresh local validation

- Designated `genpy` environment, Python 3.12.14, source import confirmed.
- `MPLBACKEND=Agg python -m pytest -q -o addopts='' -m 'smoke or (core and not optional and not slow)'`:
  **1,682 passed, 8 skipped, 3,038 deselected**, 99 warnings, 75.81 seconds.
- Ruff, the 12-skill catalog, and the initial whitespace check passed.
- Local mypy could not run: the module is not installed. No dependencies
  were installed or changed to repair the shared environment.
- Installed NetKet is 3.21.0, below the declared optional floor of 3.22.
  Quimb, Cotengra, Autoray, and Symmray are development builds, so this local
  result is not a minimum-release or complete VMC compatibility check.
- `pip check` reports an unrelated shared-environment mismatch: notebook
  7.5.0 requires jupyterlab <4.6, but installed jupyterlab is 4.6.4.

## Remaining maintenance observations

- The historical project roadmap still labels the package version 0.4.1;
  use current metadata, changelog, implementation, and tests for status.
- Several implementation modules remain large (symmetric tensors and the
  MPS optimizer each exceed 11,000 lines). Further extraction is a deferred
  maintenance opportunity, not an established correctness defect.
- No new failing package check was found in this review. The full local
  suite was not rerun; current hosted full-suite evidence is recorded above.

# 2026-09-25 — Dependency minimum CI follow-up

- Scope: finish checking the latest dependency-minimum changes and repair
  failures found in hosted CI.
- Branch / baseline: `develop` / `af28262`.
- Commit status: repair committed and pushed to `develop` as `d55e10c` with
  user approval. This final hosted-result update accompanies a documentation
  follow-up commit.

## Hosted result and diagnosis

- [Run 36091606731](https://github.com/quantinuum-dev/pepsy/actions/runs/36091606731)
  completed with failures in both core and extended test jobs. Docs, package,
  type checks, agent guidance, and both MPI jobs passed.
- Both failed jobs annotated the same test:
  `test_composed_dependency_profiles_preserve_feature_boundaries` in
  `tests/test_package_layout.py`. Its expected extra still required
  `autograd>=1.6`; the audited metadata now requires `autograd>=1.7`.
- This resolves the pending hosted status in the
  [dependency-minimum handoff](2026-09-24-dependency-minimums.md).
  That handoff's recorded 58-test pass does not establish a pass for the final
  committed tree: the stale assertion also failed locally at `af28262` today.
- The extended job stops at its first failure, so later tests are unverified
  for this hosted run.

## Repair and validation

- Updated only the stale expected Autograd floor to `>=1.7`, preserving the
  exact dependency-set and feature-boundary assertions. Runtime code and
  package requirements are unchanged.
- Before the fix, the exact failing test reproduced locally: **1 failed**.
- After the fix, `python -m pytest -q -o addopts='' tests/test_public_api.py
  tests/test_package_layout.py`: **58 passed**, with eight existing
  compatibility-alias deprecation warnings.
- `python -m ruff check src tests` and `git diff --check` passed.
- Checks used the required local Python 3.12 environment. No dependency
  installation or full numerical rerun was needed for this test-only change.

## Hosted verification after repair

- [Run 36149515229](https://github.com/quantinuum-dev/pepsy/actions/runs/36149515229)
  tested `d55e10c6c31693e18efc5c35b9559eafb73cd54f` and completed successfully.
- All eight jobs passed: minimum-version core tests, extended tests with the
  60% coverage gate, docs, package, type checks, agent guidance, and both MPI
  configurations.
- The pending hosted verification is complete. This final handoff update is
  documentation only; it does not alter the tested implementation or tests.

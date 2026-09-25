# 2026-09-25 — Dependency minimum CI follow-up

- Scope: finish checking the latest dependency-minimum changes and repair
  failures found in hosted CI.
- Branch / baseline: `develop` / `af28262`.
- Commit status: test correction and this handoff accompany the repair commit;
  the user approved committing and pushing it to `develop`.

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

## Remaining work

- Verify the new hosted core and extended results after publication. The
  corrected tree has not yet been tested by GitHub Actions.

# 2026-09-25 — Follow-up CI review

- Scope: review and refine the lightweight CI configuration again.
- Branch / baseline: `develop` / `925cbe0`.
- Commit status: prepared for the authorized commit and push to `develop`.

## Findings fixed

- All five test/docs/package jobs used setup-python's same pip-cache key:
  identical platform, Python version, and `pyproject.toml` hash. Different
  dependency profiles could compete for one immutable cache. Switched to
  explicit per-job keys, including platform, Python minor version, package
  metadata, and the owning workflow. No cross-profile fallback restores.
- Routine packaging built its wheel directly from the checkout. It now uses
  `python -m build` to build the wheel from the source archive, matching the
  release workflow and checking that archive's completeness.
- Added the missing ten-minute timeout to the release-artifact job.
- Updated the contributor guide. Routine job counts, test selections,
  nightly/manual coverage, branch policy, and GitHub-only artifacts remain.

## Validation

- The preceding lightweight run completed successfully: all three jobs in
  [run 36197086357](https://github.com/quantinuum-dev/pepsy/actions/runs/36197086357),
  with 3 minutes 36 seconds elapsed. This resolves the pending hosted result
  in the [initial handoff](2026-09-25-light-ci.md).
- Actionlint passed all three workflows; YAML and shell syntax checks passed.
- Verified five distinct cache keys and timeouts on every job.
- Built the source archive and its wheel locally using the designated
  environment without build isolation; both passed Twine validation.
- Hosted execution of this follow-up is pending push. No numerical code or
  test selection changed, so numerical tests were not repeated.

The full nightly/MPI workflow has not yet been run after the move. Its new
schedule still needs promotion to the default branch; manual execution on
`develop` is available. The cache correction's speed benefit is unmeasured.

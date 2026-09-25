# 2026-09-24 — Python 3.12 baseline and CI diagnosis

- Scope: fix remaining CI issues; user explicitly selected Python 3.12+ and
  requested aligned packaging, CI, and documentation.
- Branch / baseline: `develop` / `0b63985`.
- Status: baseline changes accompany this commit; extended CI diagnosis is
  still in progress.

## Changes and evidence

- Raised `Requires-Python` to `>=3.12`, removed the development TOML backport,
  and aligned Ruff, mypy, release builds, and maintained installation docs.
  The CI matrix now has one core and one comprehensive Python 3.12 profile.
  Default local pytest still uses the smoke selection.
- The prior hosted run passed core 3.11/3.12, MPI, packaging, docs, type
  checks, and agent guidance. Python 3.10 exited during collection (code 2).
  The Python 3.12 extended suite independently failed after 29 minutes.
  Raising the Python minimum does not establish that the extended failure
  is fixed.
- GitHub's authenticated raw-log endpoint repeatedly reset CLI connections;
  the connector exposed statuses but did not return log contents. Public job
  pages showed only exit codes. Added escaped pytest failure annotations to
  make exact tracebacks visible on those pages in the next run. The extended
  job stops at its first failure; successful runs retain full collection and
  the existing 60% coverage gate.

## Local validation

- Combined core/smoke selection: **1671 passed, 8 skipped** on Python 3.12.
- Wheel and source distribution built and passed `twine check`; wheel
  metadata requires Python 3.12+, excluding 3.10/3.11.
- Focused mypy, Ruff, workflow YAML/matrix checks, changed-document links,
  and whitespace checks passed.
- An isolated deliberately failing pytest run retained its failing exit code
  and emitted one escaped annotation under `GITHUB_ACTIONS=true`; the same
  test emitted no annotation outside CI.

## Remaining work

- Read the newly annotated extended CI failure and repair the demonstrated
  cause. Do not infer hosted success from the local development dependency
  stack or remove upstream numerical compatibility guards solely because the
  Python minimum changed.

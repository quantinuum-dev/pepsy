# 2026-09-25 — CI installation-message fix and review

- Scope: fix the reported CI failure and assess CI cost and readability.
- Branch / baseline: `develop` / `b82b73d`.
- Commit status: prepared for the authorized commit and push to `develop`.

## Fix

The missing-Nevergrad test expected `pepsy[layout]` after the runtime hint
changed to the GitHub checkout command `python -m pip install '.[layout]'`.
Updated the assertion to require that command. Runtime behavior is unchanged.
Also corrected the nightly workflow comment: VMC/NetKet runs in push CI's
combined extended job, not a separate VMC job.

This corrects the validation gap in the [cleanup handoff](2026-09-25-final-cleanup-polish.md):
the full local run preceded final message edits, and its focused follow-up
missed this test. The [readiness record](2026-09-25-package-readiness.md)
therefore did not establish a passing full suite for the final message text.

## Validation

- Reproduced the exact failure locally before editing.
- Complete tree-layout module: **108 passed**, four warnings.
- Ruff and `git diff --check`: passed.
- Parsed the nightly workflow before and after: configuration is identical.
- Full local suite: **4,606 passed, 121 skipped, one failed** in 315.18
  seconds. The only failure was the version contract: the shared environment's
  editable installation still advertised 0.4.1 while this checkout is 0.5.0.
- Refreshed this checkout's editable installation with `pip install
  --no-index --no-deps --no-build-isolation -e .`; no dependencies changed.
  Public API and package-layout checks then passed: **58 passed**, including
  the failed version contract. Runtime and installed metadata both report
  0.5.0. The complete suite was not repeated after this metadata refresh.
- Hosted CI for this fix: pending push; prior run failed one of eight jobs.

## CI review

[Run 36187304313](https://github.com/quantinuum-dev/pepsy/actions/runs/36187304313)
spent **30.70 minutes** in the extended job, including **28.50 minutes** in
its test step. Each of the other seven jobs took **0.15–2.53 minutes**.
These are observed job durations, not billing figures or forecasts.

The workflow has useful separation between minimum core dependencies,
optional backends, two MPI sizes, wheel installation, strict docs, and focused
type checks. It is readable and thorough, but not lightweight: every push
to either branch runs the full optional suite, while nightly repeats much of
that suite without the VMC extras. Lint/syntax checks repeat across profiles,
and several jobs install the whole development extra.
Hosted jobs currently exercise Linux and Python 3.12 only; this is not
evidence for every platform or newer Python version allowed by metadata.

Proposed improvements, not implemented in this fix:

- Cancel superseded runs on the same branch using
  [workflow concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).
- Add [pip caching](https://github.com/actions/setup-python#caching-packages-dependencies)
  keyed from package metadata. This can reduce installation work; it will not
  eliminate the dominant numerical test time.
- Consolidate duplicate static checks, narrow per-job tool installs, and add
  explicit job timeouts. Review the overlap between push and nightly runs
  before changing when numerical coverage is required.

CI dependencies are separate from Pepsy's runtime installation. The release
workflow only builds and uploads GitHub artifacts; it does not publish to
PyPI or TestPyPI. No numerical coverage gate, trigger, dependency, release
tag, or branch policy changed here.

# 2026-10-01 — Validate and publish pending package changes

- Scope: the user explicitly authorized committing and pushing both Pepsy and
  the sibling Pepsy Examples repository.
- Pepsy branch / baseline: `develop` / `5eadd9f`. Fetched `origin/develop`
  at `c796530`; preserve both the local PEPS fixes and upstream MPO work.
- Pending package work: exact pure-vector tree sampling, experimental factor
  sampling and bounded call-local caches, sampler lifetime fixes, and correct
  SciPy termination messages. Include their existing API documentation,
  regressions, and dated evidence; keep factor sampling opt-in.

Fresh pre-publication checks in the existing Python 3.12 environment:

- Factor/sampler/canonical/entropy/unitary-stability/gradient/public-API/package
  suites: **444 passed, one skipped**, four compatibility warnings. The skip
  requires two CUDA devices; available single-device paths were exercised.
- `python -m ruff check src tests` and `git diff --check`: passed.
- Examples plotting: **81 passed**; both analysis notebooks retain their saved
  figures. Initial energies were independently checked against diagonal-wall
  preparation, including the analytic extension through 9×10.

The earlier CPU full-suite result remains **6,493 passed, 287 skipped, seven
failed**; this is not a full-suite pass. The failures were reproduced in
unchanged MPS/PEPS tests, as recorded in the
[factor integration handoff](2026-10-01-tree-factor-sampling-integration.md#finalization--2026-10-01).
The full suite is not repeated solely for publication. Commit and push status
is reported after remote integration and final branch verification.

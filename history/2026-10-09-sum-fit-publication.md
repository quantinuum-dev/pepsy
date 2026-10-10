# 2026-10-09 — Sum-target FIT publication

- Scope: user-authorized publication of cached sum-target FIT and its Gaugy
  MPOEngine integration; unrelated MPS, PEPS, and example edits are excluded.
- Publication baseline: `origin/develop` at `8d53933`, using an isolated
  checkout. The shared checkout's branch, index, and working files are preserved.

## Changes

Publish the implementation, regressions, API guide, and evidence from the
[sum-target work](2026-10-09-fit-sum-targets.md) and
[cache review](2026-10-09-fit-sum-cache-review.md). Their original uncommitted
status describes the earlier sessions. Source and tests in this publication
match the reviewed local implementation byte for byte.

## Validation

With the shared Python 3.12 environment activated and the isolated source
selected explicitly by `PYTHONPATH`:

`python -m pytest -q -o addopts='' tests/test_fit_sum_targets.py
tests/test_mps_fit_kernels.py tests/test_optimize_mpo.py tests/test_fit_hotpaths.py
tests/test_fit_gate_schedules.py tests/test_public_api.py
tests/test_package_layout.py`: **373 passed**, eight warnings, 49.89 seconds.

`python -m ruff check src tests` and `git diff --check` passed. Gaugy's isolated
MPO integration snapshot also passed 358 tests against this Pepsy source.
No full-suite, GPU, or compilation claim. Publication targets `origin/develop`;
the confirmed push result is reported after Git completes.

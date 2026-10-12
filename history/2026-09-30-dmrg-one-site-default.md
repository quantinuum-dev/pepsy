# 2026-09-30 — One-site default for generic MPS DMRG

- Scope: user requested `fit_block_size=1` by default for generic `dmrg`,
  avoiding two-site SVD updates.
- Branch / baseline: `develop` / `69b8294` (already two commits ahead).
- Commit status: user subsequently requested commit and push; this handoff is
  included with the implementation commit. Publication outcome is reported
  in the session response and can be checked against the remote branch.
  Pre-existing operator, examples, changelog and other session edits preserved.

## Changes and findings

- Generic `dmrg` and `fit` resolve omitted/None block size to one; private
  replay default agrees. Named schedules and explicit block choices remain.
- Native one-site preparation skips dense bond padding and preserves sectors
  for existing native guess construction. Native reference tests caught this
  preparation issue after changing the default; their numerical assertions
  were preserved.
- Added dense exact-state and native U1/U1U1 one-site schedule checks. Existing
  tests of block truncation and exact pair shortcuts now request block size
  two explicitly; timing-only sweep counts explicitly disable early stopping.
- Updated API docs, changelog, and MPS/FIT guidance. Existing catalog and
  upload-manifest entries already cover the changed skills/resources.
- Target/guess preparation may still use SVD. One-site FIT can retain
  redundant guess bond dimensions. No speedup or fully SVD-free replay claim.
  See the [implementation and upstream audit](../docs/development/notes/2026-09-30-dmrg-one-site-default.md).

## New validation

Activated the existing Python 3.12 environment; numerical checks used one
CPU thread, Agg, and hidden CUDA devices. No dependency changes.

- Final MPS domain, FIT schedule/hotpath, public API and layout checks:
  `python -m pytest -q -ra -o addopts='' -m 'not slow'
  tests/test_optimize_mps.py tests/test_mps_*.py
  tests/test_fit_gate_schedules.py tests/test_fit_hotpaths.py
  tests/test_public_api.py tests/test_package_layout.py`:
  **1031 passed, 37 skipped, 33 deselected** in 101.92 s.
  Log: `/tmp/pepsy-one-site-domain-final.log`.
- Native fermionic suite separately: **149 passed, 33 slow deselected**.
- Ruff `src tests`, `git diff --check`, both changed skill validators,
  skill catalog validation and affected local documentation links pass.
- Earlier broad run was interrupted during slow native stress tests after
  exposing eight native-padding failures. The preparation correction passes
  the final non-slow suite; slow stress completion is not claimed.

## Limits

No full-package suite or GPU execution claim. Skips require CUDA/CuPy/Metal;
the 33 slow native stress tests were excluded from final validation. No running
application jobs or datasets were changed.

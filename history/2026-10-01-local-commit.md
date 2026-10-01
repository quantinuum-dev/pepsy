# 2026-10-01 — Commit pending metric and gate fixes

- Scope: user requested local commits in Pepsy and Gaugy.
- Branch / baseline: `develop` / `7773d2e`.
- Commit status: included in the local fix commit; nothing pushed.

## What changed

- Committed the pending metric-cap precedence, initial sweep normalization,
  exact-target gate-cap, and dimensional final-compression fixes, with their
  tests, API documentation, changelog, and pending review records.
- Corrected the optional-cap regression identified by the
  [post-fix review](2026-10-01-post-fix-review.md): explicitly selected mapping
  `chi=None` now reaches exact metrics or existing boundary handles.
- Run records preserve the resolved optional cap. Delegated sweep/global
  normalization distinguishes an omitted cap from an explicitly resolved
  `None`. Named numeric cap validation remains unchanged.
- Added real exact norm/infidelity checks for per-call, stored, and shared
  mappings, plus optional per-run cap and record checks.
- Preserved historical review statements and unresolved TreeSampler findings;
  this commit does not implement the deferred sampler changes.

## Fresh validation

Activated the shared `envs/py312` environment for all Python commands.

- Before the optional-cap fix, its regression selection reproduced five
  failures; the existing four numeric-precedence cases passed.
- CPU affected suite: `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -o addopts=''
  tests/test_optimize_peps.py tests/test_gate.py tests/test_gate_cutoff.py
  tests/test_optimize_global.py tests/test_prepare_boundary_inputs.py
  tests/test_public_api.py tests/test_package_layout.py`: **540 passed,
  1 skipped**, 22.07 seconds. Skip: CuPy CUDA availability.
- `python -m ruff check src tests` and `git diff --check`: passed.
- Relative file links in all eight previously pending Markdown records resolve.
- No fresh full-suite or GPU validation; the earlier full-suite resource
  failures remain documented in the
  [fix handoff](2026-10-01-metric-gate-fixes.md).

The [compatibility note](../docs/development/notes/2026-10-01-metric-gate-compatibility.md)
records the upstream capability audit. No dependency or numerical contraction
policy changed during this commit preparation.

# 2026-10-01 — Preserve initial sweep normalization cap

- Scope: user requested fixing the issue in the latest-commit review.
- Branch / baseline: `develop` / `7773d2e`.
- Commit status: working-tree changes only; nothing staged, committed or pushed.

## Changes

`SweepOptimizer` constructor renormalization now defaults to the stored
`normalize_kwargs["chi"]`, falling back to the environment `chi` when absent.
Explicit `renormalize_kwargs["chi"]` still overrides only the initial call.
This completes `PepsOptimizer`'s independent normalization-cap forwarding.
Updated the PEPS/sweep API guides and changelog.

Four regression cases exercise actual constructor and metric contractions,
with only the numerical optimizer run replaced by a subsequent normalization:
automatic caps on both engines, an explicit pair, and an initial-only override.
They check routing, dense normalization and input preservation. Before the
fix, three cases failed at the initial cap and the explicit-override case
passed. See the [review](2026-10-01-latest-commit-review.md) for the diagnosis.

## Fresh validation

Activated the existing `envs/py312` environment in each Python shell.

- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -o addopts=''
  tests/test_optimize_peps.py tests/test_prepare_boundary_inputs.py
  tests/test_public_api.py tests/test_package_layout.py`: **343 passed**, five
  warnings, no skips, 20.24 seconds.
- `python -m ruff check src tests`: passed.
- `git diff --check`: passed.
- Reviewed the final diff; existing review documents were preserved.

No full-suite run, dependency changes, numerical algorithm changes or sibling
repository edits. The previously recorded TreeSampler issues are outside this
fix's scope and remain unresolved.

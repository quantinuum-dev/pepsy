# 2026-09-28 — Readability inventory and vector API clarification

- Scope: the user approved a ranked readability/API audit and improvements
  one domain at a time. This batch completed the inventory and vector-sampler
  documentation corrections.
- Branch / baseline commit: `develop` / `4e398e4`.
- Commit status: uncommitted working-tree changes; no new commit or push.

## Findings and changes

- Parsed all 211 source Python modules; manually reviewed selected orchestration,
  sampling, result, and builder entry points. This is structural coverage of
  the package, not a claim that every line or function was manually verified.
- Recorded six priorities with concrete owners and acceptance criteria in
  the [readability/API review](../docs/development/notes/readability_api_2026_09.md).
- Corrected `VecSampler`'s nonexistent constructor `basis` parameter and stale
  fixed-backend claim on `refresh()`. Clarified normalization, source/cache
  lifetime, shapes, conversion defaults, Torch gradients, and joint weights.
- Added a small executable example, a sampler chooser, and a vector result
  table to existing source/API documentation. No new dependency or test file.
- Link validation found a stale reference to the deleted benchmark script;
  replaced it with the examples guide and retained the dated study as history.
  The deleted directory was not restored.

## Validation

- AST comparison against the baseline after removing docstrings: all
  executable nodes and function signatures unchanged.
- New class-docstring example: **3 statements passed**.
- `MPLBACKEND=Agg python -m pytest -q -ra -o addopts='' tests/test_sampler.py -k vec_sampler`:
  **16 passed, 1 skipped, 90 deselected** in **1.40s**. The skip requires CuPy,
  which is not installed in this environment.
- `python -m ruff check src tests`: passed.
- Changed Markdown local file links and `git diff --check`: passed.
- No full numerical suite or hosted CI run for this documentation-only batch.
  No numerical upstream audit was needed because implementations are unchanged.

## Remaining work

The ranked refactors are proposals. The next bounded code change is reviewing
the duplicate sub-MPO parser against its existing shared implementation,
including private callers and import boundaries. Long numerical kernels and
public signature changes remain separate tasks with domain-specific validation.

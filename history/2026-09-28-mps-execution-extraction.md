# 2026-09-28 — Extract MPS replay dispatch

- Scope: improve MPS optimizer readability by separating mode dispatch and
  control-delimited replay from the main optimizer implementation.
- Branch / baseline commit: `develop` / `1b3dd96`.
- Commit status: included in the local commit containing this entry; no push.
  Earlier test-tier edits remain outside this commit.

## What changed

- Added `mps/_execution.py` for mode dispatch and gate/control segmentation.
- Bound both helpers back onto `MpsOptimizer`; all state and mode-specific
  behavior remains on the live optimizer and continues to dispatch through its
  methods.
- Moved the default FIT initialization constant with its dispatch helper and
  documented the new module in `docs/development/modules/optimizers.md`.
- Preserved the moved function signatures and executable bodies. The only
  differences are documentation/comments and the equivalent constant name.

## Validation

- Ruff and `git diff --check` passed for the changed Python files.
- An AST comparison against the original commit confirmed both moved function
  signatures and executable bodies are unchanged.
- After the user requested behavioral validation, ran all MPS test modules
  plus public API, package layout, and import-boundary checks:
  `MPLBACKEND=Agg python -m pytest -q -ra -o addopts='' tests/test_optimize_mps.py tests/test_mps_*.py tests/test_public_api.py tests/test_package_layout.py tests/test_import_boundaries.py`.
  Result: **981 passed, 37 skipped, 20 warnings** in **280.08 seconds**.
- Skip reasons: 20 missing-CuPy cases, 16 unavailable-CUDA cases, and one
  unavailable-Metal case. Those hardware paths remain unvalidated here.
- The whole-package suite was not rerun after this extraction; its earlier
  passing result predates the change.

## Follow-up: separate `run()` preparation and execution

- Scope: the user approved simplifying `run()`, testing the affected MPS
  paths, and committing the verified refactor.
- Extracted layout-plan selection and queue mapping into the existing
  `_layout_execution.py`; mode-option validation has an explicit optimizer
  helper; the existing `_execution.py` now owns prepared replay and its shared
  temporary-layout cleanup.
- Reduced the implementation after `run()`'s docstring from 522 to 194 lines.
  The public signature/docstring, shot routing, empty-stream behavior, and
  option-validation order are preserved. No new configuration container or
  dependency was introduced.
- Removed the duplicated cleanup block for gate-only and control-event
  replay. Added three failure-path cases for ordinary gates, measurement, and
  a cap that shortens the register, plus one check that layout/legacy-option
  warnings still point to the caller.
- Static comparison confirmed that the extracted option-validation logic and
  all other existing numerical methods are unchanged. Warning stack levels
  account for the new preparation helper frame.
- Initial focused check: **4 passed**. The broader selection above, plus
  `tests/test_trajectory_noise.py`, completed with **1,081 passed, 37 skipped,
  20 warnings** in **280.46 seconds**. Skip reasons are the same 20 missing
  CuPy, 16 unavailable CUDA, and one unavailable Metal cases.
- Ruff across `src` and `tests`, whitespace checks, and comparisons of the
  moved layout preparation and shared cleanup bodies passed.

## Limits

- Numerical mode implementations and live state remain on `MpsOptimizer`.
  No numerical algorithm or public API changed.
- The refactor, its regressions, module map, and this handoff are committed
  locally together. Nothing was pushed.

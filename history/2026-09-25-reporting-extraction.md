# 2026-09-25 — GitHub-only release workflow and MPS reporting extraction

- Scope: implement the first two changes accepted after the simplification
  review: remove registry-publishing jobs and extract MPS reporting helpers.
- Branch / baseline: `develop` / `7bde713`, preserving the existing Markdown
  readability edits and review handoffs.
- Commit status: included in the documentation and MPS reporting cleanup commit.

## Changes

- Release workflow now contains only the existing build/metadata/install
  validation and artifact-upload job. Version-tag and manual triggers remain;
  PyPI/TestPyPI inputs, publishing jobs, unused job output, and OIDC permission
  are removed. Workflow length decreases from 105 to 59 lines.
- The existing MPS `diagnostics.py` now owns private FIT timing aggregation
  and layout-report formatting. It has no imports or public exports and does
  not read clocks or access optimizer state.
- Optimizer delegates retain class formatting hooks, including subclass
  overrides. Timing collection, replay, normalization, and numerical
  diagnostics retain their existing ownership. `optimizer.py` decreases
  from 11,970 to 11,835 lines.
- Added focused regression coverage and updated the import-boundary check,
  module map, development guide, and Unreleased changelog.

## Validation

- API, package-layout, import-boundary, and new reporting regressions:
  **75 passed**, with expected compatibility deprecation warnings.
- Existing MPS layout/compression tests selected by `timing or report`:
  **13 passed, 137 deselected**, with one expected layout deprecation warning.
- After extending the lazy-import test to call the reporting helpers in a
  clean process, that test passed again. These checks used the current source
  tree and its matching 0.5.0 metadata, not the older installed wheel.
- AST comparison confirms all 255 optimizer methods outside the three
  formatting helpers are unchanged. The moved timing-summary function has an
  identical AST. No numerical upstream behavior was changed or re-audited.
- Parsed workflow checks confirm unchanged build steps and tag trigger,
  retained manual dispatch, one build job, and only `contents: read` permission.
- Ruff, changed-documentation links, and `git diff --check` passed.
- Strict Sphinx HTML build passed with an empty diagnostic log.

## Limits

The full numerical suite and hosted workflows were not rerun. This is an
organization change, not a measured numerical speedup or dependency-size
reduction. Existing remote release tags/workflows are unaffected until changes
are committed and pushed; no package publication was attempted.

# 2026-09-24 — Canonical imports and lazy discovery

- Scope: implement priorities 1 and 2 from the user-approved
  [package simplicity assessment](../docs/development/notes/package_simplicity_2026_09.md).
- Branch / baseline commit: `develop`, `64f7e1f`.
- Commit status: uncommitted; nothing staged or published. Existing unrelated
  changes and the device-local override were preserved.

## What changed

- README, getting-started, API navigation, and the flat-PEPS example now teach
  owning-namespace imports. Added `interop` to missing ownership tables and
  clarified optional installation, advanced workflows, and experimental status.
- The layout guide describes current ownership rather than asking agents to
  repeat an already-completed tensor module split. The discovery facade is
  explicitly separate from implementation ownership.
- Added `__dir__` to the root, nine core entry namespaces, `experimental`, and
  `vmc`. It combines advertised exports with existing attributes without
  resolving names. Nested implementation packages were not restructured.
- Synchronized root `TYPE_CHECKING` imports with the existing export registry:
  70 previously missing symbols now have declarations. The 309-name root
  compatibility registry, aliases, and runtime resolution remain unchanged.
- Added a fresh-process discovery regression and a static export-declaration
  consistency check. Updated the public package guide and changelog.

## Validation

- `MPLBACKEND=Agg python -m pytest -q -o addopts='' tests/test_import_boundaries.py
  tests/test_public_api.py tests/test_package_layout.py`: **56 passed**, with
  8 existing compatibility deprecation warnings.
- Discovery regression covers all 12 changed entry namespaces: advertised
  exports and existing attributes remain visible, no imports or attribute
  caching occur during `dir()`, and no warnings are emitted.
- Both README Python examples and `examples/pauli_mpo_trace_flat_peps.py`
  executed successfully with the selected local environment.
- Ruff passed for `src`, `tests`, and the changed example. All 12 skills passed
  catalog validation; local Markdown link targets and whitespace checks passed.
- No numerical implementation changed. The full numerical suite was not rerun;
  earlier recorded failures remain outside this task. No installation-size or
  numerical-speed improvement is claimed.

## Deferred work

Internal compatibility-import migration, dependency-extra consolidation, and
large-module extraction remain proposals. Existing 0.x alias compatibility
promises still apply. These deferred items do not expand a future agent's scope.

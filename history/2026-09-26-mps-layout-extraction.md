# 2026-09-26 — Commit module organization and extract MPS layouts

- Scope: user approved committing the validated changes and continuing with
  MPS layout handling as the next responsibility-based split.
- Branch / baseline: `develop` / `22bff20` at session start.
- Commit status: the preceding symmetry/MPS organization is committed locally
  as `23b5e35`. The validated layout extraction is recorded in the local commit
  containing this entry; no push or release.

## Implemented

- Rechecked all 190 source modules against the previously validated wheel;
  no intervening source changes were present. Reviewed and committed the
  work described in the [previous handoff](2026-09-25-module-organization.md).
  Its full-suite result belongs to that earlier refactor, not this extraction.
- Extracted 21 layout operations into `mps/_layout_execution.py` (780 lines).
  `mps/optimizer.py` shrank from 11,124 to 10,508 lines. Public signatures and
  documentation stay on the optimizer; private aliases retain normal method
  and staticmethod binding. No new base class or state container was added.
- Layout helpers own selection pilots, installation, mapping, reordering,
  schedule installation, and logical readout. Canonicalization, norm tracking,
  native swaps, stream validation, and replay remain optimizer hooks. Geometry
  search stays in `layout.py`.
- Updated API/module documentation, the MPS skill, changelog, and dated
  [compatibility evidence](../docs/development/notes/mps_layout_scheduler.md).

## Validation

- AST comparison: all 21 moved operation bodies/signatures and all 236 other
  optimizer methods are unchanged. Public layout docstrings also match.
- New regressions cover subclass hooks through product-state reordering,
  logical readout and independent copies, and importing the helper module
  without the optimizer/FIT. The initial new numerical assertion compared a
  vector with Quimb's column-vector output; flattening both reference/readout
  values corrected the test oracle without changing production code.
- Focused layout, layout-upgrade, diagnostics, public API, package-layout,
  and import-boundary suites: **142 passed**.
- Ruff, focused mypy, MPS skill validation, the 12-skill catalog, and
  whitespace checks passed.
- Strict Sphinx build passed with no diagnostic output. **4,081** rendered
  local links/anchors resolved across 11 pages, and all **60** existing public
  `MpsOptimizer` API anchors remain present.
- Built an sdist and wheel from it; all **191** wheel Python modules match
  working-tree source. Installed-wheel persistent layouts, repeated replay,
  logical readout, and independent copying passed outside the checkout.
  Build tools and the test installation stayed under `/tmp`.
- Full local suite: **4,621 passed, 121 skipped**, 728 warnings, in
  325.32 seconds. Skips cover unavailable CUDA/CuPy, sandbox Metal, and
  single-process MPI cases. No new GPU or multi-rank validation is claimed.

## Limits

Other large optimizer modules and symmetric MPO construction remain separate
future tasks. No new numerical algorithm, dependency, or performance claim is
part of this extraction. Remote branches and release tags are unchanged.

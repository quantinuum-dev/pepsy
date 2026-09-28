# 2026-09-28 — Tree-PEPS two-layer bond diagnostics fix

- Scope: user's authorization to fix the confirmed defect from the
  [Pylint review](2026-09-28-pylint-design-review.md).
- Branch / baseline: `develop` / `f410fa0`; earlier uncommitted readability
  work is preserved. This fix is not staged, committed, or pushed.

## Changes

- Each FIT, two-layer, and fused application branch now supplies
  `uncompressed_bonds` together. One common calculation initializes
  `transient_max_bond` before report assembly. Two-layer diagnostics use the
  existing operator/state bond products without constructing a fused network.
- Preserved contraction calls, seed handling, normalization, state assignment,
  and public signatures. The fix removes the post-update `UnboundLocalError`
  when `track_bond_diagnostics=True` on the two-layer path.
- Strengthened the existing three-mode two-layer/fused regression with bond
  diagnostics enabled, checking transient dimensions, the chi comparison,
  live bond dimension, agreement between layouts, and exact dense output.
  No additional test cases, source modules, dependencies, or CI changes.
- Documented the diagnostic meaning in the Tree-PEPS guide and the fix in
  the unreleased changelog.

## Validation

- Before the implementation fix, the strengthened regression failed in all
  three existing parameter cases (`sdc`, `src`, `zipup`) with the reproduced
  `UnboundLocalError`: **3 failed, 85 deselected in 1.43s**.
- After the fix, `python -m pytest -q -ra -o addopts=''` on
  `tests/test_tree_peps_optimizer.py`, `tests/test_tree_peps.py`, and
  `tests/test_tree_pepo.py`: **130 passed, 2 warnings in 1.47s**, exit 0.
  Warnings describe Quimb's default SVD options and the existing SDCR cutoff
  compatibility policy.
- Ruff on `src tests` passes. Targeted Pylint 4.0.9 checks
  `used-before-assignment` and `possibly-used-before-assignment` both pass
  for `src/pepsy/optimizers/tree_peps/optimizer.py`, exit 0. No rule suppression
  or project lint configuration was added.
- Default smoke: `MPLBACKEND=Agg python -m pytest -q` → **89 passed,
  2 compatibility-alias warnings in 16.96s**, exit 0.
- All 49 local Markdown links in changed guides/handoffs and
  `git diff --check` pass. Nothing is staged.

## Limits

The full suite was not rerun for this single-module bookkeeping fix. The
earlier full-suite and package-wide Pylint results remain historical results
recorded in the linked review; this does not resolve the other Pylint findings.
No numerical kernel or upstream dispatch behavior changed, so the continuing
task's compatibility audit remains applicable.

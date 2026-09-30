# 2026-09-24 — Package simplicity assessment

- Scope: assess a lighter, clearer package, fewer aliases, and agent guidance
  against current public Python packaging guidance.
- Branch / baseline commit: `develop`, `64f7e1f`.
- Commit status: uncommitted; no staging, publication, or dependency changes.

## Findings and changes

Saved the [assessment](../docs/development/notes/package_simplicity_2026_09.md)
and linked it from the notes index. It distinguishes existing strengths from
proposals: canonical import examples, lazy discovery, internal compatibility
imports, extra composition, and gradual module extraction. No implementation
or agent-policy change was made in this review.

## Validation

- Import boundaries, public API, and package layout: 54 passed, 8 deprecation
  warnings, using the selected local environment and `MPLBACKEND=Agg`.
- Fresh-process import probe found no NumPy, Quimb, SciPy, Torch, JAX, or
  Symmray imported by the root facade. One timing observation is not a benchmark.
- Checked new local Markdown link targets and whitespace.
- Full numerical tests were not rerun; prior recorded failures remain outside
  this assessment. Hosted CI and installation size were not measured.

## Proposed next step

Align canonical import documentation and add lazy namespace discovery before
moving implementations. Existing 0.x compatibility promises remain in force;
these recommendations do not authorize agents to remove aliases.

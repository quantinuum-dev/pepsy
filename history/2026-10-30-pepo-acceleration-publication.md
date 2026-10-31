# 2026-10-30 — PEPO acceleration publication

Requested scope: commit and push the prepared PEPO acceleration and its
dependency changes. Branch: `develop`; baseline: `e658bed6`.

This publication commit packages saved-factor QR backward, dense gauge-scale
reuse, `PreparedSimpleUpdate`, focused tests and API documentation. The earlier
[QR](2026-10-30-pepo-qr-reuse.md) and
[prepared-SU](2026-10-30-prepared-simple-update.md) records retain their original
uncommitted baseline statements. Unrelated MPS/sampling edits remain local.

Fresh validation used an isolated export of the selected Git index, alongside
the selected Gaugy package, with the existing Python 3.12 environment:

- 296 tests passed across prepared SU, saved-factor QR (CPU/CUDA), adaptive
  and captured QR, SU scale/zero-weight handling, gauge scales, public API and
  package layout. Expected singular-chart/deprecation warnings remain.
- Ruff over `src tests` and staged whitespace checks passed.
- Downstream Gaugy: 239 tests passed across prepared PEPO, gradients,
  conventions, tensor engines and public package behavior.

No optimization was rerun. The previously documented native-Z2 failures are
outside this selection and remain unresolved; this is not a full-suite result.

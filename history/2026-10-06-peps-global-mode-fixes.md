# 2026-10-06 — Fix global PEPS mode

- Scope: user authorized the issues identified in the global-mode review.
- Branch / baseline: `develop`, `8f7c896`.
- Commit status: working-tree changes only; prior PEPS work and concurrent
  solver changes preserved. Nothing staged, committed, or pushed.

Global output now returns to its input backend/dtype/device, including native
blocks. The PEPS driver owns final normalization and honors the opt-out.
Standalone normalization honors its own option mapping. NLopt restores its
best finite evaluated vector and exposes recovery diagnostics and the correct
returned score; a run with no valid candidate retains the input/warm start.

Added 17 regressions and updated API docs/changelog. Final focused validation:
**479 passed**, covering new recovery tests, real NumPy/Torch multi-batch runs,
complex64/complex128, native U1/U1U1, boundary engines, sweep safeguards,
public API, and package layout. Ruff, local links, and diff checks passed.
No full suite or GPU validation.

Detailed changes, Quimb compatibility-shim contract, numerical recovery
evidence, and limits are in the
[implementation note](../docs/development/notes/2026-10-06-peps-global-mode-fixes.md).
This supersedes the earlier review's unfixed status for these findings.

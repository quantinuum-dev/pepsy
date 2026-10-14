# 2026-10-13 — PEPS tVMC paper review and stability diagnostics

- Scope: user requested code/API improvements, then comparison with Wu and
  Nys emphasizing implementation and numerical stability.
- Branch / baseline commit: `develop` / `9b2bef2`.
- Commit status: working-tree edits; nothing staged, committed or published.

## Changes and rationale

- Torch TDVP exposes the squared unregularized normal-equation residual,
  complex sampled tangent residual and absolute tangent norms. A regression
  shows an exactly solved normal equation can miss the full physical motion.
  Direct/minSR and holomorphic/real-coordinate cases share these diagnostics.
- Both drivers reject holomorphic compressed PEPS; NetKet rejects invalid
  driver controls before optional imports. Its forward-mode QGT rejection also handles known
  constructors wrapped in `functools.partial`.
- Updated public API and changelog; the
  [paper review](../docs/development/notes/2026-10-13-tvmc-paper-review.md)
  maps implemented mechanisms, missing stability ingredients and coding
  priorities. The review does not certify trajectory convergence.

## Validation

- CPU TDVP/NetKet/backend-neutral API/public API/package-layout selection:
  **172 passed**, 245.15 s, before the final Torch chart guard and partial-QGT
  regression. After those edits, Torch TDVP, native flat-Z2 compile/derivative
  checks and selected NetKet guards: **49 passed, 34 deselected**, 60.94 s.
  Selections overlap; this is not a full-suite or GPU claim.
- Ruff over src/tests and whitespace checks passed; 12 local Markdown link
  targets resolved. Expected optional-library/deprecation warnings occurred
  in the broader selection; no failures remained.
- Installed NetKet/JAX/Flax signatures and versions inspected; no dependency
  changes. Deterministic tests only; no new physical benchmark trajectory.

## Remaining work

Accepted-state NetKet diagnostics, a validated real-coordinate sample-space
TDVP solver, explicit sample-null-mode treatment and physical-slice derivative
storage remain proposed. Native Symmray gauge projection and paper-scale
stability have not been established. Full repository suite was not run.

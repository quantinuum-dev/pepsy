# 2026-11-09 — Explicit reusable Pauli damping MPO

- Scope: advance DAOE damping beyond the implicit application helper.
  The user explicitly deferred geometry rules and 2D before implementation.
- Branch / baseline commit: `develop`, `ad523b6`; the prior
  [Pauli review/filter edits](2026-11-09-pauli-mpo-daoe.md) were already present.
- Commit status: uncommitted working-tree edits; nothing staged or pushed.
  Existing sampling edits and earlier records were preserved.

## What changed

- Added public `operators.PauliDampingMPO`, frozen static parameters, reusable
  diagonal cores, exact factor queries and independent dimension-four Quimb
  superoperator MPO export.
- Added `PauliMPO.apply_damping`; routed `apply_daoe` through the same reusable
  implementation. Kept sparse/native application, dtype/device/gradient and
  no-op/in-place report behavior. Removed the duplicate private helper.
- Updated [API examples](../docs/api/operators/pauli_mpo.md#reusable-damping-mpo),
  operator ownership and changelog. Added
  [audit and evidence](../docs/development/notes/2026-11-09-pauli-damping-mpo.md).

## Validation

- Focused damping/Pauli suites: **123 passed**, including exact MPO matrices,
  Quimb MPO-on-MPS/native agreement, export ownership, coefficient gradients,
  JAX JIT, a 64-site no-expansion case and exact/lossy compression references.
- Public API/package suites: **54 passed, 1 failed**; existing installed Pepsy
  version metadata is 0.5.0 versus unchanged source 0.6.0.
- Ruff, `git diff --check` and local Markdown file-target checks pass.
  Two API examples execute; full suite and accelerator/CuPy runs not performed.

## Remaining boundary

The damping MPO acts in Pauli coefficient space; it is not physical operator
multiplication. Filtering is exact up to arithmetic, but finite damping
changes dynamics, and ordinary SVD can alter protected coefficients.
A local Heisenberg integrator with explicit schedules and convergence
diagnostics remains the next 1D step. Geometry/2D is deferred at user request.

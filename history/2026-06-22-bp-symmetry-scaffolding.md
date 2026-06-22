# 2026-06-22 — BP / quimb / symmetry scaffolding

- Milestone: M0 — scaffolding
- Branch / commit: `develop` (committed in this snapshot)

## What changed
- Added `PLAN.md` (root): roadmap for three workstreams — Belief Propagation
  (+ loop/cluster expansion), quimb integration, and symmray-backed symmetric
  & fermionic tensors.
- Added `history/` (this folder) with `README.md` + entry template for
  session-to-session hand-off.
- Added `learning/` with conceptual docs: `bp.md`, `quimb.md`, `symmray.md`,
  plus `README.md` index.

## Why
- Capture the intended direction and the source material (two BP papers +
  symmray) before any code lands, and set up a durable channel for passing
  context between agent sessions.

## How it was validated
- Documentation-only change; no code or tests touched. Not run: `pytest`.

## Decisions / findings
- BP belongs in a new subpackage that mirrors `boundary/` (peer to
  boundary-MPS contraction), not inside `boundary/`.
- Symmetric tensors are an array/backend concern (symmray via autoray), kept
  optional; algorithms stay backend-agnostic.
- Prefer wrapping `quimb`'s BP/gauging over hand-rolling; final wrap-vs-own
  decision deferred to M1 after a quimb API audit.

## Next step (do this first next time)
- M1: audit `quimb.tensor` BP + gauging API; prototype a BP fixed point on a
  3×3 PEPS built via `pepsy.build_bra_ket`, and compare the BP norm against
  `pepsy.contract_boundary` at large `chi`. Record the API mapping in
  `learning/quimb.md`.

## Open questions / blockers
- Exact pepsy-vs-quimb division of labor for BP (see PLAN.md §7).
- symmray version to pin (currently v0.2.x).

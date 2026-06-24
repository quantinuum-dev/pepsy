# 2026-06-24 — Dimension-aware SWAP routing

- Milestone: B0 — gate-routing audit / quimb integration prerequisite
- Branch / commit: `main` working tree, uncommitted

## What changed
- Added dimension-aware routed SWAP construction in `src/pepsy/operators/gates.py`.
- Routed `gate(..., contract="split"|"reduce-split")` in 1D/2D/3D now infers
  the live physical dimension at each adjacent SWAP step.
- Routed `gate_simple(...)` now does the same for simple-update SWAP chains.
- Added mixed-dimension PEPS tests for direct `split`, direct `reduce-split`,
  and simple-update routing through a spectator site.
- Updated `PLAN.md` with the quimb audit finding and the implementation scope.

## Why
- Tensy PF PEPS replay has mixed physical dimensions (`dim=4` frame sites,
  `dim=2` measurement sites, and potentially larger selector sites). The old
  hard-coded `qu.swap(dim=2)` route was only valid for DEM-style binary sites.
- Quimb can already apply adjacent rectangular two-site tensors with
  `split`, `reduce-split`, and `gate_simple_`, so Pepsy only needed to build
  the correct SWAP tensor per live adjacent pair.

## How it was validated
- `python -m pyflakes src/pepsy/operators/gates.py tests/test_gate.py` -> passed.
- `python -m pytest -q tests/test_gate.py` -> `92 passed`, 2 warnings.
- `python -m pytest -q tests/test_public_api.py tests/test_package_layout.py` ->
  `404 passed`.

## Decisions / findings
- No new public flag was added. Dimension-aware SWAPs are now the internal
  default for routed paths; binary sites still use an exact binary SWAP.
- Generic/mocked tensor networks that cannot report physical index sizes keep
  the previous binary fallback for compatibility.

## Next step (do this first next time)
- Wire Tensy PF `to_2dpeps(..., gate_method="simple_update")` or direct
  `gate_opts={"contract": "reduce-split"}` back to this Pepsy route and run the
  `sf_2dpeps_pf.ipynb` PEPS replay cell on a small layout.

## Open questions / blockers
- None for Pepsy routing. Tensy still needs the notebook/runtime-side choice of
  direct `reduce-split` versus simple-update for the PF workflow.

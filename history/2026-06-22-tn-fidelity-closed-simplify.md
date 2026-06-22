# Session - 2026-06-22 - Simplify closed tn_fidelity overlaps

## Goal
Add opt-in closed-network simplification to `pepsy.tn_fidelity(...)` and
`pepsy.tn_norm(...)` so callers can simplify scalar norm/overlap tensor
networks before contracting.

## What changed
- `src/pepsy/tensors/core.py` - `tn_fidelity(...)` now accepts
  `simplify=False` and `simplify_seq="R"`. When `simplify=True`, it builds
  each closed overlap TN, runs `full_simplify_(..., output_inds=())`, then
  contracts.
- `src/pepsy/tensors/core.py` - `tn_norm(...)` now accepts the same
  `simplify=False` / `simplify_seq="R"` keywords for its closed norm network.
- `tests/test_core_seed.py` - added tests proving closed networks are
  simplified when requested and skipped by default.

## How it was validated
- `pytest -q tests/test_core_seed.py`
- A Tensy notebook-shaped cycle-2 check confirmed DEM/PF simplified TNs now
  have matching outer legs, but the full `tn_fidelity(...)` contraction still
  timed out after 90 seconds for that larger global comparison.

## Decisions and constraints
- Simplification is opt-in and happens only after the norm/overlap networks are
  closed, with `output_inds=()`. This avoids the open-output-leg simplification
  issue seen in the PF decoder.
- The default sequence is conservative rank simplification (`"R"`). Callers can
  pass `simplify_seq="ADCR"` for a more aggressive closed-network attempt.

## Open items / next steps
- [ ] Consider a higher-level output-distribution fidelity helper that validates
      matching outer legs and freshens internal indices before calling
      `tn_fidelity`.

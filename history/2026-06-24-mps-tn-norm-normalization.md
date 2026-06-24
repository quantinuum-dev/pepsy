# 2026-06-24 — MPS normalization via `tn_norm`

- Milestone: M1 — non-unitary MPS diagnostics and normalization hardening
- Branch / commit: `main` working tree, pending commit

## What changed
- Switched `MpsOptimizer._canonical_span_norm(...)` from contracting a selected
  canonical span into a dense tensor to measuring the raw working-data norm with
  `pepsy.tensors.core.tn_norm(..., strip_exponent=True)`.
- Temporarily clears and restores `p.exponent` while measuring the raw tensor
  data, so automatic normalization and `track_norm_infidelity=True` do not
  accidentally include the represented global scale.
- Expanded non-unitary norm-infidelity smoke coverage across `dmrg`, `mpo`,
  `swap`, and `svd` modes, with an explicit skip for older quimb versions that
  do not expose `gate_with_auto_swap_`.

## Why
- The old span contraction was correct only when the canonical span stayed
  small. For long-range gates or swap/sweep paths, the span can retain many
  physical legs; contracting it as one dense block can explode even though the
  norm itself is just a scalar double-layer contraction.
- `tn_norm(..., strip_exponent=True)` keeps the measurement consistent with the
  Pepsy normalization model: normalize raw data locally, store the removed scale
  in `p.exponent`, and leave `p.norm()` to report the represented state norm.

## How it was validated
- `python -m pytest -q tests/test_optimize_mps.py` -> `48 passed`.

## Decisions / findings
- The `fallback` argument remains accepted for compatibility, but the measured
  path now uses `tn_norm` directly rather than the old dense-span fallback.
- True overlap fidelity still goes through `tn_fidelity`; this change only
  affects raw norm measurements used by normalization and norm-infidelity
  diagnostics.

## Next step
- Keep the Tensy `to_mps(..., contraction_opt=optimizer_cotengra)` path wired
  through this same contraction optimizer so large DEM/PF non-unitary streams
  use the intended double-layer contraction route for norm diagnostics.

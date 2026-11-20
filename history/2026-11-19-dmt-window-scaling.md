# 2026-11-19 — Larger DMT regions and scaling

- Scope: user requested protection beyond three sites, more accuracy tests,
  profiling at larger chi/L, and a measured implementation improvement.
- Branch / baseline: `develop`, `f5d31006d02821028c890a04b17e4b15b9683d0b`.
- Commit status: uncommitted. Preserved all earlier DMT/MPS/figure edits;
  nothing staged, committed, pushed or published.

## Implemented

- `_region_patterns` indexes the maximal inventory onto crossing bonds once.
  The plan, string ordering, cover choices and tensor mathematics are unchanged.
- Added `examples/dmt_window_scaling.py`: complex128 3/4/5-site compression
  with product-mixture oracles, small exact statevector dynamics, larger-chain
  gated timing/audits, optional profiles and supported Torch SVD policies.
- Existing mixture benchmark helper accepts an optional strength (same default).
- Added focused protection/planning/oracle tests and a public workflow example.

## Measurements and limits

The [numerical record](../docs/development/notes/2026-11-19-dmt-window-scaling.md)
contains protocols and results. Pauli compression: 27 points through L=256,
chi=128, widths 3/4/5, all with actual rank reduction and maximum independently
measured window error 3.28e-14. Ordinary MPO and NumPy cross-checks also pass.
Planning alone speeds up 139x at L=512; complete five-site chi=32 compression
improves 3.584 s to 1.516 s (2.36x). At large chi, QR/SVD still dominate.

Exact L=12, t=3 dynamics shows wider protection is not uniformly better.
Existing optimized seeds with five-site protection give chi=128 final
five-site RMSE 0.0016195 versus 0.0070874 for three-site paper DMT (4.38x).
Plain five-site chi=128 performs badly (0.3320); the outlier is retained.
Negative eigenvalues remain. MPS is much better for this small/short-time
case, and chi=64 is already exact. No universal DMT/MPS superiority claim.

Native CPU Torch SVD fails on two tested mixture inputs. A finite failing
136x136 complex matrix reconstructs with SciPy GESVD to 2.95e-15 relative
error. Optional existing regularized linalg policy provides native SVD with
GESVD fallback. Always-GESVD full evolution was stopped for cost; do not
present it as completed. No backend defaults or installed libraries changed.

## Validation

DMT domain selection: 399 passed, 1 skipped, 1 expected failure. Final focused
new file: 43 passed. Plan equality spans all selectors, multiple physical
dimensions/identity labels and nested/gapped inventories; compressed arrays
are bitwise identical in the legacy/indexed regression. Full region checks
are enabled in focused physical-oracle tests. Ruff/whitespace checks pass;
not a full-suite or GPU result. Upstream audit and exact installed versions
are recorded in the note.

Saved 37 compression cases and 19 exact-dynamics comparisons with raw timing
samples, accuracy CSV, profiles and a summary PNG/SVG. A paired L=512,
chi=64 comparison using the same supported fallback policy improves
6.879 s to 4.894 s (1.41x). Final large-chain fallback runs are recorded in
the aggregate JSON; the always-GESVD attempt is explicitly incomplete.

Completed all 13 long-chain evolution cases, including L=128, chi=128,
five-site protection with the supported fallback policy. Worst sampled
current-target marginal defect is 2.79e-15; largest final trace defect
6.89e-14. These are two-step mixed-state timing/audit runs, not long-time
global accuracy tests. Largest paired compression-oracle error at L=512,
chi=64 is 1.27e-12. Aggregate JSON is complete and the figure was inspected.

# 2026-11-17 — PEPO constrained ALS investigation

- Scope: investigate protected SVD/ALS truncation for fixed PEPO regional maps;
  preserve existing, uncommitted TFIM experiment and unrelated sampling edits.
- Added optional `dmt_refine_method="als"` with guarded equality-constrained
  least-squares half-sweeps in SVD-orthonormal factor bases. Existing default
  remains `"slsqp"`. Per-update regional residual and pair-error checks and
  a final original-map check retain the feasible seed if needed.
- A controlled complex matrix test and a lossy 2×3 PEPO test verify rank,
  regional constraints and local improvement. Focused PEPO optimizer, example,
  joint-update and public-API tests: 111 passed, 8 accelerator skips.
- The same-source 4×5 lossy gate gave feasibility-seed/ALS/SLSQP pair errors
  1.62e-4/9.48e-5/5.30e-5, with protected-density errors below tolerance. On
  the two-step run, ALS reduced local objectives but had a larger worst L error
  against the exact trajectory than the feasibility-only run. Details and
  reproducible commands: [ALS note](../docs/development/notes/2026-11-17-pepo-als-investigation.md).
- ALS is a nonconvex refinement of a feasible seed, not a general protected
  SVD, global optimum, PEPO physicality guarantee, or large-scale algorithm.
  Explicit maps and dense least-squares factor updates remain costly.
- No commit or push requested; changes remain in the working tree.

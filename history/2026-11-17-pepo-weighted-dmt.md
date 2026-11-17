# 2026-11-17 — Weighted PEPO regional objective

- Scope: add the requested weighted soft-region objective with protected hard
  regional constraints to experimental PEPO DMT. The user's earlier ALS/TFIM
  work and unrelated sampling edits remain uncommitted and preserved.
- `dmt_objective_regions` accepts weighted coordinate regions. ALS uses
  nullspace-projected iterative least squares; SLSQP uses the corresponding
  weighted objective and analytic derivative. Maps shared by hard and soft
  regions are built once, and the allocation budget covers their union.
- A single-region probe exposed unstable near-null corrections. Added guarded
  backtracking and the configurable `dmt_objective_max_pair_growth=2.0` seed
  bound; it can restrict the explored feasible set and is not an optimum
  guarantee. Cost reporting uses the PEPO mantissa with exponent separately.
- A fixed-target 4×5 lossy split reduced independent weighted cost from 4.24e-9
  to 2.22e-19 with ALS, while hard density residual stayed below tolerance.
  A two-step weighted run had worse worst-L trajectory error than the prior
  pair-objective ALS and feasibility-only runs. Full numbers and scope are in
  [weighted investigation](../docs/development/notes/2026-11-17-pepo-weighted-dmt.md).
- Closest PEPO optimizer, Ising example, joint-update and public-API tests:
  117 passed, 8 accelerator tests skipped. Ruff and `git diff --check` passed.
  Explicit maps, nonconvex optimization, NumPy complex128 and open-boundary
  restrictions remain; no full-suite or large-scale claim.
- Changes are uncommitted; no commit or push requested in this turn.

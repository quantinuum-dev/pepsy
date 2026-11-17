# 2026-11-17 — 4×5 TFIM PEPO/MPO DMT experiment

- Branch/baseline: `develop`, `b945df1e`. Authorized a numerical comparison
  and PEPO improvement; preserve unrelated pre-existing sampling edits.
- Added [reproducible driver](../examples/pepo_dmt_ising_4x5.py) and
  [results](../docs/development/notes/2026-11-17-pepo-tfim-4x5.md). The full
  20-qubit state-vector reference and independent 48-region readout are part
  of the driver; one-gate ordering/readout is tested separately.
- Added feasibility retry after premature gradient stopping, within the stated
  evaluation budget; map-budget validation before environment allocation; and
  `dmt_refine=False` to skip optional expensive objective refinement while
  keeping hard map residual checks. Nonconvex/global-optimum and finite-boundary
  limitations remain.
- Full one-timestep, two-timestep feasibility-only, lossy local split and
  all-region cap-failure findings are in the results note. No long-time or
  large-chi scaling claim. Changes remain uncommitted unless subsequently
  committed explicitly by the user.
- Focused PEPO optimizer, example and MPO DMT replay tests passed: 95 tests.
  Ruff and `git diff --check` passed. Independent two-step local physicality
  checks found Hermiticity defects and negative eigenvalues of Hermitian parts;
  see the results note for numbers and the qualification.

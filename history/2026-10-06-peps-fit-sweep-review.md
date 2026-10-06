# 2026-10-06 — Review FIT inside PEPS sweeps

- Scope: investigate and report correctness; no new implementation fixes.
- Branch / baseline: `develop`, `8f7c896`.
- Commit status: only new uncommitted review notes; prior PEPS and concurrent
  solver changes preserved. Nothing staged, committed, or pushed.

Reviewed boundary FIT and PEPS local-solver responsibilities, defaults,
canonicalization, cache reuse, gradients, scaling, normalization, and failure
handling. Evidence and reproducible probe parameters are in the
[review note](../docs/development/notes/2026-10-06-peps-fit-sweep-review.md).

Fresh focused validation: 374 PEPS/boundary/hot-path tests plus 22 FIT kernel
tests passed. Exact-reference probes across 11 row/column updates matched
losses within 1.11e-16 and directional gradients within 4.28e-12.

Confirmed a missing returned-candidate validity guard by controlled NaN
solver-output injection; do not present this as a natural default NLopt
failure. Reconfirmed the existing negative-initial-loss early-exit code
issue; its real numerical reproduction belongs to the preceding review.
Both fixes remain proposed. No full suite or GPU validation.

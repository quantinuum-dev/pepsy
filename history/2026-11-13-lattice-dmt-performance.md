# 2026-11-13 — Lattice DMT performance review

- Scope: profile and optimize geometry-aware Pauli-MPO DMT while retaining
  per-split preservation checks and existing backend behavior.
- Branch / baseline: `develop`, `cd71159d`. Work was prepared for a focused
  commit; unrelated sampling edits were present and preserved.
- Reused normalized identity prefixes and suffixes in physical-region checks
  and the workspace trace-environment cache for protected-vector contraction.
  Replaced the local physical-index einsum with a matrix multiply and layout
  restoration. No public API or truncation policy was changed.
- Added tests for cache invalidation on nonmonotonic gate sweeps, equivalence
  to freshly contracted environments, and small/large-scale or zero partial
  traces against independently computed partial operators. A Torch gate
  gradient was checked against finite differences.
- [Performance evidence](../docs/development/notes/2026-11-13-lattice-dmt-performance.md)
  records same-process 5x5 and 6x6 before/after timings. Full physical-region
  panels and the current explicit numerical checks remain in the algorithm.
- Validation: 30 lattice tests passed, including the slow 5x5, 6x6 and 8x8
  checks; the subsequently added Torch gradient check passed separately with
  one rank-deficient QR warning. The existing DMT/cutoff/optimization/large
  selection passed 128 tests with 19 slow tests deselected and six existing
  Torch QR warnings. Ruff and `git diff --check` passed. No new full-suite or
  GPU run was performed; the prior full-suite BP dispatch failure remains
  recorded in the lattice DMT handoff.
- A separate 8x8 complex128 rank-128 cap-112 run took 11.2 seconds for
  compression and agreed with an independent product-mixture oracle on all
  49 plaquettes to `6.72e-15` in Frobenius norm; trace error was `5.85e-15`
  and Ising-energy error `6.81e-15`. This was one CPU run, not a controlled
  speed comparison with the previous 8x8 measurement.

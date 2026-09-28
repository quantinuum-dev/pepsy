# 2026-09-28 — Scalar full traces of cluster expansions

- Scope: implement a faster path when only the trace of one or more ordered
  exponentials is needed.
- Branch / baseline: `develop`, `4e398e4`.
- Commit status: working-tree edits only; nothing staged, committed or
  published. Earlier working-tree changes were preserved.

## What changed

- Added `trace_exp` to compiled MPO/PEPO cluster products and single
  Pauli PEPO bases. It computes connected scalar residuals and sums all
  compatible placements with a bounded, cached subset recursion.
- Kept factor order, finite-lattice bond occurrences, symmetry reuse,
  backend gradients, and unnormalized/normalized output choices.
- Documented its relationship to the trace of a constructed, possibly
  truncated representation in the owning API guides and changelog.

## Validation and limits

- See the [dated trace evidence](../docs/development/notes/2026-09-28-cluster-trace-only.md)
  for the derivation, independent references, CPU 5×6 timings and memory.
- Broad cluster/MPO/PEPO/public API/layout selection: **247 passed**, two
  existing deprecation warnings. Two more trace regressions were then
  added and the complete trace file passed **11 tests**. Full Ruff,
  relative documentation-link checks and whitespace checks passed.
  No new full-suite or GPU result is claimed.
- Complete Torch Dynamo capture was probed and failed in the existing
  Autoray dtype dispatch; eager Torch gradients and JAX JIT gradients pass.
- The trace is of the complete chosen-order cluster expansion, not of a
  rank-truncated or collection-bounded MPO/PEPO. It is a full bosonic trace;
  high-order local matrices and large subset recursions remain costly.

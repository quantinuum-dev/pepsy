# 2026-09-29 — Measure the constructed PEPO

- Scope: the user requires cluster PEPO construction before trace measurement.
- Branch / baseline: `develop`, `b343ced`.
- Publication: local working-tree changes only; no Pepsy commit or push.
  Earlier cluster feature edits and unrelated optimizer/skill work preserved.

## Changes and findings

Public `trace_pepo` and active `.trace()` contract actual PEPO blocks or a
materialized Quimb network. PEPO `trace_exp` constructs that operator with
requested compression; `partition_trace_exp` retains the old scalar shortcut.
Gaugy uses only public Pepsy methods. Dense/graph/routed-square construction,
rank caps, fixed projections, and graph physical index conventions are covered.

Actual-PEPO measurement exposed a short-periodic-axis construction bug hidden
by the scalar endpoint. Construction/compiler preparation now use existing
finite located clusters when an axis can wrap within the cutoff. Independent
dense operator checks preserve full-cutoff exactness and repeated bonds.

See [technical evidence and upstream decisions](../docs/development/notes/2026-09-29-pepo-operator-trace.md)
and [current API](../docs/api/operators/cluster_expansion.md#trace-of-the-constructed-pepo).

## Validation

Shared Python 3.12, CPU, single-threaded BLAS/OpenMP; JAX CPU.
The trace/construction/factorization/reuse selection initially passed 129
tests with one obsolete generic-cache assertion failing after the finite
periodic route fix. That assertion now checks finite-cluster counts and
verified reuse; no numerical tolerances changed.
The final new actual-operator trace module passed **18 tests** in 12.09s,
including compression, Quimb contraction options, dense values, Torch/JAX
gradients at nonzero/zero parameters, periodic full operators, and budgets.
The additional affected-suite selection passed **222 tests** in 121.64s,
including the corrected periodic-cache test, shared plans, square/graph
materialization, fixed compiler/JIT paths, public exports and package layout.
Together the completed runs cover **352 distinct passing Pepsy tests**.
Two existing public-alias deprecation warnings remain.

Gaugy's integration/package selection passed **153 tests** in 35.55s (six
existing Quimb mode warnings), and its remaining API/symmetry/optimization/
numerics selection passed **40 tests** in 23.77s: **193 passed** in total.

No full-suite or GPU claim. Large sparse traces remain budgeted and can be
expensive. Pepsy full source/test Ruff and whitespace checks passed.

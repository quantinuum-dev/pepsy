# 2026-09-29 — Square routing, frozen compression and materialization reports

- Scope: remaining joint-exponential gaps 3–5, after factor reuse and graph
  autodiff from the prior handoff.
- Branch / baseline: `develop`, `b343ced`.
- Commit status: Pepsy changes remain uncommitted and unpublished. Preserved
  unrelated tree/optimizer/noise/skill/docs work, including concurrent edits.

## What changed

- Explicit `layout="square"` routes dense graph PEPO bonds onto square
  virtual legs, including crossing/pass-through wires, with unchanged
  interaction clusters. Auto selection is unchanged.
- `prepare_compression` returns `ClusterCompressionPlan`; `compression=plan`
  replays fixed reference tree subspaces with backend gradients. Smaller
  local ranks are chosen before square routing. Geometry/binding modes are
  cached; numerical projections are explicit user-owned reference data.
- `return_report=True` counts actual product/exponential reuse, exponential
  batches, lower contractions and graph partition products. Gaugy exposes
  reports and reference compression through its public bound operator.

## Validation

Shared Python 3.12 environment, CPU, one BLAS/OpenMP thread; JAX CPU.
The focused cluster/operator/public-API selection passed **400 tests**, with
two existing deprecation warnings, in 180.43 seconds. It includes the prior
385-test selection plus the initial 15 routing/compression/report regressions.
Two subsequently added complex64/complex128 regressions passed separately.
The final routing/compression/report and square-plan selection passed **40
tests** in 15.24 seconds, including those dtype tests and a new explicit
charge-metadata rejection check. Across these runs, **403 distinct Pepsy
tests passed**. Ruff and diff whitespace checks passed again after the final
implementation edits.

Gaugy's focused cluster/PEPO/package selection passed **299 tests**, with
six existing mode-deprecation warnings, in 139.22 seconds. Pepsy full
`ruff check src tests` and diff whitespace checks passed; Gaugy changed-file
lint and documentation link checks are recorded in its handoff.
No full-suite, GPU, native Symmray, or new projected-objective Torch compiler
claim. Existing fixed-factorization/JIT/compile regressions passed. The
external example notebook remains absent, as recorded in the earlier task.

## Limits and decisions

Rank/subspace selection is outside differentiation. Fixed projections
differentiate an approximate operator; refresh the reference explicitly.
Reported reference errors are local, not current/global bounds. Scalar trace
closure remains uncompressed. Routing can multiply virtual dimensions.
Native charge/fermion/string routing is explicitly rejected. No fitting,
boundary, or backend driver code was changed.

See [API](../docs/api/operators/interaction_clusters.md#explicit-differentiable-compression)
and [implementation/compatibility evidence](../docs/development/notes/2026-09-29-square-routing-compression-reports.md).

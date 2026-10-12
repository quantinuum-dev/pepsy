# 2026-09-29 — Exact symbolic MPO channel sharing

- Scope: structural MPO reuse before compact projection, chi convergence,
  peak tensor memory and forward/backward evidence; spin/Pauli scope only.
- Branch / baseline: Pepsy `develop`, `c1ff0f8`.
- Commit status: Pepsy changes remain unstaged/uncommitted; nothing published.
  Earlier changes and unrelated journals are preserved. Gaugy is unchanged
  by this follow-up (its existing local HEAD is `6675450`).

## What changed

`_cluster_channel_structure` computes a conservative exact prefix/suffix
quotient using independent formal residual entries and charge-aware sparse
axis grouping. `cluster_channels` applies this before QR by default, with
`structural_reuse=False` as the prior-gauge opt-out. Separate sum/select maps
at the two endpoints preserve multiplicity. `_cluster_channel_assembly`
accepts those endpoint maps while keeping replay fused and free of SVD/QR.
Native sectors and protected singleton rails are retained.

Public API, MPO guide, README, changelog, implementation map and status ledger
now distinguish exact sharing from reference-based numerical approximation.
A reproducible CPU benchmark records fixed-order chi sweeps and Torch memory
timelines. See [the API](../docs/api/operators/cluster_channels.md) and
[measurements, literature and limits](../docs/development/notes/2026-09-29-symbolic-cluster-channels.md).

## Validation

Shared Python 3.12; single BLAS/OpenMP thread, Torch CPU/CUDA and JAX CPU.
All pytest commands use `-q -o addopts=''`.

- Initial compact-channel regression: **56 passed**, 22.22 s.
- Final new proof tests: **12 passed**, 7.78 s. Covers arbitrary independent
  complex residual values and adjoints, graph crossings, reference-value
  independence, capped-to-exact limits, and four native charge groups with
  zero/nonzero live derivatives and repeated physical charges.
- Broader Pepsy affected selection: **288 passed**, two existing deprecation
  warnings, 215.28 s. Files: `test_cluster_channel_structure`,
  `test_cluster_channels`, `test_cluster_channel_assembly`,
  `test_native_cluster_autodiff`, `test_native_cluster_mpo`,
  `test_cluster_fixed_factorization`, `test_joint_cluster_parity`,
  `test_graph_pepo_autodiff`, `test_graph_pepo_product`,
  `test_pepo_routing_compression_reports`, `test_square_cluster_plan`,
  `test_public_api`, `test_package_layout`. This run included the first eight
  structural tests; the final 12-test run above adds four native-charge cases,
  yielding **292 distinct passing Pepsy cases** across the two runs.
- Gaugy downstream selection: **159 passed**, six existing Quimb deprecation
  warnings, 44.36 s. Files: `test_cluster_channels`, `test_joint_cluster_parity`,
  `test_cluster_pepo_bindings`, `test_cluster_materialization_options`,
  `test_cluster_exponential_api`, `test_package`.
- Ruff across `src`, `tests` and the new benchmark, plus `git diff --check`:
  pass. Local link targets in all eight edited documents exist.
- No full repository suite or large-lattice performance claim.

## Decisions and remaining limits

The exact six-site/order-four example reduces peak bond 37 to 25 and output
entries 13,304 to 6,456 with roundoff-level operator/gradient errors. Smaller
caps reduce tensor memory further but require accuracy checks. Gradient error
need not improve monotonically with operator error.

Preparation still enumerates sparse histories and local dense residuals.
This quotient does not prove global automaton minimality or eliminate every
repeated local contraction recipe. No one-dimensional sharing rule is applied
to PEPO loops/branching. Gaugy remains a caller; traces still measure actual
constructed networks. No dependency or installed-library changes were made.

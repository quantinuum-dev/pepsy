# 2026-09-29 — Fused compact MPO/PEPO assembly

- Scope: fix compact construction during replay, retaining no-SVD/QR autodiff,
  actual returned-operator traces, and Pepsy ownership.
- Branch / baseline: `develop`, `c1ff0f8`.
- Commit status: all Pepsy work remains unstaged/uncommitted; no push.
  Earlier user changes and unrelated journals are preserved.

## Changes and evidence

New `_cluster_channel_assembly` compiles fixed local factors, MPO gap/crossing
wires and square routing directly into compact tensors. `cluster_channels`
adds `pack_residuals` / `bind_assembler`; legacy block packing is diagnostic.
Recursive inputs require exact reference topology within the collection
budget. Square reference gauges now use graph residuals. `_cluster_native`
adds a constant-bound exact sector kernel. The PEPO binding helper permits
singleton-only channel plans without relaxing the older compression API.
Gaugy's existing public adapter needs no algorithm changes.

See the [API](../docs/api/operators/cluster_channels.md) and
[implementation decisions, timing, memory and accuracy measurements](../docs/development/notes/2026-09-29-fused-cluster-channels.md).

## Validation

Shared Python 3.12, one BLAS/OpenMP thread; Torch CPU/CUDA and JAX CPU.
Commands use `python -m pytest -q -o addopts=''`.

- Final Pepsy affected selection: **169 passed**, two existing deprecation
  warnings, 44.94 s. Files: `test_cluster_channel_assembly`,
  `test_cluster_channels`, `test_graph_pepo_autodiff`,
  `test_graph_pepo_product`, `test_pepo_routing_compression_reports`,
  `test_square_cluster_plan`, `test_public_api`, `test_package_layout`.
  Includes 31 new fused construction cases and the original 25 channel cases.
- Earlier broader selection: **215 passed, two compiler tests failed**,
  191.64 s. It additionally covered `test_native_cluster_autodiff`,
  `test_native_cluster_mpo`, `test_cluster_fixed_factorization`, and
  `test_joint_cluster_parity`. Both graph/square compiler failures were fixed
  by moving tree metadata/identity setup outside replay; both pass in the
  final selection above. No numerical failure was waived.
- Final Gaugy affected selection: **159 passed**, six existing Quimb
  warnings, 43.26 s. Its adapter, trace, binding, joint-product, materialization
  and package checks all pass with the finalized Pepsy implementation.
- Source/tests Ruff, whitespace and edited-document local link checks pass.
  No full repository suite, compiler-throughput or large-graph scalability
  claim is made.

## Remaining limits

Positive caps define approximations. Preparation enumeration and local dense
cluster costs remain; replay avoids expanded history/product tensors.
Full symbolic minimization and a compact recursive planner beyond exact
reference enumeration are not implemented. Existing native-PEPO/fermionic
exclusions remain. Gaugy documentation is committed separately; no publication
of these Pepsy working-tree changes is claimed.

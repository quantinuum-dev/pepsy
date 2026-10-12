# Cluster PEPO backend and downstream optimization status

Reviewed 2026-10-12 at `bf83def`, the published Pepsy `develop` → `main` merge.
This page owns the current capability and validation routes. Detailed earlier
updates are preserved in the [archived ledger](notes/2026-10-12-cluster-status-archive.md).
Its older “working-tree” and “unpublished” statements retain their original dates.

## Current capabilities

| Area | Supported scope | Owning guide |
| --- | --- | --- |
| Package ownership | Pepsy owns graph/tensor-network construction, contraction and general solvers; Gaugy consumes public APIs for its objectives | [Package layout](package_layout.md#pepsy-and-downstream-packages) |
| Interaction clusters | Shared graph inventories, ordered product adapters, structural reuse and explicit construction budgets | [Interaction clusters](../api/operators/interaction_clusters.md) |
| Constructed PEPO traces | `trace_exp` traces the selected PEPO with its caps/compression; `partition_trace_exp` names the separate scalar shortcut | [Trace contract](../api/operators/cluster_expansion.md#trace-of-the-constructed-pepo) |
| Native bosonic cluster MPOs | Exact Torch/JAX autodiff and sector-preserving NumPy/Torch compressed assembly within documented modes | [Native MPO scope](../api/operators/mpo_cluster.md#native-block-sparse-cluster-mpos) |
| Repeated cluster evaluation | Experimental compact channels; supported frontier/automaton MPO and symbolic/algebraic PEPO preparation retain separate capability and budget checks | [Compact channels](../api/operators/cluster_channels.md) |
| Sum-target FIT | Separate cached overlap environments contribute to one shared local update, including supported layered and native targets | [FIT sums](../api/fitting/local.md#fitting-a-sum-of-target-networks) |
| Dense direct MPO derivatives | Opt-in `mpo_factorization="projector"` with fixed numerical rank and resolved-chart checks; QR remains the default | [MPO replay](../api/optimizers/mpo.md), [paired derivatives](../api/boundary/metrics.md) |
| Host gradient solvers | Native Torch/JAX parameters with SciPy/NLopt host control and explicit budget/result validation | [Gradient solvers](../api/solvers/gradient.md) |

## Publication and validation

The [promotion handoff](../../history/2026-10-12-develop-main-promotion.md)
records integration into `main`. These commits are later development work,
not a new tagged release. The latest local smoke checkpoint passed **94 tests**;
it covers API/import/layout contracts and does not certify every numerical path.

| Dated check | Result and scope | Evidence |
| --- | --- | --- |
| 2026-10-10: paired MPO derivatives | 221 focused tests, including bounded CPU/CUDA derivative and cap checks | [MPO handoff](../../history/2026-10-10-mpo-paired-gradients.md) |
| 2026-10-10: PEPS sweeps | 614 package tests and 13 example tests; both comparison notebooks executed | [Sweep publication](../../history/2026-10-10-peps-sweep-publication.md) |
| 2026-10-10: sum FIT and traced MPS integration | 161 focused passes and one skip | [Workspace handoff](../../history/2026-10-10-workspace-publication.md) |

These selections overlap and use the devices/environments stated in their
records. The older 8,581-pass development run crossed a changing workspace;
it is not a full-suite result for the unchanged final commit. Documentation
cleanup adds no numerical or GPU validation.

## Limits and proposals

- A finite cluster cutoff or channel/rank cap defines an approximation.
  Backend and derivative agreement does not establish physical accuracy or
  a universally optimal representation.
- Projector derivatives require supported numerical charts. Rank changes,
  native compression and backend-specific execution retain their own contracts.
- Exact graph/channel state growth and PEPO contraction can still be expensive;
  preserve declared work, memory and precision budgets.
- Direct native fermionic 2D PEPO automata remain a
  [deferred proposal](plans/native_fermionic_pepo.md). Use the
  [planning index](plans/README.md) for other scoped designs.

For complete chronological measurements and earlier limitations, use the
[archived ledger](notes/2026-10-12-cluster-status-archive.md) and
[session journal](../../history/README.md).

## Historical section links

Earlier section anchors remain available below. Their full text belongs to
the [archived ledger](notes/2026-10-12-cluster-status-archive.md).
<a id="2026-09-28-local-readability-follow-up"></a>
<a id="algebraic-pauli-pepo-reduction-2026-09-29"></a>
<a id="automaton-plus-qr-notebook-comparison-2026-09-30"></a>
<a id="dense-zero-weight-simple-update-2026-10-05"></a>
<a id="direct-mpo-frontier-preparation-2026-09-29"></a>
<a id="exact-symbolic-mpo-sharing-2026-09-29"></a>
<a id="factor-exponential-reuse-and-graph-autodiff-2026-09-29"></a>
<a id="full-suite-compatibility-follow-up-2026-10-08"></a>
<a id="fused-compact-construction-2026-09-29"></a>
<a id="initial-compact-channel-replay-2026-09-29"></a>
<a id="integrated-cluster-and-peps-update-2026-09-28"></a>
<a id="integrated-cluster-mpo-automaton-plus-qr-api-2026-09-30"></a>
<a id="joint-notebook-solver-defaults-2026-10-06"></a>
<a id="joint-versus-single-exponential-review-2026-09-29"></a>
<a id="local-cluster-integration-2026-09-30"></a>
<a id="native-mpo-corrections-and-cuda-validation-2026-09-29"></a>
<a id="native-spin-mpo-autodiff-and-assembly-2026-09-29"></a>
<a id="numerical-mpo-delinearisation-2026-09-29"></a>
<a id="one-way-package-ownership-2026-09-29"></a>
<a id="pauli-mpo-automaton-reduction-2026-09-29"></a>
<a id="public-paired-factor-registration-2026-10-05"></a>
<a id="responsibility-and-supported-behavior"></a>
<a id="shared-api-review-corrections-2026-09-29"></a>
<a id="shared-cluster-api-alignment-2026-09-29"></a>
<a id="shared-interaction-graph-planner-2026-09-29"></a>
<a id="shared-square-plan-pepo-integration-2026-09-29"></a>
<a id="square-routing-frozen-compression-and-reports-2026-09-29"></a>
<a id="sum-target-fit-2026-10-09"></a>
<a id="symbolic-pepo-preparation-2026-09-29"></a>
<a id="trace-option-and-compact-plan-api-audit-2026-09-30"></a>
<a id="trace-the-constructed-pepo-2026-09-29"></a>
<a id="tracking-future-work"></a>
<a id="validation-and-publication-ledger"></a>
<a id="working-tree-56-cluster-construction-measurements-2026-09-28"></a>
<a id="working-tree-complete-graph-mpo-assembly-2026-09-28"></a>
<a id="working-tree-correctness-review-2026-09-28"></a>
<a id="working-tree-fixed-cluster-jit-gradients-2026-09-28"></a>
<a id="working-tree-fixed-construction-for-autodiff-2026-09-28"></a>
<a id="working-tree-fixed-mpo-bond-compression-benchmark-2026-09-28"></a>
<a id="working-tree-geometry-update-2026-09-28"></a>
<a id="working-tree-jaxtorch-host-solvers-2026-10-01"></a>
<a id="working-tree-joint-mpopepo-cluster-audit-2026-09-28"></a>
<a id="working-tree-spatial-reuse-2026-09-28"></a>
<a id="working-tree-trace-only-cluster-evaluation-2026-09-28"></a>

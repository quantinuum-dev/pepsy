# Cluster PEPO backend and downstream optimization status

Updated 2026-11-05 for the corrected six-site gate API on `develop`.
The earlier published `develop` → `main` merge was `bf83def` on 2026-10-12.
This page owns the current capability and validation routes. Detailed earlier
updates are preserved in the [archived ledger](notes/2026-10-12-cluster-status-archive.md).
Its older “working-tree” and “unpublished” statements retain their original dates.

## Current capabilities

| Area | Supported scope | Owning guide |
| --- | --- | --- |
| Package ownership | Pepsy owns graph/tensor-network construction, contraction and general solvers; Gaugy consumes public APIs for its objectives | [Package layout](package_layout.md#pepsy-and-downstream-packages) |
| Downstream MPS replay | Native mode discovery, named DMRG schedules and arbitrary-support sub-MPO fit windows; Gaugy preserves logical permutation indices and cached targets | [MPS guide](../api/optimizers/mps.md), [scoped alignment evidence](notes/2026-10-13-mps-mode-alignment.md) |
| Interaction clusters | Shared graph inventories, ordered product adapters, structural reuse and explicit construction budgets | [Interaction clusters](../api/operators/interaction_clusters.md) |
| Constructed PEPO traces | `trace_exp` traces the selected PEPO with its caps/compression; `partition_trace_exp` names the separate scalar shortcut | [Trace contract](../api/operators/cluster_expansion.md#trace-of-the-constructed-pepo) |
| Local cluster updates | Centered six-site loop-union patch, direct SU/BP-closed reduced environment, native dense ALS and local diagonal-gauge extraction; dense PEPO upper/lower streams | [BP cluster updates](../api/bp.md#centered-six-site-gate-updates) |
| Dense Torch cluster ALS | Opt-in certified Cholesky solves with spectral fallback, optional AOT capture, local fidelity records and per-pair stream scale bookkeeping | [Torch ALS and fidelity](../api/bp.md#dense-torch-als-and-local-fidelity) |
| Ordered PEPO residuals | `residual_exp` constructs the connected-cluster ordered product minus identity on a square OBC/PBC lattice, with an optional post-subtraction SVD bond cap | [Ordered products](../api/operators/cluster_expansion.md#ordered-pepo-products) |
| Native bosonic cluster MPOs | Exact Torch/JAX autodiff and sector-preserving NumPy/Torch compressed assembly within documented modes | [Native MPO scope](../api/operators/mpo_cluster.md#native-block-sparse-cluster-mpos) |
| Repeated cluster evaluation | Experimental compact channels; supported frontier/automaton MPO and symbolic/algebraic PEPO preparation retain separate capability and budget checks | [Compact channels](../api/operators/cluster_channels.md) |
| Sum-target FIT | Separate cached overlap environments contribute to one shared local update, including supported layered and native targets | [FIT sums](../api/fitting/local.md#fitting-a-sum-of-target-networks) |
| Dense direct MPO derivatives | Opt-in `mpo_factorization="projector"` with fixed numerical rank and resolved-chart checks; QR remains the default | [MPO replay](../api/optimizers/mpo.md), [paired derivatives](../api/boundary/metrics.md) |
| Host gradient solvers | Native Torch/JAX parameters with SciPy/NLopt host control and explicit budget/result validation; NLopt supports an opt-in maximum derivative threshold separate from step tolerance | [Gradient solvers](../api/solvers/gradient.md) |

## Publication and validation

The [Torch ALS follow-up](../../history/2026-11-05-torch-cluster-als.md)
implements guarded dense solves: three saved χ6 GPU pair
updates improved by 1.64–1.87× with matched local errors and stopping.
The implementation was published in Pepsy `6478f6b`; examples `a290a7c`
select it in the existing 3×2 PEPO benchmark. A subsequent review also
preserves gradients for raw Torch environments; its fresh checks are
recorded in the linked handoff.
The final follow-up selection passed **500 package tests** and **104 downstream
benchmark tests**. This is scoped validation, separate from the earlier
full-suite failure described below.
It adds optional local fidelity tracking and per-pair stream RMS bookkeeping.
Compilation is validated but did not consistently beat eager. This bounded
replay does not establish whole-trajectory speed or convergence; see the
[measurement record](notes/2026-11-05-torch-cluster-als.md).
The affected BP, API/layout and PEPS/PEPO full-update selection passed
**480 tests**, including CUDA. Full-suite fail-fast still stops at the
previously reproduced native fermionic exact-readout failure; the handoff
records that scope separately.
The second review fixed an AOT cache omission for autocast state/dtype;
**240 focused tests** then passed, including CUDA backward checks and
rectangular real/complex layouts. This is a later scoped check, separate
from the earlier 480-test integration selection.

The [local-cluster implementation audit](../../history/2026-11-05-loop-cluster-audit.md)
corrects half-weighted active-gauge extraction and order-insensitive tree
cache keys. Earlier long-time cluster trials used the incorrect gauge
spectra; preserve their numerical records without interpreting them as
validation of the corrected method.
The fixes are published on `develop` in `9e57d5e`. The corrected 23-qubit
homogeneous depth20 χ6 test gives **MPS-boundary cap18 cost 0.0516255525**,
versus 0.8648848491 at the same cap before the fix and matched-target HRS
0.0476403551. The downstream [current usage](https://github.com/rezaquant/pepsy_examples/blob/main/experiments/mps_magnetization/benchmark/eli_global_cost/RUNBOOK.md#current-six-site-cluster-update)
defaults to MPS scalar readout at 3χ; archived exact readouts remain historical
diagnostics. Public RMS-to-exponent bookkeeping keeps the operator amplitude
while bounding gauge scales. This is one measured case; finite-χ/cluster/ALS
convergence remains unverified.

The [Hermitian-only follow-up](../../history/2026-11-05-loop-cluster-hermitian-environment.md)
removes the diagnostic eigensolve when reduced builders receive
`psd_project=False`; the raw minimum eigenvalue is then absent. The downstream
six-site trial explicitly selects this mode. Public builder defaults retain
their existing PSD policy; long-time χ6 stability remains unverified.

The [CUDA-preservation audit](../../history/2026-11-05-loop-cluster-cuda-preservation.md)
fixes a silent Torch-to-NumPy conversion in the public cluster gate stream.
Earlier low-level CUDA checks remain valid, but the first downstream
GPU-labeled gate-stream prefix was CPU execution. New regressions assert
actual devices and the replacement trial uses a corrected snapshot.

The [centered-cluster readiness record](../../history/2026-11-05-centered-cluster-updates.md)
adds base-patch selection, one-sided dense PEPO gates and accelerator solves.
The χ6 interior-patch CPU/CUDA pilot agrees and remains finite; the corrected
full-T=6 case above provides a separate measurement. The earlier native reduced-shape failure
is fixed; five unrelated fermionic readout failures reproduce on the baseline.

The [November 5 local-gauge follow-up](../../history/2026-11-05-local-bond-svd-gauges.md)
replaces global SU regauging in BP reduced gate/compression updates with
local QR/LQ and SVD gauge extraction. Its CPU/CUDA checks are separate from
the 10×10 boundary-MPS FU campaigns and do not establish χ6 stability.

The [November 4 FU follow-up](../../history/2026-11-04-full-update-factor-reuse.md)
records SVD-factor reuse, small-core final balancing and native dense
local-solve dispatch, included with the local-cluster changes on `develop`.
The [publication review](../../history/2026-11-05-publication-review.md)
records fresh validation and baseline failures separately from the historical
downstream CPU16 performance measurements.

The [Torch scheduling follow-up](../../history/2026-10-30-prepared-su-scheduling.md)
batches compatible ready updates across independent schedules. The compiler
investigation isolates a backward-lowering failure; Inductor remains excluded.

The [October 30 prepared-SU follow-up](../../history/2026-10-30-prepared-simple-update.md)
implements dense Torch batching with per-member rank checks
and adaptive replay. AOT-eager is optional; Inductor failed complex-gradient
checks and is excluded. This supersedes the earlier proposed-only SU status.

The [October 30 QR performance follow-up](../../history/2026-10-30-pepo-qr-reuse.md)
records a saved-factor backward optimization, focused
CPU/CUDA derivative checks, and a matched finite-PBC Gaugy benchmark. It also
records three downstream native-Z2 failures reproduced with unchanged QR.
The separate CTMRG check preserves the finite torus and uses the public
projector compatibility context. The newer prepared-SU entry above records
the subsequent implementation and its limits.

The [publication batch](../../history/2026-10-30-pepo-acceleration-publication.md)
packages both changes and records fresh validation of the selected files.
The earlier records retain their original uncommitted baseline status.

The [October 21 NLopt review](../../history/2026-10-21-nlopt-gradient-tolerance.md#review-and-local-commit)
adds explicit derivative stopping and preserves the stopping trial when best-state
restoration is disabled. Its focused CPU validation and local commit status are
separate from earlier published work.

The [October 21 numerical follow-up](../../history/2026-10-21-qr-periodic-boundary-fixes.md)
repairs JAX QR derivatives, periodic direct boundary dispatch, backend-native
VMC normalization, and deterministic product-state initialization. It records
the full collection and focused revalidation, hardware exclusions, and local
commit state separately from the earlier promotion below. Downstream Gaugy
compressed-trace failures remain unresolved.

The [promotion handoff](../../history/2026-10-12-develop-main-promotion.md)
records integration into `main`. These commits are later development work,
not a new tagged release. That promotion's smoke checkpoint passed **94 tests**;
the November 5 publication review records **112** newer smoke passes.
These cover API/import/layout contracts and do not certify every numerical path.

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

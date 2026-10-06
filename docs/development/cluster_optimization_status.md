# Cluster PEPO backend and downstream optimization status

Updated 2026-10-01. An earlier Pepsy `develop` publication at
**`b4c4631`** includes
the backend correction `cf1d84c` and the newer remote work merged at
`a13031b`. Gaugy's matching API/refinement is published at **`a7af793`**.
These are development commits, not a new tagged package release.

## Public paired-factor registration, 2026-10-05

The existing paired-factor primitive is now exposed as
`pepsy.backends.register_projector_split` for downstream cube algorithms.
This adds a public lazy export, not a change to Pepsy's decompositions or
boundary defaults. See the [export validation](../../history/2026-10-05-projector-export.md)
and [factorization contract](../api/boundary/metrics.md#differentiating-rank-deficient-boundary-mps-contractions).

## Dense zero-weight simple update, 2026-10-05

The public `gate_simple` adapter projects exactly zero external weights onto
their support during unsmeared dense updates, preserving the represented
operator and restoring the original external gauge vectors. Gaugy's SU
builder and loss now use this public adapter. This does not extend native
Symmray or rank-changing derivative support. See the
[gate contract](../api/operators/gates.md#gauge-scale-extraction) and
[validation record](../../history/2026-10-05-zero-weight-simple-update.md).

## Working-tree JAX/Torch host solvers, 2026-10-01

Pepsy's `GradientOptimizer` now accepts native JAX or Torch parameters for
SciPy L-BFGS-B (`lbfgs` / `scipy-lbfgs`) and NLopt `LD_LBFGS`. Loss/gradients
stay on the selected device; the host solver exchanges a packed iterate and
derivatives. The joint Gaugy notebook exposes these choices for local/global
costs. Validation: 116 Pepsy solver/API/import checks, 28 downstream CPU checks
covering both backends and all three trace closures, and ten bounded CUDA
Run All notebooks passed, including the saved L=12 JAX settings with a two-step
budget for both solvers. Working-tree edits only; no full-suite or publication
claim. See the [solver API](../api/solvers/gradient.md) and
[handoff](../../history/2026-10-01-jax-host-solvers.md).

## Local cluster integration, 2026-09-30

The user approved committing all pending Pepsy work and merging `develop`
into local `main`, with no push and a return to `develop`. The approved
package commit contains the implementations described below and their
validation records; earlier working-tree descriptions are dated evidence.
Pre-commit validation: 682 tests passed and eight default-precision JAX GPU
checks failed; all 24 cases in the affected modules pass with explicit
highest matrix-multiplication precision. Ruff, whitespace and catalog checks
passed. No full-suite or remote-publication claim follows. See the
[local integration handoff](../../history/2026-09-30-local-cluster-integration.md)
and Git history for the content and merge commits.

## Integrated cluster-MPO automaton plus QR API, 2026-09-30

Pepsy's working-tree `exp_mpo_cluster` and `exp_mpo_cluster_product` now
accept `preparation="automaton"`/`"frontier"` and `delinearize=True` with
fixed uncapped construction. Channel-plan `exp` and `trace_exp` expose the
same explicit NumPy QR step. Reports distinguish exact channel maps from
numerically reduced output dimensions; exact sharing is no longer mislabeled
as reference QR projection. The final affected selection passed 297 tests;
Ruff and whitespace checks passed. This is uncommitted package integration,
following the earlier notebook-only comparison below, not a new full-suite
or publication result. See the [API and validation record](notes/2026-09-30-automaton-delinearisation-api.md).

## Trace option and compact-plan API audit, 2026-09-30

Constructed traces now reject a custom sparse `state_budget` when
`contract_opts` requests materialized contraction, rather than silently
ignoring the budget. Contraction options must be mappings. Compact-plan reports
now distinguish exact structural methods from reference QR projection and
count numerical residual evaluations. Symbolic and algebraic preparation skip
those evaluations when the cap is absent or cannot bind. Exact numerical-layout
methods retain their required base evaluation but skip optional samples and
tangents in the same case. The focused operator,
channel, algebra and PEPO suite passes 104 tests. See the
[handoff](../../history/2026-09-30-trace-channel-api-review.md).

## Automaton plus QR notebook comparison, 2026-09-30

The single-exponential `cluster_1d` notebook now compares automaton plus
numerical delinearisation against frontier plus delinearisation using the
same settings. Both full OBC/PBC notebook runs passed, along with 27 focused
delinearisation tests. At p=5 OBC, automaton's largest bond drops from 209
to 50, matching frontier + QR. This reuses existing public APIs; package
algorithms and the joint notebook are unchanged. Working-tree edits only;
these checks do not replace earlier full-suite results. See the
[measurements and scope](notes/2026-09-30-automaton-delinearisation-example.md).

## Pauli MPO automaton reduction, 2026-09-29

The working tree adds `preparation="automaton"` for dense qubit Pauli cluster
MPOs. It combines the existing frontier construction with certified Pauli
spans and rational state elimination. With full channels, preparation and
replay need no SVD/QR and do not enumerate complete cluster collections.
Optional caps retain reference QR; native sectors retain frontier support.
This is exact for the declared cluster family, not global minimization.

The 1D notebooks now compare automaton, frontier, numerical delinearisation,
fixed and SVD construction for one selected boundary. Operator algebra can
reduce the states of constrained models; the NN + NNN examples need not gain
further bond reduction. See the [API](../api/operators/cluster_channels.md),
[measurements and checks](notes/2026-09-29-mpo-automaton.md) and
[handoff](../../history/2026-09-29-mpo-automaton.md).

## Algebraic Pauli PEPO reduction, 2026-09-29

The local working tree adds opt-in `preparation="algebraic"` for fixed
graph/square Pauli PEPOs. It proves a closed Pauli span per cluster, then
removes exact rational dependencies between complete edge slices before
and after routing. Full channels use neither SVD nor QR, including during
preparation. Replay uses fixed Walsh projections and contractions; optional
smaller caps retain reference QR outside autodiff. No live coefficient is
used to prove an identity or choose a structural rank.

Measured exact loop bonds shrink 9→7 and a higher-body routed bond 25→13
relative to identical-slice sharing. Branch/crossing probes gain no further
chi reduction; algebraic preparation and some replay times are slower. This
is a local reduction, not global minimality or a universal speedup. Dense
cluster exponentials remain. See the [API](../api/operators/cluster_channels.md),
[measurements](notes/2026-09-29-algebraic-pepo-channels.md) and
[handoff](../../history/2026-09-29-algebraic-pepo-channels.md).

## Numerical MPO delinearisation, 2026-09-29

The local working tree adds `delinearize_mpo`, an opt-in NumPy column/transfer
compression using pivoted QR without SVD. It acts on the frontier MPO already
constructed, preserves the requested cluster target to checked numerical
accuracy, and reports local tolerances separately from cluster error. The 1D
notebooks compare fixed, frontier, delinearised frontier and SVD construction.
This is not symbolic parameter-family minimization or an autodiff path; native
arrays are rejected. See the [API](../api/operators/mpo_delinearize.md) and
[validation record](notes/2026-09-29-mpo-delinearisation.md).

## Symbolic PEPO preparation, 2026-09-29

The local working tree adds `preparation="symbolic"` for fixed graph/square
PEPOs. It shares parameter-independent graph-edge slices before routing,
then square-edge slices, and reuses linear reference templates. Remaining
routed blocks are enumerated once. Full channels preserve the chosen cluster
target; capped approximations depend on the new gauge and can have larger
errors. Replay remains fused with no SVD/QR. Gaugy only forwards options.

Branching and higher-body probes show substantial preparation memory
reductions; independent crossing wires do not share and gains are mixed.
This is conservative exact reduction, not global minimality or a guarantee
of small exact chi. See the [API](../api/operators/cluster_channels.md),
[measurements and checks](notes/2026-09-29-symbolic-pepo-channels.md) and
[handoff](../../history/2026-09-29-symbolic-pepo-channels.md).

## Direct MPO frontier preparation, 2026-09-29

`prepare_cluster_channels(..., preparation="frontier")` constructs exact MPO
transitions by retaining only unfinished clusters across each cut. Complete
collections are never enumerated, including during reference preparation.
The path supports fixed interval/exact graph and uncapped recursive sources,
crossing and gapped supports, and native bosonic sectors. It rejects bounded
or auto graph targets; PEPOs use the separate modes above. Active-state and memory
budgets raise rather than silently truncate. Optional chi projection is still
approximate, while full channels reproduce the selected cluster expansion.

A 40-site disjoint-crossing probe represents 1,048,575 nonempty collections
with four active sets per cut. This does not establish global minimality or
small ranks for general graphs. See the [API](../api/operators/cluster_channels.md),
[measurement and remaining algebraic work](notes/2026-09-29-frontier-cluster-channels.md)
and [handoff](../../history/2026-09-29-frontier-cluster-channels.md).

## Exact symbolic MPO sharing, 2026-09-29

Compact MPO preparation now merges formally equal prefixes/continuations
within charge sectors before numerical QR selection. Independent residual
atoms prevent reference-value equality from introducing invalid identities.
Separate sum/select endpoint maps preserve multiplicity; replay remains fused
and uses no SVD/QR. The opt-out is `structural_reuse=False`. Graph/square PEPOs
retain their existing connectivity-aware construction.

A fixed-order chi sweep records operator/gradient errors, forward/backward
times and Torch tensor-allocation peaks. Full structural dimensions recover
the uncompressed family; tighter caps remain approximations. Preparation still
enumerates topology, and the conservative quotient is not a globally minimal
automaton. See the [API](../api/operators/cluster_channels.md),
[measurement](notes/2026-09-29-symbolic-cluster-channels.md) and
[handoff](../../history/2026-09-29-symbolic-cluster-channels.md).

## Fused compact construction, 2026-09-29

The working tree now assembles compact tensors directly from local residual
factors. Replay avoids sparse history buffers, unprojected crossing MPO
products and routed square wire products. The separate reference preparation
still enumerates channels. Fixed recursive MPO sources are accepted when
their exact reference topology fits the source collection budget. This is
not full symbolic minimization or an unrestricted compact recursive planner.
Square channel preparation uses graph residuals, preserving the unprojected
target while changing the virtual gauge of newly prepared capped plans.

The residual-to-array assembly boundary is `pack_residuals` /
`bind_assembler`; legacy block packing remains diagnostic. Gaugy's adapter
requires no algorithm changes. See the [current API](../api/operators/cluster_channels.md),
[new measurements and limits](notes/2026-09-29-fused-cluster-channels.md), and
[validation handoff](../../history/2026-09-29-fused-cluster-channels.md).

## Initial compact channel replay, 2026-09-29

The local working tree adds experimental `prepare_cluster_channels` for
fixed direct MPOs and fixed graph/square PEPOs, including native MPO sectors.
Host QR selects fixed bases from explicit reference and tangent snapshots;
replay constructs compact tensors from sparse blocks without SVD/QR. It
supports actual projected-operator traces and an array-only compilation
boundary. Graph materialization also removes global edge padding exactly.

This is a bounded first implementation, not a complete symbolic history
minimizer or environment-optimal compressor. Dense expanded virtual tensors
are avoided; sparse histories and local residuals still grow. Small-case
measurements show storage reductions but mixed timing gains, with measurable
operator/gradient errors at tight caps. See the
[API](../api/operators/cluster_channels.md),
[literature and evidence](notes/2026-09-29-compact-cluster-channels.md), and
[validation handoff](../../history/2026-09-29-compact-cluster-channels.md).

## Native spin MPO autodiff and assembly, 2026-09-29

The local follow-up adds exact structural sector factors and native
materialization on Torch/JAX, including zero-parameter gradients. Direct
and uncapped recursive construction support JAX tracing. Native recursive
and streaming assembly now support NumPy/Torch sector compression; Torch
uses paired-factor projector derivatives with numerical rank selection.
Public semantic adaptive compression preserves sectors and index conventions;
fixed-rank compression rejects an unspecified sector allocation.

This supersedes the native backend/assembly restrictions in the earlier dated
entries below. User scope is Pauli/spin only. JAX adaptive native compression,
derivatives across rank changes, and large-lattice performance are not claimed.
Tensor-network implementation remains entirely in Pepsy. The broad selection
passed 555 tests and Gaugy's existing downstream selection passed 137; see the
[final checks and limits](notes/2026-09-29-native-cluster-autodiff.md) and
[handoff](../../history/2026-09-29-native-cluster-autodiff.md) for final focused
rechecks and working-tree publication status.

## Native MPO corrections and CUDA validation, 2026-09-29

The remaining conserved-hopping and zero-cutoff native MPO failures are fixed
in the local working tree. Residual SVDs preserve charge sectors, assembled
virtual channels carry inferred flux, and the cutoff/cap selects from the
combined spectrum. A further repeated-physical-charge dense-export ordering
error is fixed. NumPy direct assembly is checked for U1, Z2, U1U1 and Z2Z2,
including graph gaps and crossing/nested collections.

Joint dense MPO and graph/square PEPO values and all tested parameter
gradients now pass bounded CUDA checks; Gaugy also checks frozen-compression
trace gradients against finite differences. This validates correctness on
the tested device, not large-lattice performance. Native cluster autodiff,
native streaming/recursive assembly, fermionic histories and derivatives
through changing retained ranks remain outside the supported contract.
See the [evidence](notes/2026-09-29-native-cluster-charges.md) and
[handoff](../../history/2026-09-29-native-cluster-charges.md).

## Joint versus single exponential review, 2026-09-29

The dense joint MPO/PEPO/Gaugy workflows now have additional independent
union-geometry, finite-cutoff, single-factor-limit and live-gradient checks.
Unused Quimb compression options are rejected before joint PEPO evaluation.
Gaugy graph factories expose Pepsy's spatial reuse switch. The review found
native U(1) MPO hopping and zero-cutoff null-channel failures in both single
and joint construction, now corrected by the scoped follow-up above.
See the [capability review](notes/2026-09-29-joint-product-parity.md) and
[validation handoff](../../history/2026-09-29-joint-product-parity.md).

## Trace the constructed PEPO, 2026-09-29

Local PEPO `trace_exp` now constructs and traces the selected operator.
`trace_pepo`/active `.trace()` contract its actual physical blocks and virtual
bonds, preserving Torch/JAX gradients and requested compression. The old
scalar shortcut is explicit as `partition_trace_exp`; Gaugy exposes the
corresponding bound `partition_trace`. Short periodic axes now use finite
located construction even for uniform terms, fixing site/channel aliasing
and parallel-bond counting hidden by the old scalar-only trace.

See the [API contract](../api/operators/cluster_expansion.md#trace-of-the-constructed-pepo)
and [handoff](../../history/2026-09-29-pepo-operator-trace.md). This supersedes
older trace-only statements below; MPO scalar trace semantics are unchanged.
Changes remain local. Sparse trace budgets bound work, and large exact PEPO
contractions can still be expensive.

## One-way package ownership, 2026-09-29

Pepsy owns the independent graph/tensor-network implementation; Gaugy owns
its Pauli expansion and domain objectives as a downstream public-API client.
The runtime code already follows this direction. The Gaugy-importing graph
handoff regression has moved into Gaugy, and a fresh-interpreter Pepsy test
blocks Gaugy imports while exercising graph/square materialization,
compression and reports. See [ownership](package_layout.md#pepsy-and-downstream-packages)
and the [validation record](../../history/2026-09-29-package-ownership.md).

## Square routing, frozen compression and reports, 2026-09-29

The local working tree now implements explicit square virtual routing for
diagonal/NNN/higher-body dense interactions, leaving the union interaction
plan and cluster inventory intact. Reference tree subspaces provide explicit
smaller differentiable materializations; ranks and reference bases remain
fixed during replay. Ordered builder reports count actual exponentials,
whole-product reuse and lower contractions. Gaugy exposes the same controls.

These changes remain unpublished Pepsy edits. The
[API contract](../api/operators/interaction_clusters.md#explicit-differentiable-compression)
and [handoff](../../history/2026-09-29-square-routing-compression-reports.md)
describe validation and limits. Rank-changing autodiff, global optimal
compression, native symmetry routing and GPU performance remain unverified
or unsupported as specified there. Earlier statements below about missing
diagonal routing describe their dated baseline.

## Factor exponential reuse and graph autodiff, 2026-09-29

The local working tree adds per-factor exponential reuse to square and graph
ordered products, plus Torch/JAX graph PEPO materialization. Numerical caches
remain local to an evaluation; independent vector slots and opaque callbacks
keep their binding semantics. Square batches now pool representatives across
factors as well as clusters. Graph fixed splits preserve zero derivatives;
uncapped tensor-backend auto construction uses the same fixed route.

These changes are not yet published. See the
[API contract](../api/operators/interaction_clusters.md#representation-boundaries),
[implementation and validation handoff](../../history/2026-09-29-factor-reuse-graph-autodiff.md),
and [compatibility evidence](notes/2026-09-29-factor-reuse-graph-autodiff.md).
Rank-capped SVD derivatives, large fixed-channel memory costs, and GPU
performance remain separate limits.

## Shared square-plan PEPO integration, 2026-09-29

Shared-plan PEPO factories now select the existing 2D square builder for
complete NN square graphs with fixed Pauli product terms. Selection is
explicitly controllable through `layout`; other supported graphs retain the
generic builder. Parameter identities, factor order, periodic multiplicities
and the shared cluster inventory are preserved. Gaugy's matching conversion
uses the same selection. Diagonal/NNN routing remains a separate next step.
See the [API](../api/operators/interaction_clusters.md#shared-square-plans-and-2d-pepo-construction)
and [validation/publication record](../../history/2026-09-29-square-cluster-plan.md).

## Shared interaction-graph planner, 2026-09-29

The publication after `a233a40` includes lazy `pepsy.operators.ClusterPlan`, exact
finite inventories and counts, verified symmetry candidates, bounded structural
caches, common `from_plan` adapters, and general ordered graph-PEPO products.
Gaugy's matching implementation adds automatic local Pauli equivalence and
shared graph factories. It is published as `b6cfe55` on Gaugy develop,
with 255 focused tests passing. Graph-PEPO matrix orientation/dtype and mixed backend
residuals were corrected during cross-representation validation.

See the [API and limits](../api/operators/interaction_clusters.md) and
[implementation/validation record](../../history/2026-09-29-interaction-cluster-plan.md).
The planner and API alignment are included in the publication containing the
[final review record](../../history/2026-09-29-cluster-final-review-publication.md).
That review also corrected legacy graph-MPO runtime rebinding, Torch graph
block metadata/isolated-site materialization, and invalid term-index errors.
Unrelated MPS working-tree changes are outside this publication.

## Shared API review corrections, 2026-09-29

A second review reproduced and fixed dense-plan dataclass replacement with
`order`, MPO partial-default coefficient vectors, and early PEPO runtime
binding conflict validation. These fixes are included in the shared-planner
publication above. See the
[review and validation record](../../history/2026-09-29-cluster-api-review.md).

## Shared cluster API alignment, 2026-09-29

The shared-planner publication on `develop` after `a233a40` aligns the main dense/Pauli
PEPO, MPO and Gaugy Pauli entry points. PEPO exposes the spatial
`cluster_size` alias; dense plans add `compile_exp().exp(step)` with preserved
`build(beta)` behavior. MPO adds `from_bases`, runtime term vectors,
positional parameter bindings and explicit materialization. Numerical
cluster algorithms and existing defaults remain unchanged.

See the [common API table](../api/operators/exponentials.md#shared-connected-cluster-calls)
and [validation handoff](../../history/2026-09-29-cluster-api-alignment.md).
These Pepsy changes are included in the publication above; the September 28
records below describe earlier commits. Gaugy's cutoff alias continues to
work with the established public Pepsy constructor keyword.

## Responsibility and supported behavior

| Owner | Implemented responsibility | Guide |
| --- | --- | --- |
| Pepsy cluster backend | Complete ordered exponential products, local residual assembly, fixed history factors, physical trace before dense allocation | [Cluster module map](modules/cluster_expansion.md) |
| Pepsy boundary/backend | Opt-in `contract_flat(..., method='mps', mps_factorization='projector')`, composed paired-factor first-order Torch VJP | [Boundary API](../api/boundary/metrics.md#differentiating-rank-deficient-boundary-mps-contractions) |
| Gaugy | Local objective, circuit/ITF gauge parameterization, parameter metadata, scalar Pauli engine, shared optimizer and validation acceptance | [Gaugy common API](https://github.com/rezaquant/gaugy/blob/develop/learning/cluster_optimization_api.md) |

The public Pepsy factorization default remains `"qr"`; Gaugy's cluster MPS
companion explicitly selects `"projector"`. The latter applies at every
canonicalization, reduction and compression stage, without replacing global
QR/SVD drivers. It requires dense NumPy/Torch data, first derivatives,
gauge-invariant paired factors, locally fixed rank, and a resolved truncation
gap. It does not extend automatically to SU, CTMRG, JAX, native Symmray,
fermionic traces or higher derivatives. See the
[derivation and rejection behavior](notes/2026-09-26-projector-boundary-gradients.md).

The ordered cluster product is formed before connected residual subtraction.
Spatial cluster order, internal history rank, and boundary chi are distinct
approximations. Structural trace pruning removes only certified forbidden
sectors. Numerical null removal during boundary factorization is a separate
operation with a local-rank derivative contract.

## Validation and publication ledger

| Baseline/scope | Recorded result |
| --- | --- |
| Backend correction before remote integration | Full Pepsy suite: 4,789 passed, 121 skipped; 135 focused checks; Ruff passed |
| Merged dependency at `b4c4631` | 136 projector/flat-backend/public-API/layout/Torch-SVD checks passed; full Ruff passed |
| Downstream API refinement with merged dependency | 28 Gaugy API/PEPO checks passed |
| Earlier downstream shared API suite | 547 passed, followed by 14 focused API checks including a newly added case |
| Downstream 4×4 OBC t1/depth10 study | Both engines reduced common exact-order3 validation cost from 2.0611e-5 to about 1.284e-5; final directional errors <4.5e-12 Pauli / <1.6e-10 PEPO |

The full-suite result predates the remote integration; the 136 merge checks
are not full validation of all newly fetched sampling/qMERA/MPS work.
The downstream reference is an exact contraction of a finite-cluster PEPO,
not the exact global evolution operator. Acceptance guarantees only its
fixed deterministic measured cost, not global fidelity or cutoff convergence.

Detailed records:

- [Backend correction](https://github.com/quantinuum-dev/pepsy/blob/develop/history/2026-09-26-projector-boundary-gradients.md)
- [Dependency merge and successful publication](https://github.com/quantinuum-dev/pepsy/blob/develop/history/2026-09-26-cluster-api-dependency-sync.md)
- [Current downstream ledger](https://github.com/rezaquant/gaugy/blob/develop/DEVELOPMENT_STATUS.md)


## Integrated cluster and PEPS update, 2026-09-28

Pepsy `develop` commit **`8673ef7`** publishes the cluster MPO/PEPO work
and PEPS optimizer/sampler corrections described in the dated sections below.
Those sections retain their original "working-tree" and "uncommitted"
wording as historical evidence from baseline `4e398e4`. The commit was
rebased over remote `89e6d29`; the two overlapping documentation entries
were combined, and numerical source merged automatically.

On the integrated tree, the affected cluster/PEPS/API/layout selection
passed **547 tests, 2 skipped**. The full CPU suite passed **5,313 tests,
105 skipped** with 684 warnings; full Ruff and whitespace checks passed.
The [publication handoff](../../history/2026-09-28-cluster-publication.md)
records the synchronization and validation. These are new integrated-tree
results; earlier full-suite counts below refer to their own baselines.

## 2026-09-28 local readability follow-up

The dense square `ClusterExpansionPlan.build` now delegates cluster families
and report assembly to private methods, preserving a shared sector allocator
and the original subtraction order. This is included in the local readability
and diagnostics commit on `develop`, based on `f410fa0`, alongside FIT/BP/Torch
VMC/MPO edits. It does not change the published backend or downstream status
above. The [commit handoff](../../history/2026-09-28-readability-commit.md)
records the combined validation and remaining limits; the
[four-domain handoff](../../history/2026-09-28-bp-vmc-mpo-pepo-readability.md)
retains the earlier focused evidence.
## Working-tree trace-only cluster evaluation, 2026-09-28

A further uncommitted Pepsy update on baseline `4e398e4` adds
`trace_exp` to compiled MPO and Pauli PEPO cluster products and single
Pauli PEPO bases. It computes exact local ordered-product traces, scalar
connected residuals and the complete selected-order collection sum.
No MPO/PEPO assembly or rank compression is involved. This trace is
different from the trace of a rank-truncated or collection-bounded
representation. A state budget raises before omitting collections.
The [trace-only evidence](notes/2026-09-28-cluster-trace-only.md) records
the new checks and 5×6 order-four CPU measurements; no published commit
or full-suite result is claimed for this update.

## Working-tree geometry update, 2026-09-28

On `develop` baseline `4e398e4`, an uncommitted update adds per-size oriented
and C4 tree/loop inventories and bounded reuse of immutable shape levels and
finite translated embeddings. This does not change the numerical residual
construction. New cluster/API/layout validation: **120 passed**, full Ruff and
whitespace checks passed; no new full-suite result. Geometry-only benchmarks
and scope are recorded in the
[dated evidence](notes/2026-09-28-cluster-geometry-cache.md). These edits are
not part of the published commits listed above.

A subsequent uncommitted compile-path review prepares located static maps and
tree topology before evaluation, shares local maps across placements, prepares
higher-order homogeneous source maps, and fixes mixed-product route selection.
New validation: **122 cluster/API/layout tests passed**, including repeated
value/gradient references; full Ruff and whitespace checks passed. The
[compile review](notes/2026-09-28-cluster-compile-review.md) records static
preparation time and memory reductions and their limits. No new full-suite or
end-to-end performance claim is made.

The subsequent compiled-API review adds direct report access and optional
paired returns to cluster MPO calls, stable compiled-callable reuse, and direct
PEPO shape/cache inspection. New validation: **160 MPO/PEPO cluster/API/layout
tests passed**, Ruff and whitespace checks passed, and the documented 2D
compiled example ran successfully. These remain uncommitted working-tree
changes; see the [API review evidence](notes/2026-09-28-cluster-api-review.md).

## Working-tree spatial reuse, 2026-09-28

A further uncommitted update on baseline `4e398e4` enables Hamiltonian-aware
local target reuse in cluster MPOs and `PauliPEPOBasis`, controlled by
`spatial_reuse=True`. It retains coefficient identities, directed/parallel
bond occurrences, factor order, residual subtraction and all placements.
The 5x6 open uniform X + ZZ example at p=4 has 492 placed targets and six
local representatives. This does not change graph collection approximations.
New focused validation: **175 passed**, two existing deprecation warnings.
The new full CPU-only gate passed: **5199 passed, 105 skipped**, with 777
warnings (26:29). Full Ruff and whitespace checks passed. The interrupted
GPU gate/resource limitation and exact validation scope are recorded in the
[dated evidence](notes/2026-09-28-cluster-spatial-reuse.md), together with scoped
timings and backend limits. This work is not part of the published ledger above.

## Working-tree complete graph MPO assembly, 2026-09-28

A further uncommitted change on baseline `4e398e4` adds
`assembly="recursive"`: shared remaining-site MPO subproblems include all
compatible disjoint residual collections, with optional per-batch compression.
`assembly_state_budget` guards structural work and raises without dropping
collections; collection_budget is unused in this mode. Streaming direct plans
now preserve separated residual products, and assembly truncation prepares an
orthonormal environment. Reports expose subproblem/cache counts and rank
reductions. No native sector/fermionic recursive support is claimed.

New CPU-only validation: **270 passed**, two existing deprecation warnings,
full Ruff and whitespace checks passed. This is a focused MPO/API/layout gate;
the earlier full-suite result above predates these edits. Measurements include
a 5x6 p=2 chi=4 assembly in 8.31 s (65,805,402 collections, 703 subproblems)
and a compile-only 5x6 p=4 plan (249,479,463,512 collections, 33,514 subproblems).
See [construction, checks and limits](notes/2026-09-28-mpo-recursive-assembly.md).
Nothing from this update has been committed or published.

## Working-tree fixed construction for autodiff, 2026-09-28

A further uncommitted update on baseline `4e398e4` adds
`factorization="fixed"` to cluster MPOs and Pauli PEPOs. Exact shape-based
splits avoid SVD and preserve zero-residual gradients; the default `"auto"`
retains numerical factorization/compression policies. Generic PEPO tree
structure is cached. Supplied finite-lattice `spatial_symmetries` are validated
against graph multiplicity, operators and coefficient identities; automatic
local translation/rotation/reflection reuse remains term-aware. Fixed mode
rejects internal truncation and unsupported native MPO conversion.

The [implementation and measurements](notes/2026-09-28-fixed-cluster-autodiff.md)
link the owning API guides and record compiler/backend limits. The
[current handoff](../../history/2026-09-28-fixed-cluster-autodiff.md) records
validation: **5241 passed, 105 skipped** in the full CPU gate, followed by
**277 focused MPO/API/layout checks** covering the final narrow host-step and
dense-term parsing fixes. Ruff, whitespace and documentation link checks pass.
The full gate started before those last fixes and the separate compiler test
file; it is not an all-files-at-final-HEAD claim. Local Torch
full-graph and JAX JIT checks do not establish whole-builder machine-code
compilation. Exact global bonds and contraction can still be expensive.
Nothing from this update has been committed or published.

## Working-tree correctness review, 2026-09-28

The subsequent [correctness review](notes/2026-09-28-cluster-correctness-review.md)
found and fixed a cluster MPO backend mismatch for positional tensor parameters
and callable coefficients with host-valued steps, and enabled declared
`MPOParameter` defaults. Evaluated fixed-mode targets are aligned before
subtraction, without additional callback evaluations. PEPO numerical code is
unchanged. An independent square-lattice set-partition reference checks orders
2, 3 and 4 for both MPO/PEPO with and without symmetry reuse. New validation: **374 focused domain/API/layout checks passed**, and the
expanded new review file passed **15 tests**. Ruff, whitespace and relative
documentation link checks passed. Exact scope is
recorded in the [review handoff](../../history/2026-09-28-cluster-correctness-review.md);
the earlier full-suite counts above predate this narrow fix. No commits or
publication.

## Working-tree fixed MPO bond compression benchmark, 2026-09-28

An [explicit post-compression benchmark](notes/2026-09-28-cluster-bond-compression.md)
measures time, final bonds and Frobenius error against an uncompressed fixed
p-cluster MPO. A dense MPO error-estimation issue at small truncation errors
was fixed by canonicalizing the difference before its norm contraction.
The [handoff](../../history/2026-09-28-cluster-bond-compression.md) records
**274 affected checks passed**, followed by **22 compression/sector checks**
after the final dense-data guard. Full Ruff, whitespace and affected link
checks passed. The earlier full-suite result predates this change.
Post-compression still requires the full exact MPO to be constructed first;
internal numerical assembly compression remains a separate policy. No
commit or publication was made.

## Tracking future work

The additional 2026-09-28 local maintenance batch extracts the generic dense
residual fit from cluster assembly, retaining rank schedules, seeds, warm
starts, and lower-order subtraction. The
[maintenance handoff](../../history/2026-09-28-remaining-maintenance.md) records
validation and publication for this follow-up; it does not extend the native
fermion or derivative support described above.

Keep API behavior in the owning guide, implementation ownership in module
maps, derivations/benchmarks in dated notes, and session/publication events
in history. Historical "no push" or "unfixed" statements retain their
original baseline; use this ledger and Git for current status. Update these
links and measured scope when the implementation changes, without rewriting
old evidence into a claim of current validation. Whole-cost PEPO compilation,
general rank-changing derivatives and native cluster-symmetry optimization
remain deferred; they are not features of this published correction.

## Working-tree fixed cluster JIT gradients, 2026-09-28

Complete three-site order-three and 2×2 square order-four fixed MPO/PEPO
scalar losses pass JAX `jit(value_and_grad)` against independent dense
exponentials, including zero coefficient and zero time. Two-site complex-time
cases pass analytic checks.
Torch `compile(backend="aot_eager", fullgraph=True)` captures local
exponential/identity/fixed-split and PEPO tree-factorization forward and
backward with correct gradients. Narrow native Torch tensor factories and
Python shape products avoid tracing failures.
The complete Torch MPO/PEPO builder remains eager: full-graph probes fail in
Python/Autoray/Quimb assembly, and partial-graph compile also failed in this
installed stack. No end-to-end Torch graph or performance claim is made.

The [JIT evidence](notes/2026-09-28-cluster-jit-gradients.md) and
[handoff](../../history/2026-09-28-cluster-jit-gradients.md) record scope and
limits. The final affected cluster gate passed 183 tests on CPU, including all ten
new JIT cases.
The preceding full suite predates this update. No commit or publication.

## Working-tree 5×6 cluster construction measurements, 2026-09-28

The [PEPO stage profile](notes/2026-09-28-pepo-stage-profile.md) measures
the previously unmeasured lower-residual contraction, Pauli expansion and
sparse block insertion on the uniform 5×6 order-four model. Reusing verified
symmetry-equivalent lower contractions reduced their calls from 462 to five
and their median time from 0.693 s to 0.014 s. The complete PEPO evaluation
fell from 1.313 s to 0.550 s, with fresh-process peak RSS about 0.223 GiB.
All 28,470 active sparse blocks agreed with the no-reuse reference to
3.19e-16 maximum entry difference. These are CPU-only scoped observations.

The [graph MPO numerical record](notes/2026-09-28-graph-mpo-5x6-numerical.md)
extends the 33,514-state order-four plan to actual numerical assembly and
compares selected matrix elements with an independent, complete order-four
collection sum. It separates collection completeness from compression error.
With batch four, chi-1/chi-2 assembly took 486.314/1237.625 s and peaked
at 0.810/1.017 GiB RSS. The two measured relative errors were
44.3%/80.7% at chi one and 13.6%/42.9% at chi two: neither cap is accurate
enough for these elements, and global error remains unverified. The focused
CPU domain/API/layout gate passed **386 tests**, with full Ruff and whitespace
checks. The final result and validation are recorded in the
[handoff](../../history/2026-09-28-cluster-cost-closure.md). This work remains
uncommitted and unpublished; earlier full-suite checks predate it.

## Working-tree joint MPO/PEPO cluster audit, 2026-09-28

The [joint implementation audit](notes/2026-09-28-joint-cluster-audit.md)
checks two noncommuting ordered factors on a 2×2 square against an
independent set-partition reference at orders two through four, for both
MPO and PEPO with spatial reuse on/off. Complete fixed joint two-site
MPO/PEPO JAX JIT values and coefficient/step gradients match dense
ordered exponentials. Order-four joint PEPO Torch values and gradients
also match the full dense product with reuse on/off. Joint located PEPO
evaluation now skips unused
homogeneous component maps; verified joint lower contractions still
fall from nine to three at order four. The affected CPU
domain/API/layout gate passed **391 tests**, with two existing warnings;
the final correctness-review file then passed **19 tests** after the
last Torch regression. Full Ruff, whitespace and affected link checks
passed. The
[handoff](../../history/2026-09-28-joint-cluster-audit.md) records
scope and limits. The earlier full-suite result predates this audit;
nothing was committed or published. The large 5×6 graph MPO rank-cap
accuracy limitation above remains.

A [follow-up joint review](notes/2026-09-28-joint-cluster-followup.md)
found no further numerical mismatch in periodic directed/parallel-bond
PEPO, mixed trainable MPO/PEPO products, or small joint streaming and
recursive MPO cases. One independent periodic joint regression was added;
the correctness-review file then passed **20 tests**. The
[follow-up handoff](../../history/2026-09-28-joint-cluster-followup.md)
records the exact CPU scope. No production implementation or large-graph
accuracy claim changed.

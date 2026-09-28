# Cluster PEPO backend and downstream optimization status

Updated 2026-09-26. Pepsy `develop` at **`b4c4631`** is published, including
the backend correction `cf1d84c` and the newer remote work merged at
`a13031b`. Gaugy's matching API/refinement is published at **`a7af793`**.
These are development commits, not a new tagged package release.

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

## Tracking future work

Keep API behavior in the owning guide, implementation ownership in module
maps, derivations/benchmarks in dated notes, and session/publication events
in history. Historical "no push" or "unfixed" statements retain their
original baseline; use this ledger and Git for current status. Update these
links and measured scope when the implementation changes, without rewriting
old evidence into a claim of current validation. Whole-cost PEPO compilation,
general rank-changing derivatives and native cluster-symmetry optimization
remain deferred; they are not features of this published correction.

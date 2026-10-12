# 2026-09-28 — Joint cluster MPO/PEPO correctness audit

## Scope

This review checks that recent fixed-channel, symmetry-aware and complete
collection improvements apply to **ordered joint products**
`exp(A_S) @ exp(B_S) @ ...` in both operator representations. The working
tree is on `develop` with baseline `4e398e4`; prior uncommitted cluster
work and unrelated user jobs were preserved.

The shared `ClusterReusePlan` caches geometry, operator labels, coefficient
identities and factor order. The joint PEPO path uses the same verified plan
for exact targets and frozen lower-support contractions. The graph MPO path
forms ordered local targets, subtracts connected partitions and, with
`assembly="recursive"`, retains all compatible disjoint collections.
Numerical arrays are built for each evaluation and are not stored in the
structural plans.

## Concrete improvement

When any joint PEPO factor has located terms, the product uses the located
evaluator for **all** factors. It previously also built homogeneous
onsite/edge component tensors for each factor, although this route never
reads them. `PEPOClusterProductExpansion.exp` now selects the route once
and constructs only the component tensors that route consumes. This removes
unneeded backend work and autodiff graph nodes. It does not change the
local target, connected residual, geometry plan or tensor factorization.
The homogeneous route still builds its homogeneous components.

This is an **adopt** of the existing located component path; no upstream
contraction, SVD, canonicalization or backend dispatch policy changed.
The installed upstream capability audit in the earlier
[spatial reuse record](2026-09-28-cluster-spatial-reuse.md) remains
applicable. No dependency was modified.

## Independent checks

A new 2×2 open square test uses two noncommuting factors: onsite X plus
nearest-neighbor ZZ, followed by onsite Z plus nearest-neighbor XX. It builds
the selected-order result from dense local ordered exponentials, explicit
connected set partitions and independent basis-index embedding. For
`p=2,3,4`, both the fixed joint PEPO and complete recursive joint MPO
match that reference with spatial reuse off and on. The test also checks
that both compiled plans actually reuse targets, that the joint PEPO's
order-four lower-support contractions fall from nine to three, and that
the located route does not evaluate the unused homogeneous maps. Before
the implementation cleanup, the standalone independent probe found
maximum entry errors at most `1.56e-15` for PEPO and `4.45e-16` for
MPO across these six settings.

A separate two-site joint test compares JAX
`jit(value_and_grad)` through each complete fixed-channel MPO/PEPO builder
against dense ordered exponentials. It checks coefficient and step
derivatives at a zero coefficient, zero step and nonzero inputs. A separate
order-four located joint PEPO Torch regression compares full dense values
and coefficient/time gradients against the exact ordered product, for
reuse on/off and zero/nonzero inputs. Existing single-factor fixed/symmetry
suites remain in the affected validation gate.

## Limits

The 5×6 `p=4` graph MPO result in the
[numerical record](2026-09-28-graph-mpo-5x6-numerical.md)
still has large errors at assembly bond caps one and two. The complete
collection plan does not certify bond-compression convergence. These
small-system reference and JIT checks do not establish global large-system
operator error, arbitrary-geometry JIT support, complete Torch full-graph
capture, or native sector/fermionic recursive assembly.

## Validation for this review

In the activated existing Python 3.12 development environment, CPU only
(`CUDA_VISIBLE_DEVICES=''`, `JAX_PLATFORMS=cpu`, one BLAS/OpenMP thread),
the affected cluster/MPO/PEPO/public API/layout gate passed **391 tests**
with two existing deprecation warnings in 136.56 s. The gate covers the
new ordered cross-representation references and JAX JIT regressions,
existing Torch joint gradients, spatial reuse, fixed factorization,
recursive assembly and compression. After adding the final order-four
Torch joint-gradient regression, the complete correctness-review file
passed **19 tests** in 7.30 s. Full `python -m ruff check src tests`,
`git diff --check` and affected relative documentation links passed.
The preceding full-suite count predates this review; no new full-suite
or GPU result is claimed.

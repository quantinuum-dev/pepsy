# 2026-09-28 — Exact fixed-index MPO/PEPO construction for autodiff

## Status and ownership

Implemented in the working tree on `develop`, baseline `4e398e4`; no staging,
commit or publication. This implements the preceding
[design/prototype](2026-09-28-svd-free-cluster-design.md) for the cluster MPO and
Pauli PEPO constructors. It preserves the earlier uncommitted geometry,
compile/API, symmetry-reuse and recursive-assembly work plus unrelated sampler
changes. The legacy dense numerical `ClusterExpansionPlan` API is unchanged.

Public contracts: [MPO](../../api/operators/mpo_cluster.md#exact-construction-for-autodiff-without-svd),
[PEPO](../../api/operators/cluster_expansion.md#svd-free-differentiable-construction).

## Implemented

- Shared `factorization="fixed"` policy, with `"auto"` preserving existing
  numerical behavior. Fixed residual splits use M=I M or M=M I according to
  matrix shape. There are no numerical rank decisions, singular-vector bases
  or zero-value pruning in this MPO path. Trainable zero residuals keep their
  derivatives and channels. The policy covers both chain intervals and graph
  MPOs, including ordered products and complete recursive collection assembly.
  The one-shot facade also uses fixed splits when parsing general dense local
  operators into its temporary Hamiltonian basis; this closes a hidden SVD
  before residual construction. Existing user-created MPOBasis objects retain
  their original Hamiltonian initialization policy.
- Pauli PEPO fixed channels already covered common orders through four and
  uncapped localized histories. Fixed mode now also covers generic homogeneous
  trees at orders five through nine. Loop interactions remain in local targets
  and subtraction even when their tensor factorization uses a spanning tree.
- A bounded 256-entry immutable generic-tree topology cache is warmed during
  compilation. Geometry, maps, coefficient binding identities and symmetry
  plans are reusable; numerical values and autodiff tapes are fresh per call.
- Fixed mode rejects internal rank/cutoff compression, a final facade MPO chi,
  native sector MPO compilation, and ordered PEPO compress=True. MPO cutoff
  must explicitly be 0 or None. PEPO max_tree_rank must be None. Users can
  compress the resulting representation separately under the existing policies.
- `spatial_symmetries` accepts finite-site permutations (MPO chain indices,
  PEPO coordinates). Declarations must preserve graph edge multiplicities,
  operator labels, coefficient identities and factor order. Invalid or opaque
  unprovable assertions raise. Declarations do not bypass automatic local
  labeled-graph matching, and do not force equality of independent coefficient
  overrides. The normalized declarations enter MPO compiled cache keys.
- Automatic reuse already includes valid local translations, rotations,
  reflections and graph relabelings. A global finite-lattice declaration is
  distinct from such local equivalence: open-boundary wraparound translations
  are rejected, while local translated clusters can still reuse values.
- Fixed-mode C4 block transport requires each homogeneous edge slot to be
  invariant under endpoint reversal, so independent coefficient overrides
  cannot invalidate transport. Other oriented interactions should use automatic
  matching with symmetry=None or validated declarations. This guard does not
  change the legacy auto-mode C4 policy.
- Mixed Python/Torch Pauli coefficients now reuse the existing dtype-aware
  scalar converter. Previously a Python constant could first be rounded to
  Torch's default float32 and only then promoted when stacked with float64
  trainable parameters. Dense references exposed this at about 2e-9 on a
  four-site example; it is fixed rather than relaxing tolerances.
- Fixed MPO evaluation infers a backend for Python/NumPy time steps from
  trainable coefficients/operators, including bond-only models whose singleton
  targets are empty. Host scalars first retain their NumPy precision before
  conversion. Real and imaginary steps are checked at zero/nonzero coupling.

## Backend/compiler audit

The installed environment is unchanged from the
[recursive-assembly audit](2026-09-28-mpo-recursive-assembly.md#upstream-audit--classification):
Quimb 1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305,
Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1,
Symmray 0.4.1.dev7+g83fb22865 and Torch 2.6.0+cu124.
The prior official upstream source/doc audit applies to this continuing cluster
task. No dependency or installed-library changes were made.

**Adopt:** existing backend exponentials, semantic MPO assembly, sparse PEPO
blocks, shared dtype-aware conversion, and Autoray namespace allocation.
**Compatibility shim:** Torch fixed identities use the native tensor factory
`matrix.new_ones((rank,)).diag()`, scoped to this shared exact split. This
preserves dtype/device, including meta tensors in a no-GPU allocation check,
and avoids an installed Autoray/Torch Dynamo dtype-conversion failure.
Static MPO shape products now use Python integer arithmetic, avoiding a NumPy
scalar graph break. No global Torch SVD/QR registration is changed.

**Measured:** after warming lazy backend dispatch, a local fixed two-site MPO
kernel passes `torch.compile(backend="aot_eager", fullgraph=True)` value and
zero-parameter gradient checks. This checks captured forward/backward graphs,
not code-generation speed. The local JAX JIT value/zero-gradient check also
passes. **Defer:** whole-builder machine-code capture, general GPU timings,
and native sector/fermionic fixed construction. Structural compile_exp reuse
must not be described as an end-to-end Torch/JAX JIT guarantee.

## Validation

The focused MPO/PEPO/API/layout gate passed **213 tests**, with two existing
deprecation warnings, before the last three device/cache/C4 tests and JAX
kernel check were added. The initial new feature file independently passed
**15 tests**; the separate Torch full-graph regression passed **1 test**.
A subsequent fixed/compiler/MPO gate passed **85 tests** after adding
bond-only backend-inference regressions. Final dense-local-operator and
MPO/API checks are recorded in the handoff.
The new tests prohibit SVD entry points during fixed construction and compare
full dense values and coefficient/time gradients, including zero coefficients
and zero time, repeated calls, fixed core/block shapes, ordered factors and
crossing disjoint collections. They validate supplied reflections, rotations,
periodic translations, invalid claims, independent overrides and cache reuse.
The generic five-site PEPO is compared to a direct 32x32 matrix exponential.

The full CPU-only gate passed **5241 tests, 105 skipped**, with 781 warnings
(27:20). It started before the final narrow MPO host-step/dense-term parsing
fixes and separate compiler file. The subsequent final MPO/API/layout gate
passed **277 tests** with two existing deprecation warnings (52.06 s), covering
those changes. Full Ruff, whitespace and 31 relative documentation link checks
passed. Exact scope and commands are in the
[session handoff](../../../history/2026-09-28-fixed-cluster-autodiff.md).
The earlier 5,199-pass gate predates the recursive and fixed construction work.
No GPU gate was run because the user's GPU workloads remain active.

## Scoped timing evidence

CPU float64, one BLAS/OpenMP thread, one warmup plus five timed calls, medians.
Every timed call constructs the operator, materializes its small dense matrix,
and differentiates its real-entry sum with respect to h and time. Uniform
X + 0.3 ZZ, h=0.2, step=0.04. CPU regression work ran concurrently, so these
are indicative observations rather than controlled performance guarantees.

| Workload | Numerical auto | Fixed | Agreement |
| --- | ---: | ---: | --- |
| Four-site cycle MPO, p=4, recursive/no assembly compression | 58.3 ms | 46.0 ms | max value 1.33e-15; max gradient 1.78e-15 |
| 2x2 open Pauli PEPO, p=4 | 279 ms | 268 ms | identical values/gradients; this path already used fixed channels |
| 1x5 open Pauli PEPO, p=5 | 1.690 s | 1.529 s | max value 2.22e-16; max gradient 7.11e-15 |

These complete small-workload observations are much more modest than the
8–11x local split speedup in the earlier isolated prototype. They establish no
large 2D, GPU or generic end-to-end speedup. Construction, materialization,
contraction and backward can each dominate in other regimes.

## Remaining limitations

Exact fixed local channels can increase memory and contraction cost. They do
not solve global MPO bond growth or bound PEPO dense materialization. The
recursive state budget counts subproblems, not tensor bytes. A fixed reduced
basis would be a separate approximation, not an exact substitute for this
construction. Native sector conversion and boundary contraction can still use
decompositions outside the fixed constructor. Explicit native sector MPO fixed
construction is rejected; optional charge/fermionic extensions are not claimed.

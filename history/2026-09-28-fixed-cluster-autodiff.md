# 2026-09-28 — Fixed-index differentiable cluster MPOs and PEPOs

- Scope: user authorized SVD-free MPO/PEPO construction, geometry/term-aware
  symmetry reuse, supplied symmetry declarations, and reusable compiled plans.
- Branch / baseline: `develop`, `4e398e4`.
- Commit status: working-tree changes only; no staging, commits or publication.
  Earlier cluster work and unrelated sampler/live-job changes are preserved.

## Implementation

`factorization="fixed"` gives exact structural splits with no numerical SVD,
retains zero-channel derivatives, and rejects internal compression. It covers
MPO intervals/graphs/ordered products and Pauli PEPOs including generic trees.
Prepared generic-tree topology is cached. `spatial_symmetries` declarations are
validated against finite geometry, operators and coefficient bindings; local
reuse remains conservative under independent overrides. Fixed-mode C4 block
transport is guarded. A mixed Python/Torch coefficient precision issue found
by the new dense references is corrected with the existing conversion helper.
Fixed MPO calls infer a tensor backend for host-valued time steps, including
bond-only models. One-shot dense local-operator parsing uses fixed splits too,
closing a hidden SVD in its temporary Hamiltonian basis.

See [implementation, audit and measurements](../docs/development/notes/2026-09-28-fixed-cluster-autodiff.md)
and the linked owning API guides for details and limits.

## Validation

- Focused MPO/PEPO/API/layout gate: 213 passed, 2 existing deprecation warnings
  (64.82 s), before the final four device/cache/C4/JAX tests were added.
- New fixed-construction file: 15 passed (9.12 s).
- Separate Torch full-graph local-kernel regression: 1 passed (4.09 s).
- Final affected MPO/API/layout gate: **277 passed**, 2 existing deprecation
  warnings (52.06 s): tests/test_cluster_fixed_factorization.py,
  tests/test_cluster_fixed_compile.py, tests/test_mpo_cluster.py,
  tests/test_mpo_cluster_recursive.py, tests/test_mpo.py,
  tests/test_public_api.py and tests/test_package_layout.py. This includes
  dense-local-operator values/operator-entry gradients with SVD forbidden,
  and real/imaginary host steps at zero/nonzero coupling.
- Full CPU-only suite: **5241 passed, 105 skipped, 781 warnings**
  (1640.97 s / 27:20). It started
  before the final narrow MPO host-step and dense-term parsing fixes; the
  subsequent 277-pass gate above covers those fixes. The separate compiler
  file was added after full-suite collection.
- Full `ruff check src tests`, `git diff --check`, and 31 relative documentation
  link checks passed.

CPU-only, one BLAS/OpenMP thread. No GPU tests were run. Initial new PEPO
references caught mixed-scalar precision loss; corrected without loosening
reference tolerances. A local Torch compiler failure was narrowed to dtype
conversion and replaced by a device-preserving native tensor factory, with
value, zero-gradient and meta-device regressions.

## Limits

Machine graph capture is checked only for the local kernel, not the whole
builder. Larger exact bonds/materialization can dominate runtime. No native
sector/fermionic fixed MPO support or GPU speedup claim. Small complete
construction/dense-contraction/backward timings are in the evidence note;
no general fastest-method claim is made.

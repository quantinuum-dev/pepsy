# 2026-09-28 — Cluster implementation correctness review

- Scope: user requested another correctness review of the compiled,
  symmetry-aware SVD-free cluster MPO/PEPO implementation.
- Branch / baseline: `develop`, `4e398e4`.
- Status: uncommitted working-tree changes; no staging, commits or publication.
  Previous cluster and unrelated optimizer/sampler changes are preserved.

## Findings and fix

Fixed a reproducible backend mismatch for positional tensor parameters and
callable coefficients with a host-valued time step. Fixed-mode local targets
now align before subtraction without extra callback calls; float32 and complex
components are preserved. Cluster parameter defaults are now honored.
Only cluster MPO implementation code changed. Independent square-lattice
partition references validate both MPO and PEPO through p=4, including disjoint
products and loops, with and without symmetry reuse.

See [detailed evidence and audit](../docs/development/notes/2026-09-28-cluster-correctness-review.md).

## Validation

- New independent/binding regressions: 15 passed (7.34 s).
- Broader affected domain/API/layout gate: **374 passed**, two existing
  deprecation warnings (92.83 s), covering test_cluster_correctness_review,
  test_cluster_fixed_factorization, test_cluster_fixed_compile,
  test_cluster_spatial_reuse, test_cluster_expansion, test_mpo_cluster,
  test_mpo_cluster_recursive, test_mpo, test_public_api and test_package_layout.
  This run collected the first 13 review cases; the last complex/JAX cases
  passed in the separate final 15-case run. All implementation changes were
  present for both runs.
- Full Ruff, whitespace and affected relative documentation link checks passed.

All numerical checks are CPU-only using the existing development environment.
The preceding 5241-pass full suite predates this narrow MPO fix; no new full
suite or GPU result is claimed. No remaining failure was observed in the new
regressions. Whole-builder JIT and large-system memory/convergence guarantees
remain outside the validated scope.

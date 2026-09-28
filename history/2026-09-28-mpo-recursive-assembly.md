# 2026-09-28 — Complete graph MPO add/compress assembly

- Scope: implement the user's add-MPO-contributions-and-compress request while
  retaining all compatible collections in the selected spatial expansion.
- Branch / baseline: `develop`, `4e398e4`.
- Commit status: working-tree edits only; nothing staged, committed or published.
  Earlier cluster work and independent sampler/live-job files are preserved.

## Changes and findings

- New `assembly="recursive"` caches a remaining-site DAG, shares subproblem
  MPOs and sums/compresses branches. Compilation has a hard state budget,
  separate from tensor rank caps, and never drops collections on overflow.
- Fixed streaming direct-plan loss of separated residual products. Prepared
  orthonormal environments before truncating both graph assembly paths.
- Documented controls, diagnostics, backend limitations and complexity limits.
  Detailed implementation, audit and measurements are in the
  [evidence note](../docs/development/notes/2026-09-28-mpo-recursive-assembly.md).

## Validation

CPU-only, activated existing development environment, one BLAS/OpenMP thread:

```text
python -m pytest -q -o addopts='' tests/test_mpo_cluster_recursive.py \
  tests/test_mpo_cluster.py tests/test_cluster_spatial_reuse.py \
  tests/test_mpo.py tests/test_public_api.py tests/test_package_layout.py
270 passed, 2 existing deprecation warnings (47.92 s)
```

Full `python -m ruff check src tests` and `git diff --check` passed.
Relative links in affected guides/evidence/ledger were checked. The prior
5,199-pass CPU full-suite gate predates this assembly change; no new full-suite
result is claimed. Initial reference checks exposed the compression-gauge
issue, now protected by left/right low-rank accuracy regressions. Torch
repeated-call values and coefficient/time gradients passed with and without
assembly compression. No GPU tests were run because user workloads are active.

## Remaining limits

- 5x6 p=2 chi=4 numerical assembly measured; 5x6 p=4 planning measured only.
  Numerical accuracy for the 30-site example is unverified. A 3x3 chi=8
  comparison against explicit complete assembly has relative error 4.08e-7.
- Native charge/fermionic recursive assembly is rejected. Rank-dependent
  derivatives retain existing SVD smoothness restrictions.
- DAG width, local residual ranks and temporary sum/product bonds still cost
  memory/time; the state budget is not a byte bound. Compression convergence
  is separate from spatial-cutoff convergence.

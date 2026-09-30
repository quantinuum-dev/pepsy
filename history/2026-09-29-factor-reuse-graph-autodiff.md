# 2026-09-29 — Individual factor reuse and graph PEPO autodiff

- Scope: implement the user's requested gaps 1 and 2 and report results.
- Branch / baseline: Pepsy `develop`, `b343ced`; Gaugy `develop`, `3da7543`.
- Commit status: Pepsy changes are in the working tree, not committed or
  pushed. Existing fitting/tree, skill, optimizer, and journal changes were
  preserved; the changelog received only an additional cluster entry.

## Implemented

- Square localized/uniform products and graph local products reuse each
  factor's exponential when other factors break complete-product symmetry.
  Binding/operator/axis proofs remain structural; independent vectors and
  opaque callbacks retain their semantics. Numerical caches live for one
  evaluation. Square residual traces receive the same binding information.
- Square localized exponentials pool representatives across factors and
  clusters, without redundant exponentials. Common backends are retained
  through mixed constant/trainable factors, including with reuse disabled.
- Ordered graph PEPOs accept `factorization="fixed"` and retain Torch/JAX
  tensors through tree splitting, active blocks, explicit networks and dense
  contractions. Uncapped backend auto uses exact fixed splits; NumPy auto
  retains numerical SVD compression. Graph tensor zeros are never pruned by
  inspecting runtime values, preserving derivatives and JAX tracing.
- Fixed factorization rejects rank caps. Capped backend auto uses the existing
  SVD and its existing rank/gap limits. Backend/fixed reports mark uncomputed
  reconstruction error/norm as `None`. No backend driver or dependency changed.
- Gaugy needs no evaluation implementation change. Its capability-gated
  integration test and API guide now cover the graph path. Affected dense
  references were corrected against SciPy without relaxing tolerances.

## Measured savings and checks

The three-factor regression deliberately gives the middle factor independent
site/edge parameters, preventing complete-product reuse. It measures actual
matrix targets passed to the exponential helpers, not wall-clock speedup:

| Case | Reuse disabled | Factor reuse | Reduction |
| --- | ---: | ---: | ---: |
| 2x2 square, full order four | 39 | 21 | 46.2% |
| Three-site graph chain, full order three | 18 | 12 | 33.3% |

Both materialization and scalar residual traces show these counts. Repeated
parameter sets and Torch/JAX gradients match independent dense products.
Independent equal/zero vectors do not acquire symmetry; stateful callback
counts and values are preserved. Additional checks cover JAX JIT, coefficient,
step and factor-scale derivatives, zero steps/coefficients, real/complex steps,
loops, higher-body interactions, disconnected sites, partial cutoff versus
MPO assembly, rank-cap forward values, and Gaugy partial-default handoff.

Final affected Pepsy gate: **385 passed, two existing deprecation warnings,
in 165.90 s**, including 22 tests in the two new files:

```sh
python -m pytest -q -o addopts='' \
  tests/test_cluster_factor_reuse.py tests/test_graph_pepo_autodiff.py \
  tests/test_square_cluster_plan.py tests/test_interaction_cluster_plan.py \
  tests/test_cluster_expansion.py tests/test_cluster_trace.py \
  tests/test_cluster_fixed_factorization.py tests/test_cluster_correctness_review.py \
  tests/test_mpo_cluster_recursive.py tests/test_cluster_api.py \
  tests/test_cluster_api_review.py tests/test_cluster_spatial_reuse.py \
  tests/test_cluster_fixed_compile.py tests/test_cluster_jit_gradients.py \
  tests/test_mpo_cluster_compression.py tests/test_mpo_cluster.py \
  tests/test_graph_pepo_product.py tests/test_public_api.py tests/test_package_layout.py
```

Final affected Gaugy gate: **294 passed, six existing warnings, in 129.16 s**;
see its [handoff](../../gaugy/history/2026-09-29-factor-reuse-graph-autodiff.md).
Both used `source ~/envs/py312/bin/activate`, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=''`, and
`JAX_PLATFORMS=cpu`. Pepsy `python -m ruff check src tests`, Gaugy changed-file
Ruff, local Markdown links, and `git diff --check` passed.

No full-suite, GPU, native Symmray, or whole-objective Torch compiler
validation is claimed. The prescribed external Gaugy notebook remains absent.

## Numerical and compatibility evidence

See the [dated review](../docs/development/notes/2026-09-29-factor-reuse-graph-autodiff.md)
for installed versions, primary-source checks, mixed-backend corrections,
Torch singleton precision diagnosis, prepared topology cache preservation,
and the adopt/compatibility/defer decisions. The
[API guide](../docs/api/operators/interaction_clusters.md) describes usage,
backend defaults, reporting and rank restrictions.

Large fixed-channel memory costs, general rank-changing derivatives, GPU
performance, and diagonal/higher-body square virtual routing remain outside
this change. Do not treat implementation in this working tree as publication.

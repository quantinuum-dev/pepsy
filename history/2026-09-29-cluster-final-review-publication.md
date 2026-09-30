# 2026-09-29 — Final cluster review and authorized publication

- Scope: user's explicit request to review again, commit, and push the shared
  MPO/PEPO/Pauli cluster API and interaction-graph planner work.
- Branch / baseline: `develop`, `a233a40`.
- Publication: this record accompanies the implementation commit; earlier
  journals' local/uncommitted statements describe their original baseline.
- Concurrent MPS edits, their changelog entry, and unrelated journals are
  excluded from this publication and retained in the working tree.

## Final review corrections

- Runtime MPO coefficient rebinding preserves legacy explicit-graph support
  rules. A term on sites 0 and 2 of a three-site path may use the connected
  three-site cluster; rebinding no longer accidentally applies the stricter
  shared-plan support check. Repeated values agree with dense exponentials.
- Graph active-block dtype metadata uses empty backend views instead of
  passing Torch dtype objects to NumPy. Tensor-network materialization keeps
  matrix orientation and gradients, including sites without virtual edges.
  The latter also fixes the common block helper's missing isolated-site mask.
- Shared-plan MPO factories validate term indices before indexing site labels,
  giving an explicit `ValueError` instead of leaking `IndexError`.
- The new graph-planner changelog entry is under Unreleased; API documentation
  describes the shared-plan versus legacy graph support validation contract.

## Final validation

Activated the shared Python 3.12 environment; used `JAX_PLATFORMS=cpu` and
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for numerical checks.

Pepsy: **341 passed, 2 compatibility deprecation warnings, 107.67 s**:

```sh
python -m pytest -q -o addopts='' \
  tests/test_interaction_cluster_plan.py tests/test_cluster_expansion.py \
  tests/test_cluster_trace.py tests/test_cluster_fixed_factorization.py \
  tests/test_cluster_correctness_review.py tests/test_mpo_cluster_recursive.py \
  tests/test_cluster_api.py tests/test_cluster_api_review.py \
  tests/test_cluster_spatial_reuse.py tests/test_cluster_fixed_compile.py \
  tests/test_cluster_jit_gradients.py tests/test_mpo_cluster_compression.py \
  tests/test_mpo_cluster.py tests/test_graph_pepo_product.py \
  tests/test_public_api.py tests/test_package_layout.py
```

Gaugy with the same Pepsy working tree: **255 passed, 6 upstream deprecation
warnings, 90.60 s**. Selection: `test_interaction_cluster_plan`,
`test_cluster_exponential_api`, `test_cluster_api_consistency`, `test_cluster`,
`test_cluster_numerics`, `test_cluster_optimization_api`, `test_exact_compiled`,
`test_ordered_product`, and `test_package`.

- `python -m ruff check src tests` passed in Pepsy; Ruff passed on Gaugy's
  changed cluster implementation and integration tests.
- `python examples/interaction_cluster_plan.py` passed: connected counts
  `{1: 4, 2: 5, 3: 4}`, collection count 12, shared MPO/PEPO topology and trace.
- `git diff --check` passed in both repositories.
- Upstream audit from the same task and unchanged environment was reused;
  see the [implementation record](2026-09-29-interaction-cluster-plan.md).

These are focused subsystem checks, not a full-suite claim. The earlier
baseline SU zero-rank failure and unavailable Torch Inductor Python headers
remain outside this change; they were independently reproduced before this
review and are recorded in Gaugy's matching implementation journal.

The graph product builder still materializes NumPy tensors only; its scalar
traces/residuals preserve Torch/JAX gradients. This differs from manually
supplied backend-native graph active blocks, whose tensor-network conversion
is covered by the new Torch regressions. Geometry/boundary inference and
combinatorial limits remain as documented in the
[interaction-cluster guide](../docs/api/operators/interaction_clusters.md).

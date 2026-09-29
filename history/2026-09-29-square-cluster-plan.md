# 2026-09-29 — Shared plans select square PEPO construction

- Scope: the authorized next step connecting compatible NN square plans to
  the existing 2D PEPO backend. Diagonal/NNN routing remains a later step.
- Branch / baseline: `develop`, `f1b9924`; that independently published MPS/tree
  commit was already present before this change was committed.
- Publication: this record accompanies the focused square-plan implementation
  commit. Concurrent fitting/tree changes and their journal are excluded.

## Implemented

- `PEPOClusterProductExpansion.from_plan` and `PauliPEPOBasis.from_plan`
  accept `layout="auto"`, `"square"`, or `"graph"`. Auto selects the new public
  `SquarePEPOClusterProductExpansion` for compatible inputs; direct graph
  factories retain graph output.
- Require the complete NN graph, matching 2D shape and explicit OBC/PBC
  metadata, row-major placement, and fixed one-/two-site Pauli products.
  Incompatible auto inputs use the existing graph path. Explicit square
  selection reports incompatibility. No boundary edges are invented.
- Preserve original factor order/scales, term slots and parameter objects.
  Each supplied length-two periodic term gets one deterministic virtual
  route; repeated physical terms are neither dropped nor doubled.
- Square localized records consume the shared plan inventory. Existing square
  verified target/subsupport reuse remains available; numerical values and
  runtime coefficient-vector slots stay fresh and independent.
- Square `exp` returns active blocks or an explicit Quimb PEPO; active blocks
  gain `to_dense()`. Fixed channels support Torch/JAX materialization.
  Scalar partition traces stay separate from rank-capped materialization;
  local residual inspection lazily uses the shared MPO engine.
- Updated public guides, status ledger, changelog, and runnable
  [example](../examples/square_interaction_cluster_plan.py). Matching Gaugy
  integration selects this public factory while preserving Pauli endpoints.

## Validation

Activated the shared Python 3.12 environment with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu`.

```sh
python -m pytest -q -o addopts='' \
  tests/test_square_cluster_plan.py tests/test_interaction_cluster_plan.py \
  tests/test_cluster_expansion.py tests/test_cluster_trace.py \
  tests/test_cluster_fixed_factorization.py tests/test_cluster_correctness_review.py \
  tests/test_mpo_cluster_recursive.py tests/test_cluster_api.py \
  tests/test_cluster_api_review.py tests/test_cluster_spatial_reuse.py \
  tests/test_cluster_fixed_compile.py tests/test_cluster_jit_gradients.py \
  tests/test_mpo_cluster_compression.py tests/test_mpo_cluster.py \
  tests/test_graph_pepo_product.py tests/test_public_api.py tests/test_package_layout.py
```

**363 passed, two existing deprecation warnings, 131.34 s.** The 22 new cases
cover partial/full cutoffs, OBC/cylinders/PBC, a length-three periodic seam,
asymmetric/reversed/repeated Pauli terms, algebraic ordered products, shared
parameters, independent runtime vectors, partial factor vectors, invalid
geometry fallback, Torch gradients and JAX JIT gradients including zeros.
Full-cutoff results are checked against independent SciPy/backend matrix
exponentials; partial cutoffs agree with graph assembly.

`python -m ruff check src tests`, example lint, and `git diff --check` passed.
`python examples/square_interaction_cluster_plan.py` passed: cluster counts
`{1: 4, 2: 4, 3: 4}`, maximum fixed compact bond 13, square/graph matrix
difference `1.44e-15`. No performance benchmark or full-suite run is claimed.
The matching Gaugy journal records its focused checks and absent prescribed
notebook. Earlier unrelated SU/Inductor limitations were not retested here.

## Decisions and remaining work

See [dated compatibility evidence](../docs/development/notes/2026-09-29-square-cluster-plan.md)
for the reused upstream audit, unchanged versions, public signatures, and
adopt/shim/defer decisions. No dependency or backend-driver change.

Diagonal/NNN square virtual routing is still unimplemented; those plans keep
generic graph output. General matrix/string/fermionic/charged terms also
remain on their existing paths. Graph materialization remains NumPy-only;
native Symmray square support is unverified. Square-specific factorization
options apply only to the square representation. Further representative
factorization reuse and incremental cutoff/performance work are not part of
this change.

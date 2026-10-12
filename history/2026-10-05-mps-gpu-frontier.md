# 2026-10-05 — Research-guided GPU trajectory probability batching

- Scope: user authorized the next GPU optimization and asked for CUDA-Q,
  Torch/CuPy and literature research first.
- Branch / baseline: `develop` / `bec773a`; earlier working-tree edits preserved.
- Status: uncommitted. Nothing staged, committed, installed or published.

Reviewed official CUDA-Q PTSBE documentation, its MPS simulator implementation,
NVIDIA's cuTensorNet MPS example, Torch/CuPy execution and SVD documentation,
and primary PTSBE/tensor-network trajectory papers. Rechecked the required
upstream pages and installed signatures/versions. CUDA-Q and cuQuantum are not
installed; no direct comparison benchmark is claimed. See the
[research, measurements and limits](../docs/development/notes/2026-10-05-mps-gpu-frontier-research.md).

Implemented automatic batching of compatible one-site Kraus probability blocks
across coalesced Torch CUDA/CuPy parents through Autoray. Reused amplitude
preparation and operator caching; bounded chunks by 32 parents and a conservative
32 MiB probability-workspace estimate. Preserved per-parent fallback, host RNG
order, callable proposal ordering, gradients, rare probabilities and branch
budgets. Added the maximum parent batch diagnostic and explicit generator
cleanup before subsequent gates. Whole gate/SVD segments and independent-shot
execution are not vectorized by this change. The workspace target does not
bound total retained MPS, caches, allocator reserves or autograd storage.

Alternating synchronized benchmarks on the RTX A5000 measured **1.72–1.92x**
probability-kernel speedup and **1.06–1.10x** complete coalesced replay speedup
for the recorded branching workload and both complex dtypes. No universal
crossover or CUDA-Q performance claim.

Validation:

- Combined trajectory/Kraus/automatic/dense-reference/dynamic-control/fermion,
  MPI and API/layout selection: **589 passed, 1 skipped, 30 deselected** before
  final additive diagnostics and extra cases. The skip needs a second JAX
  device; the deselected tests are slow.
- Final trajectory/importance/frontier selection: **179 passed, 1 skipped**,
  including all **31 new frontier cases**. This overlaps the earlier selection.
- GPU-backend/frontier selection: **78 passed, 3 failed, 1 skipped** before two
  final workspace-lifetime cases. Skip: Metal unavailable. Failures are the
  original CUDA complex64 ledger cases (both normalization settings) and JAX
  complex64 restored-norm ledger case. Existing thresholds are unchanged.
- Public API/layout after the diagnostic addition: **54 passed**.
- Final ledger-only confirmation with the original test body intact:
  **7 passed, 3 failed, 43 deselected**, reproducing the same three baseline
  failures; both configured-SVD checks pass.
- All-source/test Ruff, relative documentation links and `git diff --check`
  passed. No full repository suite or multi-rank GPU run.

The CUDA discrepancy was isolated to native complex64/Jacobi factorization
roundoff, not fixed by changing ledger values. The existing public
`TorchLinalgConfig(stabilized=False)` uses `gesvd` and passes both original
3e-6 comparisons in added tests. Complex128 probes agree at roughly 2e-15.
MPS replay deliberately preserves the caller's configured policy; no hidden
global registration was added. The native-driver and JAX failures remain open.

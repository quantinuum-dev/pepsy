# 2026-10-05 — CUDA-Q research and bounded GPU trajectory batches

Scope: user requested GPU optimization after reviewing CUDA-Q, Torch, CuPy and
the literature. Work is on `develop` / `bec773a`, with earlier uncommitted
changes preserved. No dependency installation, commit or publication.

## Research and decisions

- **Adopt reuse, defer altered sampling:** CUDA-Q's
  [PTSBE guide](https://nvidia.github.io/cuda-quantum/latest/using/examples/ptsbe.html)
  and [Patti et al. (2025)](https://arxiv.org/html/2504.16297v1) separate noise
  realization selection from expensive state preparation and repeated terminal
  sampling. The guide restricts PTSBE to static circuits without mid-circuit
  measurements/feed-forward. Limiting realizations and reallocating shots can
  change statistics. Pepsy keeps exact count coalescing and state-dependent
  Kraus sampling; no trajectory probability cutoff or non-proportional dataset
  sampling was introduced. Published headline speedups are not transferable
  to this workload.
- **Adopt shape-aware reuse:** CUDA-Q's
  [MPS simulator source](https://github.com/NVIDIA/cuda-quantum/blob/main/runtime/nvqir/cutensornet/simulator_mps.h)
  prepares factorization settings, recomputes MPS factorization per noisy
  trajectory, and caches samplers/workspaces by tensor extents. This supports
  explicit shape compatibility rather than assuming all branches retain the
  same ranks. No CUDA-Q source was copied.
- **Adopt the applicable principle, defer broader contraction changes:**
  [Patti et al. (2026)](https://arxiv.org/html/2604.08467v1) reuse contraction
  paths for compatible network structures and avoid duplicate partial-sampling
  work. Pepsy already has compiled stream plans and shared operator caches.
  The new path groups equal-shape probability blocks. It does not implement
  that paper's contraction-path algorithm or terminal sampler.
- **Adopt bounded workspaces:** NVIDIA's
  [cuTensorNet MPS example](https://github.com/NVIDIA/cuQuantum/blob/main/samples/cutensornet/approxTN/mps_example.cu)
  separates workspace sizing from repeated gate execution. Here we bound only
  additional probability-batch workspace; no cuQuantum dependency is added.
- **Defer whole-trajectory CUDA graphs:**
  [Torch 2.6 CUDA semantics](https://docs.pytorch.org/docs/2.6/notes/cuda.html)
  require fixed addresses/shapes and prohibit dynamic data-dependent control
  in captured execution. MPS rank changes, host RNG, measurements and adaptive
  truncation remain on their established paths.
- **Preserve explicit numerical policy:**
  [Torch SVD documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd)
  describes driver accuracy/speed tradeoffs. Its
  [numerical accuracy guide](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
  also cautions that batched and sliced computations need not be bitwise
  identical. No approximate SVD, rank padding, TF32 policy or dtype change was
  enabled. Tests compare physical states/distributions at dtype-aware tolerances.
- **Measure synchronously:**
  [CuPy performance guidance](https://docs.cupy.dev/en/stable/user_guide/performance.html)
  motivates warm-up and device synchronization around measured work. Its
  [SVD documentation](https://docs.cupy.dev/en/v14.2.0/reference/generated/cupy.linalg.svd.html)
  limits the small-matrix batched Jacobi fast path to matrix dimensions at most
  32. We therefore do not assume general batched MPS SVDs will be faster.
- **Retain workload limits:**
  [Bonnes and Laeuchli (2014)](https://arxiv.org/abs/1411.4831) benchmark
  MPS trajectories against superoperator evolution and identify entanglement
  growth and parameter regime as important efficiency constraints. A fixed
  backend or batching choice is not universally best.

No new compatibility shim was needed. Direct CUDA-Q/cuQuantum execution is
**unverified** because those packages are not installed. The review used
official documentation/source and primary papers, not a CUDA-Q benchmark.

## Installed capabilities and upstream audit

Python 3.12; NumPy 2.5.2; Torch 2.6.0+cu124; CuPy 14.1.1;
Quimb 1.15.1.dev79+gb5e316200; Autoray 0.11.1.dev9+g1291702f9;
Cotengra 0.8.3.dev7+g1d7fd333f; Symmray 0.4.1.dev11+g1a3481803.
The GPU runtime is available on the RTX A5000 used in prior measurements.
NVML via `nvidia-smi` was unavailable in this session; Torch/CuPy execution
and synchronized timing succeeded.

Rechecked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray). The
[Symmray abelian-array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
returned an internal error. Native Symmray remains on its existing path.
Installed signatures confirmed Autoray namespaces and Quimb canonicalization;
real GPU tests exercised broadcasting matmul, reductions, stacking and dtype
preservation. Web documentation can describe newer releases than these installed
versions; installed dispatch and tests are the capability evidence.

## Implemented

`noise.py` now batches one-site Kraus probability evaluation across compatible
coalesced GPU parents. It reuses the existing amplitude preparation, converted
operator cache, Autoray namespace and scaled projected-norm reduction. All
matrix products/reductions stay on Torch CUDA or CuPy. One parent-by-outcome
norm matrix is downloaded per batch and squared in host float64, retaining
tested complex64 probabilities as small as 1e-60. Existing per-parent raw-state
norm guards still synchronize separately.

Groups share backend, dtype, device and prepared block shape. At most 32
parents are prepared per chunk, with a conservative 32 MiB temporary estimate.
The estimate includes prepared/stacked blocks, operator arrays and reduction
temporaries; it is not a cap on MPS storage, caches, allocator reserves or
autograd graphs. Large blocks, CPU/native/exact/cyclic/multi-site cases and
callable importance proposals use existing per-parent evaluation. No changes
were made to RNG ordering, branch-cap accounting, physical gate application,
adaptive decomposition, normalization or cutoff.

`TrajectoryDiagnostics.max_kraus_parent_batch` reports the largest parent
batch used, default one. This is a first branch-batching stage; whole MPS gate
segments/SVDs and independent-shot replay are not vectorized by this change.

## CUDA accuracy finding

The original complex64 ledger failures reproduce before and after this change.
With three nonadjacent CNOTs on the existing seed-19 four-site test, Torch's
native/Jacobi route differs from NumPy infidelity by 3.8413e-6 without norm
restoration and 3.7025e-6 with restoration, exceeding the unchanged 3e-6 test
tolerance. Float64 scalar bookkeeping cannot repair upstream complex64
factorization roundoff. A separate same-input complex128 probe agrees across
NumPy, Torch CPU and CUDA to roughly 2e-15 in the ledger.

The existing public `TorchLinalgConfig(stabilized=False)` defaults to `gesvd`.
With that policy, the complex64 errors are -7.8446e-7 and +1.1919e-7 and both
original tolerance comparisons pass. New tests verify this supported policy;
the original native-driver tests and their tolerances are unchanged. No hidden
global registration was added to MPS replay. A fresh baseline also reproduced
the JAX restored-norm discrepancy (approximately -4.0527e-6); it remains open.

## Measured performance

Final benchmark uses synchronized wall timing, three warm-ups for both paths,
seven alternating baseline/batched repetitions and medians. One OpenBLAS/OMP
thread. Workstation GPU is shared, not reserved. Baseline disables only parent
batching; it retains the previous outcome batching. Scripts/logs are under
`/tmp/pepsy-frontier-benchmark-alternating.*`; timings below are durable evidence.

Probability workload: 16 live parents of a 16-site random MPS, chi=32,
amplitude damping p=.2 at site 8. Prepared centers and caches are warm.

| Backend / dtype | Per-parent | Batched parents | Speedup |
| --- | ---: | ---: | ---: |
| Torch CUDA / complex64 | 11.221 ms | 5.831 ms | 1.92x |
| Torch CUDA / complex128 | 7.860 ms | 4.570 ms | 1.72x |
| CuPy / complex64 | 11.426 ms | 6.007 ms | 1.90x |
| CuPy / complex128 | 11.408 ms | 6.033 ms | 1.89x |

Full replay: seed-932 random eight-site chi-8 MPS; bit-flip p=.5 at sites 0,1,2,
then damping p=.2 at sites 0,2,4,6; 64 coalesced shots, seed 173, one worker,
retain=none, default auto cutoff, instrumentation/progress off. Outer construction
and initial upload excluded; shot preparation and replay included.

| Backend / dtype | Per-parent | Batched parents | Speedup |
| --- | ---: | ---: | ---: |
| Torch CUDA / complex64 | 163.081 ms | 148.384 ms | 1.10x |
| Torch CUDA / complex128 | 171.835 ms | 155.771 ms | 1.10x |
| CuPy / complex64 | 354.131 ms | 335.005 ms | 1.06x |
| CuPy / complex128 | 401.855 ms | 380.376 ms | 1.06x |

An earlier non-alternating run overstated the Torch complex64 full-run gain
(1.36x); the alternating result is the reported comparison. These gains apply
to this branching workload, not all circuits or general CUDA-Q performance.

## Validation

- Combined trajectory, automatic scheduling, Kraus batching, dense-reference,
  dynamic-control, fermion, MPI and API/layout selection: **589 passed,
  1 skipped, 30 deselected**. This preceded the final additive batch diagnostic
  and four extra dtype/invalid-state cases.
- Final frontier plus complete GPU-backend selection: **78 passed, 3 failed,
  1 skipped**. All 29 new frontier cases and both new configured-SVD checks pass.
  Failures are the unchanged native CUDA ledger cases and JAX restored-norm
  case described above. Skip: unavailable Metal backend.
- Public API/layout after the diagnostic addition: **54 passed**.
- Final trajectory/importance/frontier selection after explicit workspace
  release: **179 passed, 1 skipped**, including all **31** new frontier cases.
  This overlaps the earlier selections; counts are not additive.
- Final ledger-only confirmation: **7 passed, 3 failed, 43 deselected**,
  preserving the complete original test body and the same baseline failures.
- Ruff and diff whitespace/link checks are recorded in the handoff. No full
  repository suite or multi-rank GPU run was performed.

Remaining work: vectorized gate/SVD segments require a separate rank/gradient
contract; automatic retained-state memory budgeting and continuation after a
coalesced cap remain deferred. This implementation bounds probability workspace
and preserves existing replay semantics, not a claim of universal speed or
error-free floating-point equivalence across backends.

# 2026-10-05 — CUDA MPS trajectory performance review

Extension of the [CPU/API review](2026-10-05-mps-trajectory-performance-review.md),
requested by the user. Baseline: `develop` / `bec773a` plus the existing
uncommitted trajectory corrections. No production implementation was changed.

## Hardware, environment and methodology

- NVIDIA RTX A5000, 24 GiB class (nvidia-smi reports 24564 MiB), driver 555.42.06.
  CUDA execution was confirmed, including a complex128 device SVD.
- Torch 2.6.0+cu124; runtime CUDA 12.4; NumPy 2.5.2; Python 3.12.
  Quimb 1.15.1.dev79+gb5e316200, Autoray 0.11.1.dev9+g1291702f9,
  Cotengra 0.8.3.dev7+g1d7fd333f and Symmray 0.4.1.dev11+g1a3481803.
  Versions match the [earlier audit](2026-10-05-mps-trajectory-corrections.md).
- Dense complex64 arrays; direct compression; cutoff=1e-8; retain=none;
  progress and replay instrumentation disabled. One OpenBLAS/OMP thread and
  one Torch intra/inter-op thread. CPU results are not a CPU thread-count sweep.
- Fresh processes had no explicit Pepsy Torch linalg policy installed;
  Autoray dispatched to native `torch.linalg.svd`. No approximate SVD policy
  or custom driver was selected.
- Warm-up plus three timed repetitions per configuration. CUDA synchronized
  before/after each timed call. Construction of the outer simulator and initial
  CPU-to-GPU upload were excluded; per-shot construction/replay were included.
  Profiler instrumentation ran separately from timing.
- The GPU was shared, not reserved. Initial utilization was zero with another
  Python process holding memory; that process had exited by the end. This is
  workstation evidence, not an isolated-hardware benchmark.

Temporary scripts and matching `.log` files:
`/tmp/pepsy-cuda-trajectory-review.py` and
`/tmp/pepsy-cuda-trajectory-kernels.py`.

## End-to-end results

Identical seeded random initial MPS arrays (`MPS_rand_state`, seed=932) were
used across backends. Each stream applies H to even sites, disjoint adjacent
CNOTs `(0,1), (2,3), ...`, and amplitude damping with p=0.15 at
`(0, L//3, 2*L//3, L-1)`. Shot seed=173. The number of shots differs between
sizes; compare columns within a row, not timings between sizes.

Median wall times in seconds:

| Sites / chi / shots | Strategy | NumPy CPU, 1 worker | Torch CPU, 1 worker | Torch CUDA, 1 worker | Torch CUDA, 4 workers |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 / 8 / 32 | independent | 0.526 | 0.755 | 1.067 | 1.265 |
| 8 / 8 / 32 | coalesced | 0.0198 | 0.0342 | 0.0486 | 0.0537 |
| 24 / 32 / 8 | independent | 0.410 | 0.635 | 0.827 | 0.895 |
| 24 / 32 / 8 | coalesced | 0.0643 | 0.0993 | 0.1271 | 0.1337 |
| 32 / 128 / 4 | independent | 1.989 | 1.614 | 1.002 | 1.010 |
| 32 / 128 / 4 | coalesced | 0.4988 | 0.4020 | 0.2505 | 0.2556 |

CUDA wins at the largest tested size: approximately 1.61x over Torch CPU and
1.99x over NumPy for independent shots. It loses on the smaller sizes. Four
GPU workers do not improve these cases. Coalescing shares the deterministic
prefix and remains beneficial on CUDA; these streams have only four binary
noise events, so this is favorable to prefix sharing. These measurements do
not locate a universal bond-dimension crossover or cover long branching streams.

## Kraus probability prototypes and profiling

Separate microbenchmark: 16-site random MPS, chi=32, seed=932, complex64 CUDA,
four computational-basis projectors on two sites. Nine warmed timing repeats.
The production helper prepares the normalized amplitude block independently
for each outcome. Two temporary prototypes reuse that block: one still reads
each outcome norm to the host, while the other batches all four matrices and
norms on the GPU and transfers one probability vector.

| Physical support | Current helper | Shared block | Shared block + batched outcomes |
| --- | ---: | ---: | ---: |
| `(6, 7)` | 1.766 ms | 0.595 ms | 0.374 ms |
| `(4, 11)` | 23.507 ms | 6.246 ms | 5.853 ms |
| `(11, 4)` | 23.795 ms | 6.189 ms | 5.877 ms |

The batched prototype is 4.7x faster on the adjacent kernel and about 4x on
the nonadjacent kernels. Dense probability reference errors were at most
1.05e-7; state preservation was checked at atol/rtol=3e-6.

For one already-warmed adjacent event, Torch profiler counted:

| Variant | CUDA kernel launches | Async copies | Stream synchronizations |
| --- | ---: | ---: | ---: |
| Current | 43 | 9 | 9 |
| Shared block | 21 | 4 | 4 |
| Shared + batched | 9 | 1 | 1 |

The prototypes omit general dispatch, represented-norm handling and production
diagnostics; stacked operators were prepared outside the timed kernel. They
demonstrate opportunities, not drop-in implementations or end-to-end speedups.
They preserve projected amplitudes rather than forming cancellation-prone
Kraus Gram matrices. Rare branches, more general channels, exact modes,
gradients and native symmetry still need dedicated integration coverage.

An independently profiled four-shot, eight-site CUDA replay produced 6108
kernel launches, 204 scalar extraction operations and 236 stream
synchronizations. Summed self CPU operator/runtime time was 118.9 ms versus
14.9 ms summed self CUDA time; neither is total application wall time.
Kernel launch CPU time was 28.1 ms. There were 132 QR calls. This supports
reducing small launches and repeated state preparation, not assuming that
matrix throughput or device bandwidth alone limits small trajectories.

## Priorities and API implications

1. **Prototype → implement with regression coverage:** prepare one exact Kraus
   block per event, batch compatible outcome applications/reductions and perform
   one probability-vector readback. Bound batching memory for large outcome
   sets. Preserve rare probabilities, original physical-leg order, untruncated
   routing, backend precision and diagnostics.
2. **Adopt after focused validation:** avoid constructor canonicalization on
   every shot by preparing an isolated template and reusing safe owned-array
   clones. Reuse authoritative norm/center results across branch normalization
   and diagnostics rather than repeating device operations and host reads.
3. **Adopt a measured execution policy:** make automatic workers aware of the
   selected device and workload; one worker is the best tested choice on this
   single GPU. CPU core count is not a GPU concurrency budget. Keep coalescing
   available for small branch trees, with explicit branch-memory limits.
4. **API proposal:** expose resolved device, dtype, workers and strategy, and
   make synchronized benchmark timing/warm-up explicit. Local chunked observable
   reduction remains valuable for releasing final device states without MPI.
5. **Defer:** whole-trajectory CUDA graphs or blanket vectorization. Dynamic
   branches, rank-changing decompositions and host-side decisions need a
   separate design; first batch outcomes or compatible small operations and
   measure the full-run benefit. No such speedup was established here.

## Correctness checks and unresolved failures

For 16 shots of the eight-site circuit, independent and coalesced strategies
were each compared between NumPy and CUDA workers=1/4. Within each strategy,
outcome histories and counts matched. Maximum CUDA-versus-NumPy state-vector
2-norm error was 2.77e-6; maximum CUDA normalization error was 2.99e-7. All
retained tensor arrays stayed CUDA complex64. Large-state runs were timed but
not validated against dense vectors.

Focused existing suite:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m pytest -q -ra -o addopts='' tests/test_mps_gpu_backend.py tests/test_trajectory_importance_regressions.py -k 'cuda or cupy'
```

Result: **16 passed, 2 failed, 48 deselected**. Both failures reproduce the
already-recorded `test_backend_ledger_matches_cpu_and_preserves_state_dtype`
cases `[cuda-False]` and `[cuda-True]`: infidelity differs from the CPU result
by approximately 3.84e-6 and 3.70e-6 against an absolute 3e-6 tolerance. The
preceding dense-state comparisons pass. No tolerances were changed. These
are unresolved numerical diagnostic discrepancies, not a clean GPU suite.
CuPy checks passed in this selection; CuPy throughput was not benchmarked.

Documentation links and `git diff --check` passed. Existing source edits were
preserved. Nothing was staged, committed or published.

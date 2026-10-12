# 2026-09-26 — PEPS sampler performance implementation

Status: implemented and validated in the working tree on `develop`, baseline
`80f451a`. This implements the authorized immediate improvements from the
[performance assessment](peps_sampler_performance_assessment.md). That earlier
note records the pre-change behavior. Existing backend conversion work and
unrelated qMERA/tree changes were preserved. Nothing was committed or published.

## Implemented

- Added `row_cache_max_bytes`, default 64 MiB; zero disables dense transfers.
  Estimate storage before building any row transfer, using source/future bond
  dimensions, bounds on conditioned bonds, dtype size, maximum prefix count,
  and a workspace factor. Uncompressed boundaries use their represented bond
  growth, which can exceed the physical Schmidt rank. Oversized estimates use
  the existing reference-center/reference-prefix path. Existing fragmentation
  and large-future heuristics remain additional routing criteria. The estimate
  is conservative, not a guarantee on process memory or planner workspace.
- Stack the small rhos of active prefix groups at each site. Validate and draw
  together with native Autoray operations; only one validation scalar and one
  integer-choice array are read back per site. Inverse-CDF sampling uses native
  uniforms, handles zero-mass categories, and keeps log-probability updates on
  the backend. Prefix contractions/compression remain separate per group.
- Contract local rhos with public `quimb.tensor.tensor_contract`, copying only
  bra tensors requiring a relabel. Preserve the network exponent explicitly.
  Reuse row-bond metadata, identity future caps, column metadata, and the local
  column contraction when forming its trace. Caches reset on `refresh()`.
- Scale rho before norm/difference operations, preserving the diagnostic
  `norm(rho-rho.H) / max(norm(rho), 1)` without complex64 squaring overflow.
  Invalid traces, non-finite matrices, and genuinely negative diagonal entries
  still fail; only dtype-scale negative roundoff is clipped.
- `row_cache_stats` includes estimate, budget, and routing reason;
  `batch_stats` includes `conditional_batches`. Public result conventions,
  original-PEPS amplitudes, and requested boundary truncations are preserved.
  Grouped RNG sequences can change from previous versions; reproducibility
  within backend/device/method remains tested.

See [implementation](../../../src/pepsy/sampling/samplers.py),
[regressions](../../../tests/test_peps_sampler.py), and
[API documentation](../../api/sampling/samplers.md).

## Environment and upstream decisions

Reused the same-task upstream audit in the
[Torch/Autoray study](peps_sampler_torch_audit.md) and
[backend fix](peps_sampler_backend_fix.md). Rechecked installed versions and
signatures/dispatch for tensor contraction, diagonal/axis reductions, trace,
array conversion, native random uniforms, and JAX source-device contexts.

NumPy 2.5.2; Autoray 0.11.1.dev3+g1b476b305;
Quimb 1.15.1.dev66+ge927f06e1; Torch 2.6.0+cu124; JAX 0.10.2;
Cotengra 0.8.3.dev7+g1d7fd333f; Cotengrust 0.2.1;
Symmray 0.4.1.dev7+g83fb22865. Activated the existing py312 environment with
checkout `src` on PYTHONPATH. No dependency changes or installed-library edits.

- **Adopt:** public tensor contraction/trace, Autoray namespaces and native RNG.
  Torch `asarray` requires a native dtype object when explicitly specifying
  dtype; integer group metadata instead uses the input NumPy int32 dtype
  inference, preserving device selection without a backend-specific branch.
- **Compatibility shim:** retain the previously tested scoped JAX device
  context for array/RNG creation; test its new `random` call on CpuDevice(1).
- **Defer:** compiled sampling loops, native batch-axis boundary/SVD kernels,
  factored dense-row environments, Symmray sampling, and JAX GPU/sharding.

## Before/after measurements

The baseline is the saved source after the prior backend fix and before this
performance implementation. Both classes ran in the same process with identical
state/cutoffs, one BLAS/OpenMP/Torch thread, greedy planning, and Torch inference
mode. CUDA timings synchronized before/after calls. Warm calls alternated old
and new order; profile/readback instrumentation ran separately from timing.

Main state: random 4x4 D=4, physical dimension 2, seed 101;
`sample_chi=16`, `marginal_chi=32`, Quimb future and ket compression, 32 shots.
One warm-up (seed 700), three measured CPU calls (800–802), two CUDA calls
(800–801). Times are medians, with limited repetitions.

| Backend / dtype | Previous batch | Improved batch | Throughput ratio |
| --- | ---: | ---: | ---: |
| NumPy complex128 | 0.761 s | 0.610 s | 1.25x |
| CUDA complex128 | 7.286 s | 5.810 s | 1.25x |
| CUDA complex64 | 1.532 s | 1.176 s | 1.30x |

Production jobs continued during measurements. The RTX A5000 was observed at
99% utilization; contention can vary over time. These are shared-workstation
measurements, not idle-device performance or a controlled precision comparison.
CPU baseline repetitions were 0.748/0.761/0.866 s and new repetitions
0.475/0.615/0.610 s, illustrating timing variability.

For seed 800, direct sampler instrumentation found:

| Readback / allocation operation | Previous | Improved |
| --- | ---: | ---: |
| CUDA complex128 validation-scalar reads | 349 | 16 |
| CUDA complex128 integer-choice transfers | 349 | 16 |
| Final proposal scalar reads | 32 | 32 |
| Final amplitude scalar reads | 32 | 32 |
| NumPy TensorNetwork copy calls | 1,514 | 1,200 |

Upstream compression may perform additional device reads. The table does not
claim all synchronization has been removed. NumPy configurations matched across
timed calls; Torch/JAX configurations can differ because their grouped RNG
algorithm changed. Evaluating the baseline proposal for a selected new
configuration agreed exactly in the comparison probe. Broader numerical
agreement is covered by the focused reference and exact-contraction tests.

### Dense-cache hazard

For 3x3 D=4, the new default routes an eight-shot batch to reference-prefix
before constructing transfers. Its cache estimate is 42,979,034,112 bytes
across possible groups, above the 67,108,864-byte budget. Three warm calls were
0.0556/0.0637/0.0629 s (median 0.0629 s).

The earlier assessment measured the old dense-cache route at 15.934 s and a
largest local transfer of 1 GiB on this state. That expensive old allocation
was not repeated here. A regression makes cache construction fail if attempted,
then checks serial sampling, batch sampling, and proposal evaluation against
an explicitly disabled cache. Separate tests bound actually represented
uncompressed and compressed bonds. Compact row caches remain checked against
full-center contractions for Quimb and DMRG engines.

### JAX limitation

A 2x2 D=2 complex64, eight-shot CPU probe took 10.28 s for the previous first
call and 17.64 s for the new first call. Following calls were
1.240/0.056 s old and 0.069/0.043 s new. Changing group shapes triggers additional
JAX primitive compilation, so averaging those calls into a speedup would be
misleading. Treat cold-start cost as a remaining regression and this as a
correctness/synchronization improvement for JAX, not a proven JAX throughput
speedup. The loop remains eager. Two-device CPU placement was tested; JAX GPU
and production 5x6/high-D throughput were not measured.

## Validation

All commands used the activated project environment. JAX was restricted to CPU;
the focused selection enabled two CPU devices with
`XLA_FLAGS=--xla_force_host_platform_device_count=2`.

- `python -m pytest -xq -o addopts='' tests/test_peps_sampler.py`:
  **50 passed**, including NumPy/Torch/JAX exact/boundary proposals,
  reproducibility, source preservation, converter/gradient preservation,
  non-default JAX placement, zero-mass draws, grouped validation/readback counts,
  large/tiny complex64 diagnostics, and cache-allocation avoidance.
- Then added the uncompressed represented-bond bound regression and ran
  `python -m pytest -q -o addopts='' tests/test_peps_sampler.py -k cache_estimate_bounds`:
  **2 passed**, 50 deselected. Thus 52 focused checks passed across these runs.
- `python -m pytest -q -o addopts='' tests/test_sampler.py tests/test_public_api.py tests/test_package_layout.py`:
  **161 passed, 1 failed**.
- `python -m pytest -q`: **152 passed, 1 failed** (smoke selection).
- Both broader failures are the pre-existing
  `test_package_version_matches_installed_distribution`: installed metadata and
  runtime are 0.4.0 while unchanged pyproject declares 0.5.0. The shared
  environment was not modified to conceal this failure. No full-suite pass is
  claimed.
- `python -m ruff check src tests` and `git diff --check`: passed.
- Repeated the four-path CUDA matrix: exact; Quimb future; DMRG/FIT future;
  Quimb identity future, on Torch CPU complex128 and CUDA complex128/complex64.
  **12/12 passed**, all source arrays unchanged and internal device/dtypes
  retained. Maximum absolute amplitude errors against dense contractions were
  1.47e-14 (complex128) and 1.73e-5 (complex64) for these unnormalized states.
  Each six-site batch transferred exactly six native integer-choice arrays.

Temporary reproduction files/logs are under `/tmp`:
`peps_sampler_performance_comparison.py/.jsonl/.log`,
`peps_sampler_before_performance.py`,
`peps_sampler_performance_cuda_matrix.py/.json/.log`,
`peps_perf_focused.log`, `peps_perf_broader.log`, `peps_perf_smoke.log`, and
`peps_perf_final_ruff.log`. They are diagnostics, not maintained public APIs.

## Remaining limits and use

Use `sample_batch` for multiple shots and inspect `row_cache_stats` when tuning.
Choose sample/marginal cutoffs using accuracy and effective sample size on the
actual state; this task did not change their numerical defaults. Large batches
can still retain many distinct conditioned networks. The cache estimate is not
a total memory limiter. Factored environments and true batched contractions
remain future work; the documented JAX startup regression remains unresolved.

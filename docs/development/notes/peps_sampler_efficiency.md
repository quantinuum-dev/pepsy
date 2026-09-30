# 2026-09-26 — PEPS sampling efficiency and bounded batches

## Scope and status

Implemented in the working tree, not committed or published. This follows the
[runner/sampler review](../../../history/2026-09-26-peps-runner-sampler-review.md)
and preserves the earlier uncommitted scaling, rho-repair, and cache fixes.
No roughening runner integration, production-job restart, dependency change,
or native Symmray support was added.

The source PEPS, original-amplitude target, cutoff policy, row sampling order,
and compression-after-row-projection policy are unchanged. Dense arrays stay
on their backend/device. The default reference proposal remains available.

## Changes

1. `row_cache_mode="factored"`, with a positive `row_cache_max_bytes`, keeps
   local column factors separate and caches right suffixes. Descendants share
   immutable suffixes and update their own left prefixes immediately. There
   is no additional compression. The first row is reused until `refresh()`;
   later rows depend on each incoming sampled prefix. The conservative budget
   estimate includes conditioned boundaries, retained factors and suffixes,
   plus workspace allowance and the persistent initial row. It is not a hard
   cap on process memory or contraction-planner intermediates. Oversized cases
   use the original reference path. Dense transfers and zero-budget defaults
   retain their previous selection behavior.
2. `sample_batch(..., chunk_size=N)` bounds live prefix groups by N and collects
   the output. `iter_samples(..., chunk_size=N)` also permits streaming/discarding
   output. Both use one continuous backend RNG and reproduce one another for
   the same seed/chunk size. Changing chunk size can change draw order. The
   collected API aggregates rho maxima/counts and group statistics across
   chunks; an iterator exposes each chunk's diagnostics. Torch diagnostic
   aggregation detaches scalar records so maxima do not retain earlier chunks'
   autograd graphs. Source tensors and probability calculations are unchanged.
3. Exact amplitudes reuse one public Cotengra contraction tree. Each physical
   tensor slice is rescaled once and cached with its log scale; selected slices
   are contracted with exponent stripping and zero checks. This preserves phase
   and physical scale even beyond the raw dtype's range. The cache retains at
   most one additional ket's array data, small scale arrays, and a plan, not
   configurations or numerical amplitude results. `refresh()` clears it.
   Exact contraction cost is still uncapped by the proposal chis. Numerical
   stabilization has overhead; no universal amplitude speedup is claimed.
4. `PEPSSampleResult.normalized_weights`, `.effective_sample_size`, and
   `.weight_diagnostics` provide stable importance diagnostics. Invalid,
   empty, or all-zero weight sets raise; zero target amplitudes have zero
   weight. `sampler.diagnostics` summarizes rho defects and repairs;
   `amplitude_stats` exposes plan reuse. Neither ESS nor small repairs certify
   support, and per-chunk normalized weights cannot be combined as global
   normalized weights. No generic error bar or support guarantee is invented.
5. [Benchmark](../../../benchmarks/peps_sampling.py) builds an evolved diagonal
   product-wall PEPS using public simple-update gates. It reports setup,
   repeated batch timings, a separate synchronized stage profile, CUDA peak
   allocated bytes, process-lifetime peak RSS, and importance diagnostics.
   At most 16 sites also get a full dense amplitude oracle for the same evolved
   PEPS. This validates sampling that state, not the accuracy of truncated
   evolution against the exact physical dynamics.

## Measurements

Existing CPU exact evolution and GPU DMRG continued throughout. CPU runs used
one BLAS/OpenMP thread; GPU probes were small and time-limited. GPU timings
include contention and cannot establish isolated-device speedups. All reported
proposal paths used Quimb future boundaries, automatic cutoffs, greedy exact
contraction, and explicit `rho_positivity="absolute"`. Five dt=0.2 evolution
steps used D as shown. Three warm CPU repeats and two GPU repeats were taken.

Final median batch times (seconds):

| State / backend | Shots / chunk | χ / χ′ | Reference | Factored |
| --- | --- | --- | ---: | ---: |
| 4×4 D=4, NumPy complex128 | 32 / 8 | 16 / 8 | 0.800 | 1.065 |
| 8×4 D=2, NumPy complex128 | 8 / one batch | 8 / 4 | 0.553 | 0.364 |
| 4×4 D=4, Torch CUDA complex64, shared GPU | 8 / 4 | 16 / 8 | 1.851 | 1.761 |

The wider CPU case improved by about **1.52×**. An earlier run gave 0.415 versus
0.276 seconds, the same qualitative improvement. The 4×4 case regressed because
building suffixes and updating prefixes outweighed cheaper local rho
contractions. Factored caching therefore remains opt-in. With 32 simultaneous
4×4 prefixes the estimated factored cache exceeded 64 MiB and correctly fell
back; chunking to eight enabled it within budget.

For the final 4×4 CPU profile, local rho time decreased from 0.274 to 0.202 s,
but row preparation and prefix updates added 0.346 and 0.166 s. Boundary updates
cost 0.290 versus 0.266 s. Exact amplitude work was about 0.023 versus 0.022 s.
These are a separate instrumented call, not a decomposition of the median
repeat. The initial cProfile baseline likewise identified local contractions
and conditioned-boundary updates as the main costs.

Earlier Torch CPU complex64 4×4 measurements before caching amplitude slices
were 0.823 s reference and 0.955 s factored (32 shots, chunks of eight).
The first shared-GPU probes were 1.126 versus 1.750 s; later shared-GPU timing
reversed that ranking. There is **no reliable GPU speedup claim**. In the final
GPU profile, boundary updates cost about 0.76/0.73 s and exact amplitude work
0.26/0.24 s. Validation/draw synchronization and many small kernels also matter.
Native batched boundary evolution remains a separate, deferred optimization.

Final GPU peak allocated memory was approximately 10.4/10.9 MiB for these small
sampler benchmarks, including their live state allocations. This excludes other
processes and CUDA-reserved/context memory; it is not total GPU usage. Source
amplitudes agreed with the independent dense oracle within 1.92e-8 on CUDA
complex64 and 2.95e-17 on the 4×4 NumPy complex128 probe. Maximum sampled log-q
versus log-Born differences were about 3.92e-4 and 2.05e-4 respectively: finite
caps remain approximate even with high observed ESS.

Raw JSON and logs are under `/tmp/peps-eff-*.json` and `/tmp/peps-eff-*.log`.
Reproduce, from an activated development environment, for example:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/peps_sampling.py --shape 8x4 --bond 2 --chi 8 \
  --chi-prime 4 --samples 8 --row-mode factored \
  --output /tmp/peps-wide-factored.json
```

For an isolated GPU comparison, use an idle device and the same state/dtype,
shot count, chunk size, and seed. Do not infer throughput at production D or
8192 shots from these probes.

## Upstream audit and decisions

Installed versions: Quimb 1.15.1.dev66+ge927f06e1, Autoray
0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1,
Symmray 0.4.1.dev7+g83fb22865, NumPy 2.5.2, Torch 2.6.0+cu124, JAX 0.10.2.

Checked the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The Symmray abelian-arrays documentation endpoint returned an internal error;
repository and installed implementation remain the fallback. This task keeps
Symmray explicitly unsupported in the direct sampler.

**Adopt:** public `TensorNetwork.contraction_tree` and
`ContractionTree.contract(arrays, strip_exponent=True, check_zero=True,
backend=..., autojit=False)`, confirmed against installed signatures/source.
The tree's tensor ordering is preserved when projecting cached array slices.
Public Quimb contractions handle factored prefix/suffix networks. No installed
library code, dispatch registry, or dependency policy was changed. Explicit
Pepsy cutoff modes remain intact despite newer upstream default changes.
**Defer:** native shot-axis boundary batching, automatic cache tuning,
approximate amplitude contraction, and general support certification.

## Validation

Final results are recorded in the
[session handoff](../../../history/2026-09-26-peps-sampler-efficiency.md).
New tests cover backend-preserving factored proposals, exact dense probabilities,
strongly truncated repaired proposals, budget fallback, cache refresh, rare
prefixes, streamed/collected replay, prefix bounds, Torch graph retention,
zero/extreme amplitudes and phase, exponent metadata, plan reuse, and stable
weights with invalid-input behavior. Existing 4×4 integration checks and other
sampling regressions remain part of domain validation.

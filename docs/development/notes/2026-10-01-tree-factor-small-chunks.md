# 2026-10-01 — TreeSampler factor strategy at small chunks

Scope: verify the conversation's recommendation that `strategy="factor"`
remains useful with small chunks and uses less memory. Baseline: `develop`
at local commit `608b664`. This is **measured** synthetic evidence, with no
implementation change or default promotion. The earlier recommendation was
too broad: neither faster sampling nor lower memory is universal.

## Method and environment

Used the existing Python 3.12 environment and the unchanged dependency versions
recorded in the [correctness audit](2026-10-01-tree-sampler-correctness-resume.md#upstream-and-environment-audit).
NumPy is `2.5.2` and cupy-cuda12x is `14.1.1`. The backend was native CuPy,
complex128, on one NVIDIA RTX A5000 with 24,564 MiB device memory. These are
sampling measurements without an autograd graph; Torch memory is unverified.

Constructed 30-site trees using `TreePlan.from_order(range(30),
structure="balanced", max_arity=2, top_arity=3)`. Each bond dimension was
`min(chi, 2**min(subtree_sites, 30-subtree_sites))`; actual maximum bonds were
32 and 256. Tensor entries were complex Gaussian values from NumPy seed 19,
divided by the square root of each tensor's element count. Canonicalized the
tree around its root before sampler construction. The production checkpoint
was not available.

Each timing used a fresh `TreeSampler(..., backend="native", threads=1)` with
the indicated strategy and chunk, default 512 MiB workspace, and either
default 128 MiB cache or `cache_bytes=0` (`factor_no_cache`). State construction
and sampler capture were excluded. Warmed with eight samples, seed 11, then
timed `sample_arrays(count, seed=2)` with CUDA stream synchronization before
and after. Two timing repeats interleaved the strategies within each chunk.
Shell thread settings were `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.

Memory was measured in separate sampling calls using the public
[CuPy allocator context](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.using_allocator.html)
and a wrapper around the default pool's allocator. At each allocation, recorded
the maximum `pool.used_bytes()` and subtracted its pre-call live baseline.
Unused pool blocks were released first. This measures peak **additional live
CuPy allocations** during sampling, including scratch, cache, random draws and
outputs. It excludes the resident source/captured tensors, CUDA context/library
allocations and other processes. It is not total application memory, reserved
pool capacity or `nvidia-smi` usage; see the official
[CuPy memory guide](https://docs.cupy.dev/en/stable/user_guide/memory.html).
The common live baseline was 549,579,776 bytes at chi 256 and 2,747,392 bytes
at chi 32. One sampler was resident at a time.

The GPU was shared: another Python process held approximately 18,394 MiB when
the standard chunk-1,000 run failed. Activity from that process was not
controlled. Timings are indicative observations with two repeats, rather than
isolated performance guarantees. Live allocator measurements are scoped to
the benchmark process.

## Results

| Maximum bond | Samples | Chunk | Strategy | Two times (s) | Median (s) | Extra peak (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 1024 | 16 | standard | 12.687, 12.998 | 12.842 | 2118.05 |
| 256 | 1024 | 16 | factor | 11.523, 13.090 | 12.307 | 776.36 |
| 256 | 1024 | 16 | factor, cache disabled | 11.489, 13.043 | 12.266 | 770.48 |
| 256 | 1024 | 128 | standard | 3.473, 3.513 | 3.493 | 2570.01 |
| 256 | 1024 | 128 | factor | 2.505, 2.482 | 2.493 | 1007.02 |
| 256 | 1024 | 128 | factor, cache disabled | 3.087, 2.999 | 3.043 | 949.30 |
| 32 | 1024 | 16 | standard | 5.362, 5.436 | 5.399 | 34.51 |
| 32 | 1024 | 16 | factor | 5.992, 7.086 | 6.539 | 33.76 |
| 32 | 1024 | 128 | standard | 0.719, 0.795 | 0.757 | 45.09 |
| 32 | 1024 | 128 | factor | 0.763, 0.791 | 0.777 | 266.59 |
| 256 | 8192 | 1000 | factor | 4.221, 4.176 | 4.199 | 2139.31 |

All variants at a given chunk produced identical configurations for the same
draws, and sampled probabilities agreed at `rtol=2e-10, atol=1e-22`. Each
timing/profile call also checked the first 16 returned probabilities through
the separate bottom-up `sampler.probabilities()` path. This checks numerical
consistency on these trees, rather than a full distribution audit. Outputs
were not compared across different chunks in this harness. The factor-only
8,192-shot run matched itself across repeats/profile and passed the separate
scoring check; no standard comparison completed at that chunk.

Observed kernel batches equaled the requested chunks: 64 batches of 16 or
eight batches of 128 for 1,024 samples; eight batches of 1,000 plus 192 for
8,192 samples. The factor cache contained 6,196,176 bytes just before cleanup
at chi 256 / 1,024 samples, and 12,080,904 bytes at 8,192 samples. At chi 32
the cache was empty. Cache-disabled runs retained zero bytes. The high chi-32
factor peak at chunk 128 therefore is not explained by a retained cache.

Standard at chi 256 / chunk 1,000 / 1,024 samples failed before completing its
first timing: CuPy requested 1,048,576,000 bytes with 4,928,837,120 bytes
already allocated by its pool. The other process remained resident. This
establishes an OOM under that shared load, not failure on an otherwise free
24 GiB GPU. No timing or complete peak was reported for that attempt.

## Interpretation and status

- At chi 256, factor reduced measured extra peak by about 61–63% at chunks
  16 and 128. At chunk 16 timing was nearly tied and varied across repeats;
  at chunk 128 factor was about 1.4 times faster by these medians.
- At chi 32, factor was slower at chunk 16. At chunk 128 the timing was nearly
  tied, but factor used about 5.9 times as much extra live memory. Small chunk
  size alone does not establish a factor benefit.
- Disabling the cache saved approximately 6 MiB at chi 256 / chunk 16 and
  58 MiB at chunk 128, while the latter median time rose from 2.49 to 3.04 s.
  A zero cache budget is an explicit memory/performance tradeoff, not a
  generally better setting. The workspace target is not a total-memory cap.
- Keep the [documented standard default](../../api/sampling/tree.md). Factor
  remains an opt-in candidate for large-bond GPU trees, with the actual tree,
  sample count, chunk and backend measured before choosing it. **Defer**
  default promotion, production-checkpoint and Torch/autograd comparisons.

Temporary harness and raw results: `/tmp/pepsy_tree_small_chunks.py`,
`/tmp/pepsy_tree_small_chunks_chi256_1024.{log,json}`,
`/tmp/pepsy_tree_small_chunks_chi32_1024.{log,json}` and
`/tmp/pepsy_tree_small_chunks_chi256_8192.{log,json}`. The chi-256/1,024 JSON
retains the completed chunk-16/128 rows despite the later chunk-1,000 failure.
The table and method above retain the essential findings if temporary files
disappear. No installed library, production algorithm, dependency or test was
changed. Documentation link and whitespace checks are recorded in the
[handoff](../../../history/2026-10-01-tree-factor-small-chunks.md).

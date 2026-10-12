# 2026-09-30 — Deeper TreeSampler algorithm review

Scope: the user requested a further review for substantially faster tree
sampling. This continues the [integrated shared-density optimization](2026-09-30-tree-shared-density.md).
Production source was not changed during this review. The new algorithms
below are isolated prototypes under `/tmp`, not integrated library features.

## Exact algorithm opportunities

1. **Carry vectors for structurally pure environments.** The root starts
   pure. When all other children of a node with a pure incoming environment
   have been measured, its remaining child also has a pure environment.
   Carry its amplitude vector instead of forming a chi-by-chi outer product.
   Tracing an unmeasured sibling can produce a mixed environment; retain the
   density route there. No purity assumption is made from a numerical cutoff.
2. **Retain known factors instead of immediately squaring them.** An incoming
   density can already be available as an amplitude factor L satisfying
   `rho[a,A] = sum_r L[r,a] * conj(L[r,A])`. Transfer the factor through a node
   T first: `Y[r,c,F] = sum_a L[r,a] * T[a,c,F]`, then form only the smaller
   child density by summing `Y[r,c,F] * conj(Y[r,d,F])` over r and F. This
   avoids constructing a large parent density merely to contract it again.
   The prototype retains suitable factors and tiles factor transfers over
   shots, targeting 512 MiB for the projected intermediate. This is not a
   bound on total device memory or all layout copies.
3. **Share exact measurement prefixes.** At a fixed node, identical already
   sampled configurations imply the same conditional environment. Use exact
   mixed-radix integer keys, evaluate unique prefixes once, then gather back
   to the original shot order. Preserve each shot's original uniform draws.
   This does not coalesce independent random choices.
4. **Cache small outputs across chunks, and group large root contractions.**
   Cache the resulting child densities, not the quartic transfer operators.
   The prototype's cache is scoped to one sampling call with a 128 MiB
   retained-data budget. It also avoids repeating the large contraction of
   the first sampled root subtree into the root tensor for identical
   prefixes within a chunk. No truncation, dtype change, or approximate
   rank reduction is introduced.

Canonical-center handling is already effective and is not the limiting
algorithm here. The useful analogy with MPS sampling is to carry the
smallest exact conditional representation, while accounting for a tree's
unmeasured sibling branches.

## Measured comparison

Same synthetic state builder as the preceding audit: 30 sites, balanced
branches, three root children, actual chi=256, complex128, native CuPy,
8,192 samples, chunk_size=2048, state seed 19, and sample seed 2, on the
available RTX A5000. Timings synchronize the GPU around the complete sampling
call and exclude state construction/capture. No competing benchmark ran.
The production 5×6 checkpoint was not supplied.

| Algorithm | Sampling time |
| --- | ---: |
| Current integrated shared-density sampler | 28.90 s |
| Structural pure-vector propagation | 17.82 / 17.88 s |
| Pure vectors plus grouping full configuration rows | 20.03 / 20.05 s |
| Pure vectors plus compact integer-key grouping | 9.88 / 9.85 s |
| Pure vectors plus integer keys and cross-chunk cache | 8.38 / 8.40 s |
| Retain mixed factors as well | 5.29 s |
| Also group the large root contraction | **3.291 / 3.293 s** |

Thus the fullest prototype is approximately **8.8 times faster** than the
current integrated sampler on this state. This is a workload-specific
measurement, not a general or production speed guarantee. Naive grouping
of full rows was slower than the pure-vector route despite fewer distinct
histories; representation of the grouping keys matters.

The expensive cached node saw 485, 495, 475, and 494 distinct prefixes in the
four chunks, but only **737 distinct prefixes across the complete call**.
The retained cache reached **12,080,904 bytes (11.52 MiB)**. The budget bounds
retained key/value storage, not temporary sorting/concatenation/gather copies.

All 8,192 configurations matched exactly between the compared three-child-root
implementations. All returned probabilities passed the reference comparison
at rtol=1e-10, atol=1e-22; independent bottom-up scoring of 16 configurations
for the fullest prototype had maximum relative error 1.29e-14.

An initial combined benchmark hit CuPy OOM on a subsequent binary-root case.
In a fresh process that geometry completed in **11.25 s** at chunk size 2048
and **11.19 s** at chunk size 512, with identical configurations and agreeing
probabilities across chunk sizes. Independent scoring errors were at most
2.13e-14 relative. The earlier baseline binary-root audit was interrupted,
so no completed baseline speedup is claimed for this geometry. The OOM's
exact cause was not established; allocation lifetime/peak-memory validation
remains necessary before integration.

## Correctness and lifecycle probes

- Pure-vector and within-chunk grouping prototypes: **192 small sampling
  calls** across NumPy, Torch CPU/CUDA, and CuPy; complex64/complex128;
  ternary local physical dimension; physical root present/absent;
  chunked/unchunked sampling; explicit and persistent RNG; single shots.
  Seeded configurations matched the current sampler, and probabilities
  matched an independent dense statevector. Maximum absolute error across
  these dtypes was 6.14e-9.
- Cross-chunk cache: **32 NumPy cases**, including a one-byte budget forcing
  fallback, persistent RNG, source replacement plus refresh, and cache
  release after each call. Dense checks passed; maximum error 3.08e-9.
- Factored cache: **32 further NumPy cases** with the same checks and a
  1 KiB projected-intermediate target forcing shot tiles. Dense checks
  passed; maximum error 5.28e-9.
- Larger CuPy checks are the comparisons above. The fullest factor/cache
  prototype is currently NumPy/CuPy only; its Torch/device/gradient behavior
  is not validated by the simpler prototype's 192 calls.
- **Additional existing lifetime finding:** `TreeSampler._amplitudes` leaves
  its recursive visitor's self-reference intact. With cyclic GC disabled,
  discarding a sampler after `amplitudes()` leaves a weak reference alive;
  `gc.collect()` releases it. This can retain sampler snapshots/configurations
  until cyclic collection. It is a demonstrated retention issue, but was
  not proven to cause the combined benchmark's OOM. The sampling visitor
  already clears its analogous cycle. No lifecycle fix was made here.

## Integration recommendation and limits

**Prototype, not integrated:** combine explicit vector/factor/density
representations with compact prefix grouping and a bounded per-call cache.
Promote this as a focused sampler change after production-quality capability
and memory handling, rather than maintaining copied recursive functions or
the prototype's temporary instance flags.

Before integration, provide exact fallbacks when mixed-radix keys cannot fit
in signed 64 bits, keep all allocations inside the captured device context,
cover Torch gradients or bypass unsafe graph-retaining caches, enforce
memory limits for all large intermediates, and test exception/concurrent-call
scratch-state lifetimes. Do not infer multi-GPU support from GPU 0 checks.
The prototype uses parent dimension 64 as a profiling heuristic; a maintained
implementation should select useful sharing from contraction cost and cache
size rather than assuming every grouping operation helps.

**Adopt/retain:** the existing canonical-region contract and native operations.
Reused the unchanged environment/upstream audit from the preceding notes;
no dependency or installed-library edits. **Defer:** changing sampling order,
changing tree geometry, lowering precision/chi, or approximating ranks.

Temporary reproducibility files: `/tmp/pepsy_tree_pure_prototype.py`,
`/tmp/pepsy_tree_pure_checks.py`, `/tmp/pepsy_tree_cache_checks.py`,
`/tmp/pepsy_tree_factor_checks.py`, `/tmp/pepsy_tree_pure_benchmark.py`,
`/tmp/pepsy_tree_pure_repeats.py`, `/tmp/pepsy_tree_factor_benchmark.py`, and
`/tmp/pepsy_tree_binary_factor_probe.py`, with corresponding JSON/log results.
The frozen starting implementation is `/tmp/pepsy_tree_sampler_shared_only.py`.
Repository numerical tests were not rerun for this audit-only turn; the
previous integration checks remain dated evidence. Documentation links and
whitespace were checked separately.

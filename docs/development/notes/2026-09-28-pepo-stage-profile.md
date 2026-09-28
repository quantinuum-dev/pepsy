# 2026-09-28 — Located PEPO construction stage and memory profile

## Scope and method

The model matches the prior [spatial-reuse measurement](2026-09-28-cluster-spatial-reuse.md):
5×6 open square, order 4, 30 located X terms at 0.2, 49 located ZZ edges
at 0.7, step `-0.01j`, and NumPy complex128. `factorization="fixed"`
uses exact Pauli histories on this located route; its factorization stage
is a Pauli-basis expansion, with no numerical SVD. Structural compilation
was completed before timing. Five alternating, warmed evaluations compared
the previous per-placement lower contraction with the new symmetry-cached
lower contraction in one process. `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1`; CPU only, on a shared machine with other jobs active.
The baseline method was reconstructed from the immediately preceding local
source patch for the timing probe; it was not installed as another API mode.
Timers wrap the target, lower-support contraction, Pauli expansion and sparse
block insertion calls. The tree timer contains both expansion and insertion,
so its time must not be added to those nested rows.

| Stage | Previous median | Cached median | Calls previous → cached |
| --- | ---: | ---: | ---: |
| Exact local targets | 0.00370 s | 0.00371 s | 6 → 6 representatives |
| Lower-support contraction | 0.69325 s | 0.01407 s | 462 → 5 |
| Pauli expansion | 0.41459 s | 0.34774 s | 462 → 462 |
| Sparse `_add_block` insertion | 0.08977 s | 0.08739 s | 28,440 → 28,440 |
| Tree routine, including expansion/insertion | 0.60557 s | 0.52399 s | 462 → 462 |
| Complete evaluation | **1.31325 s** | **0.55006 s** | — |

The lower contraction was the largest measured stage and its 49/118/295
order-two/three/four placements reduced to 1/1/3 representative evaluations.
This removes about 0.68 s of the complete 0.76 s median reduction. The Pauli
expansion is now the largest measured stage. Stage timings are scoped to this
CPU/model and are not additive because wrappers are nested and timers include
small dispatch overhead.

Fresh-process peak RSS was about **0.223 GiB** before and after caching;
the five-pair warmed comparison reached **0.233 GiB** as process caches
accumulated. The final sparse active blocks contain **28,470 blocks** and
**1,822,080 bytes** of block data. The object's `dense_nbytes` estimate is
**1,288,251,015,552 bytes** (about 1.17 TiB); that dense PEPO was not
materialized. RSS includes Python, NumPy and static caches and is a process
peak, not a per-stage allocation attribution.

## Correctness and implementation

The finite labeled-graph plan already proves that exact local targets agree
under a site permutation. The completed lower-order PEPO restricted to a
support is equivariant under the same term, coefficient, factor-order and
geometry mapping: every lower connected subcluster maps to the corresponding
subcluster on the target support. For each order, the new implementation
contracts the frozen lower active network once per representative, transports
the resulting operator with `permute_operator`, and retains all placed
residual subtractions and block insertions. Only current-call arrays are
cached; independent coefficient overrides continue to use their separate
plans.

An independent pre-change probe compared every lower contraction under its
plan mapping on 5×6; maximum matrix entry difference was
`5.56e-16` at order four. The complete optimized 5×6 active blocks were
compared key-by-key with `spatial_reuse=False`: all **28,470** block keys
matched and the largest entry difference was **3.19e-16**. A focused 2×2
regression compares dense operators and Torch coefficient/time gradients,
and checks lower contractions reduce from nine to three. Existing cluster
suites also check located, mixed-product and override behavior.

No dependency or installed library changed. The earlier
[upstream capability audit](2026-09-28-cluster-spatial-reuse.md#upstream-capability-audit)
remains applicable. This change **adopts** the existing verified structural
plan and backend-preserving axis transport; it adds no compatibility shim or
numerical approximation.

A separate warmed, instrumented evaluation measured per-call Python/NumPy
allocation peaks with `tracemalloc` (its tracing overhead is excluded from
the timing table). These are maximum **single-call transient traced
allocations**, not sums or process RSS:

| Wrapped stage | Calls | Maximum traced transient allocation |
| --- | ---: | ---: |
| Exact-target routine | 4 order-level calls (six representative evaluations) | 0.551 MiB |
| Lower-support contraction | 5 | 1.270 MiB |
| Pauli expansion | 462 | 2.006 MiB |
| Sparse block insertion | 28,440 | 0.070 MiB |

In that traced process, RSS rose from 227.2 MiB before evaluation to a
236.9 MiB process peak. The peak was first observed during the Pauli
expansion/insertion stage; the tracer itself, retained compiled data and
allocator high-water effects are included. This per-call allocation profile
does not imply that the four stage peaks add to the final active-block size.

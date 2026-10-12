# 2026-09-29 — Exact frontier preparation for cluster MPOs

This extends [symbolic channel sharing](2026-09-29-symbolic-cluster-channels.md)
and addresses its complete-collection enumeration bottleneck. Pepsy owns the
implementation; Gaugy remains a caller. No fermionic scope was added.

## Exactness and the two different errors

Full channels reproduce the chosen finite-cluster expansion, up to floating
point arithmetic. They are not intrinsically approximate. A finite cluster
cutoff can still differ from the full global ordered exponential; a smaller
numerical channel cap adds a second, separate error. Removing redundant
histories exactly changes neither error. No construction can guarantee an
arbitrary exact operator at an arbitrarily small chi.

## Implemented frontier construction

`prepare_cluster_channels(..., preparation="frontier")` stores the selected
clusters crossing each MPO cut. At a site already occupied by an active
cluster, that cluster's local factor advances. Otherwise the transition uses
the singleton residual or starts one compatible cluster whose minimum site
is the current site. Actual support masks decide compatibility; overlapping
chain spans alone do not forbid disjoint graph clusters. Clusters leave the
state after their maximum site.

Every complete disjoint collection has exactly one path, obtained by starting
each member at its first site. Forgetting a finished member is safe because
no later-starting cluster can intersect its support. Completed choices
therefore share a continuation without ever being listed as full collections.
For active set F at cut i, the local factor indices contribute a product of
ranks. Before additional symbolic sharing, the exact virtual dimension is

```text
D_i = sum over reachable F [ product over C in F r(C,i) ].
```

The empty set contributes the singleton rail. Four active sets need not mean
bond dimension four: two crossing clusters of rank four give
`1 + 4 + 4 + 16 = 25` channels. Those channels are constructed directly.

`_cluster_channel_frontier` prepares reachable sets, local transitions, native
charges and sparse reference tensors. It does not invoke the original source
MPO assembler or global collection planner. The existing symbolic quotient
operates on those transitions. QR optionally reduces them further. Replay
contracts each local factor and its carried wires into the final channel
spaces; it forms no expanded collection product core and performs no SVD/QR.
The public trace still calls `exp()` and contracts the constructed MPO.

Supported targets are fixed interval MPOs, fixed exact direct graph MPOs
without a collection-order cap, and uncapped recursive MPOs. The explicit
option rejects bounded/auto graph targets rather than silently promoting a
previously approximate target to an exact one. `frontier_state_budget` limits
active sets per cut; None inherits the source's assembly-state budget. The
plan-level override does not mutate the source. `collection_budget` is
irrelevant on this path. Exceeding a state or memory budget raises, without
silently discarding terms. Default reference preparation and PEPOs are unchanged.

## Reproducible bounded measurement

Run [the benchmark](../../../examples/frontier_cluster_channels_benchmark.py)
with the shared environment and single BLAS/OpenMP threads:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python examples/frontier_cluster_channels_benchmark.py
```

Each four-site group contains disjoint pairs `(0,2)` and `(1,3)`, repeated
along the chain. Ordered factors are `a sum ZZ` then `b sum X`; cluster size
two captures each independent pair exactly. Reference `(a,b)=(.3,.2)`, held-out
`(.31,-.19)`, step `-.1j`, full channels and symbolic sharing enabled in both
methods. Measurements use CPU Torch complex128 on a Xeon W-3375; no compiler
speedup is claimed. Forward/backward measures the actual normalized MPO trace.

| sites | nonempty complete collections | preparation ms, reference → frontier | tracked preparation peak bytes, reference → frontier | forward/backward ms, reference → frontier |
| ---: | ---: | --- | --- | --- |
| 4 | 3 | 18.5 → 6.1 | 339,618 → 154,174 | 13.8 → 10.9 |
| 8 | 15 | 65.7 → 8.2 | 2,096,606 → 273,906 | 42.3 → 15.0 |
| 12 | 63 | 337.2 → 12.3 | 19,420,926 → 392,718 | 170.1 → 22.1 |
| 40 | 1,048,575 | budget refusal → 38.1 | not run → 1,227,162 | not run → 71.4 |

Preparation timing is one unprofiled call; memory is a separate warm-source
preparation measured with Python `tracemalloc`. The allocation peak includes
tracked Python/NumPy allocations, not opaque backend workspace or total RSS.
Forward/backward is the median of five calls after two warmups. This is a
favorable disjoint-support case and one machine measurement, not a bound for
dense-connectivity graphs or a universal speedup.

Reference pre-sharing maximum bonds grow 25, 100, 400 across the small cases;
the frontier keeps 25. After symbolic sharing the old path reaches 25, 26, 27,
but still schedules 16, 128, 768 projection contractions. Frontier preparation
emits 12, 24, 36, and 120 contractions for 4, 8, 12, and 40 sites. This shows
why eliminating completed histories before assembly helps beyond shrinking
final bond dimensions.

The analytical normalized trace is
`[cos(.1*a) * cos(.1*b)**2]**(N/2)`. NumPy agrees to roundoff. Torch's existing
local exponential path has a small accumulated value bias: 1.6e-12 at four
sites and 1.6e-11 at forty sites. The old assembler has the same four/eight-site
bias; frontier and old results differ at roundoff. Gradient norm errors range
from 1.2e-14 to 1.5e-12. This was investigated before choosing the large-case
tolerance; no construction mismatch was waived.

## What full symbolic minimization would additionally require

The finite-automaton interpretation is described by
[Crosswhite and Bacon](https://arxiv.org/abs/0708.1221). Frontier sharing reduces
which states are generated. It does not prove that the resulting weighted
automaton has the smallest possible number of channels.

A stronger next reduction would replace continuation operators that are
linear combinations of others. [Hubig et al.](https://arxiv.org/html/1611.02498#S6)
distinguish deparallelisation from this stronger delinearisation. Their
numerical method does not supply a proof of symbolic parameter-family identity
for this API. Applying that idea safely here requires:

1. A declared symbolic coefficient algebra that keeps independent bindings
   distinct and preserves ordered-factor expressions.
2. Exact linear-dependency proofs within each charge block, rather than rank
   estimates from a single numerical reference. Symbolic canonicalization can
   reuse shared expressions, but arbitrary user callbacks remain opaque.
3. Endpoint maps valid throughout the parameter family. Parameter-dependent
   pivots/divisions can become singular at zero parameters; a differentiable
   fixed plan must preserve those directions or use a proven nonsingular chart.
4. Independent operator and derivative checks after every additional reduction.

These are proposed further algebraic work, not implemented or a guarantee of
small exact chi. Current symbolic atoms deliberately treat residual entries
independently. Frontier width, local dense cluster exponentials, and fixed
factor ranks can still grow rapidly. PEPO loops and branches need their own
connectivity construction and do not inherit this MPO frontier rule.

**Upstream decision:** adopt Pepsy's existing fixed factors, native charge
metadata, sparse grouping and Cotengra contraction tape. Reuse the same-task
upstream audit; no dependencies or installed libraries changed. Validation
and working-tree state are in the [handoff](../../../history/2026-09-29-frontier-cluster-channels.md).

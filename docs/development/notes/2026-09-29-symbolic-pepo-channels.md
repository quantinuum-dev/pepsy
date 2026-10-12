# 2026-09-29 — Exact PEPO edge sharing before reference preparation

This extends the fixed compact channel workflow with opt-in
`preparation="symbolic"` for dense spin graph/square PEPOs. Pepsy owns the
implementation; Gaugy forwards `preparation` and `structural_reuse`. Default
reference preparation and the ordinary source `exp()` target are preserved.
The [API](../../api/operators/cluster_channels.md) describes usage.

## Implemented construction and exactness

1. Give every local residual matrix entry its own formal atom, independent
   of parameter values or equality between coefficient bindings.
2. Build fixed local tree factors once, retaining symbolic nonzero blocks.
   A four-site chain removes 48 formally zero identity blocks. Numerical
   reference zeros are never a reason to drop a block or derivative.
3. Sweep graph edges and compare complete endpoint slices: every other
   virtual leg and both physical indices are part of the signature. Select
   one copy at an identical-slice endpoint, sum the opposite endpoint, and
   repeat until no more channels merge. Preserve the singleton rail.
4. Route the smaller graph wires onto the square lattice, then perform the
   same exact slice quotient on square edges. Route labels use mixed-radix
   arithmetic, avoiding Cartesian label dictionaries.
5. Compile constant linear gathers for reference and tangent snapshots.
   Additional samples evaluate residuals and these gathers, without another
   expanded numerical PEPO construction or routing pass. Optional host QR
   selects smaller spaces only after exact sharing.
6. Fold graph quotient maps into the existing square/graph projection
   constants. Live replay constructs compact tensors from local factors
   using the existing contraction tape, with no additional graph-map
   contractions and no SVD/QR. `trace_exp` constructs and traces that PEPO.

The edge identity is local: if endpoint slices `A[q]` are identical for all
`q` in a group, then `sum_q A[q] B[q] = A[q0] sum_q B[q]`. Since the slice
includes every remaining index, the identity holds in any surrounding
network, including loops, branches and parallel square bonds. It does not
assume MPO suffix grammar or delete loop interactions from local residuals.
The source still uses whole ordered local exponential products and connected
subtraction on the union interaction geometry.

Every reference snapshot is discarded after preparation. Plans retain only
static layouts, linear-gather indices/weights, sum/select maps and contraction
constants. Full channels reproduce the chosen finite cluster expansion for
arbitrary independent residuals; they need neither QR nor SVD. Finite spatial
cluster error relative to the full global exponential/product is separate
from any additional error imposed by a channel cap.

## Measured preparation and replay

Reproducer: [symbolic_pepo_channels_benchmark.py](../../../examples/symbolic_pepo_channels_benchmark.py).
[All 48 result rows](2026-09-29-symbolic-pepo-channels.jsonl) retain one/six
references, caps 1/4/full, both preparation modes, exact bond dimensions,
operator/gradient errors and timing/storage fields.

```bash
source ~/envs/py312/bin/activate
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  python examples/symbolic_pepo_channels_benchmark.py --repeats 5
```

These are four-site square PEPO controls, with two ordered factors: XZ pairs
(or one XYZ three-body term) and onsite X. Branch order is 4, loop/higher-body
order is 3, crossing order is 2. Crossing uses a 2x2 periodic lattice. Local
step is `-.1j`; the primary reference is `(a,b)=(.2,.3)`. Five extra references
are `(i*.1, .1-i*.03)` for i=0..4. Held-out values are `(.31,-.17)`.
Residual geometry is warmed; preparation timings are medians of five runs.
The final benchmark ran after test processes finished. Preparation peaks use
tracemalloc in a separate run: Python/NumPy allocations, **not** process RSS,
CUDA memory, total backend workspace or backward memory.

Six-reference preparation at cap 4:

| Geometry | Reference ms | Symbolic ms | Reference peak MB | Symbolic peak MB | Sparse blocks before → after graph sharing/routing |
| --- | ---: | ---: | ---: | ---: | --- |
| Branch | 124.81 | 42.13 | 11.614 | 0.658 | 448 → 160 |
| Loop | 32.82 | 33.60 | 0.524 | 0.370 | 132 → 116 |
| Higher-body | 32.05 | 18.35 | 8.458 | 0.578 | 156 → 80 |
| Crossing | 14.10 | 11.75 | 0.534 | 0.560 | 60 → 60 |

The branch uses about 17.6x less measured preparation memory and 3.0x less
preparation time in this probe. Higher-body memory drops about 14.6x. Gains
are not universal: one-reference loop preparation grows 11.72→17.25 ms,
and crossing grows 4.60→8.52 ms. Full-channel loop preparation memory grows
0.189→0.210 MB because template overhead exceeds its small reduction.

Exact square channel dimensions, without a numerical cap:

| Geometry | Reference dimensions | Symbolic dimensions |
| --- | --- | --- |
| Branch | (289,17,1,17) | (25,5,1,5) |
| Loop | (13,13,13,13) | (9,9,9,9) |
| Higher-body | (81,1,1,45) | (25,1,1,5) |
| Crossing | (5,1,5,1,1,5,1,5) | unchanged |

An allocation-guard regression additionally verifies that the exact branch
plan prepares under a 1 MiB `memory_budget` where the reference plan raises.
This budget is an explicit work-array/estimate guard, not an RSS guarantee.

Torch CPU forward/backward includes local exponentials, assembly, PEPO
construction, dense contraction and a fixed weighted scalar objective.
Full channels (six-reference plans) take reference→symbolic:
branch 39.46→31.07 ms, loop 28.03→23.98 ms, higher-body 16.20→12.89 ms,
crossing 7.21→9.67 ms. At cap 4, branch is 29.00→30.23 ms and higher-body
13.69→15.17 ms: cheaper preparation does **not** imply faster replay.
These timings are uncompiled CPU probes. `aot_eager` compilation is tested
for correctness, not advertised as an optimized runtime speedup.

## Accuracy and cap dependence

Errors compare with the original uncompressed finite-cluster PEPO. Operator
errors are relative Frobenius norms. Gradient errors are relative Euclidean
norms for the objective `sum((Re(U)+.23 Im(U))*weight)`, where the fixed 16x16
weight is `arange(256)/256`. They are not global fidelity bounds.

Six-reference symbolic plans:

| Geometry | Cap | Operator error | Gradient error |
| --- | ---: | ---: | ---: |
| Branch | 1 | 0.05368 | 0.10284 |
| Branch | 4 | 0.02192 | 0.14062 |
| Loop | 1 | 0.06198 | 0.20647 |
| Loop | 4 | 0.04900 | 0.63063 |
| Higher-body | 1 | 0.03100 | 0.04858 |
| Higher-body | 4 | 0.02192 | 0.11794 |
| Crossing | 1 | 0.04383 | 0.06297 |
| Crossing | 4 | 0.03099 | 0.32606 |
| All four | full | ≤1.85e-16 | ≤4.31e-16 |

The cap-4 loop is less accurate than the old reference gauge (operator error
0.01810, gradient error 0.07737). Exact sharing changes no uncapped target,
but local QR selection is gauge dependent and not environment optimal.
Gradient error need not decrease monotonically with this local cap choice.
This is why the new preparation is explicit and why both operator and
objective gradients must be checked at held-out values. No error threshold
or user-requested accuracy was silently assumed.

## Validation and limits

The broader Pepsy selection passed 219 checks, including the first 21 new
PEPO tests. The final focused selection also covers the budget improvement
and isolated/single-basis sources; its final result is recorded in the
[handoff](../../../history/2026-09-29-symbolic-pepo-channels.md). Tests include
arbitrary independent residual values and adjoint derivatives on branches,
loops, higher-body and periodic crossing geometries; zero parameter
initialization; independent coefficient vectors; complex64; actual PEPO
traces; Torch CPU/CUDA gradients; JAX trace JIT; and full-graph Torch assembly
compilation. Gaugy's six-file downstream selection passed 163 tests.

This is conservative duplicate-slice elimination, not global symbolic
linear-dependency minimization. It still enumerates local fixed factors and
remaining routed blocks once. Dense local exponentials and graph routing
width can grow exponentially. It does not provide native/fermionic PEPOs,
parameter-dependent bases, or automatic choice of cap/reference set. MPO
frontier preparation is unchanged.

**Upstream decision:** adopt the existing fixed factors, public Autoray
conversion and Cotengra contraction tape; keep symbolic PEPO preparation
experimental and explicit. Reuse the same-task upstream audit with unchanged
dependencies: Quimb 1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305,
Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1, Symmray
0.4.1.dev7+g83fb22865, Torch 2.6.0+cu124, JAX 0.10.2. No installed libraries
were changed. Stronger parameter-family algebra and environment-aware
approximate basis selection remain deferred.

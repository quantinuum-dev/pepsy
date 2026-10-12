# 2026-09-29 — Fused compact cluster construction

This follows the [initial channel study](2026-09-29-compact-cluster-channels.md).
The user clarified that chi must control construction, including intermediate
MPO products and PEPO routing, rather than only the final operator's storage.
Scope remains Pauli/spin; implementation ownership remains entirely in Pepsy.

## Implemented

`ClusterChannelPlan.arrays`, `exp` and `trace_exp` now evaluate local residuals
and execute a static contraction schedule into compact site accumulators.
They do not call the original expanded MPO/PEPO assembler or reconstruct the
full sparse history table. The diagnostic `pack` / `bind_projector` path is
retained for independent projected-tensor and derivative comparisons.

MPO path contributions use the original fixed residual factors. Gap sites
are identity-wire indices rather than dense diagonal cores. A compatible
crossing collection contains at most one physical cluster factor per site;
all other spanning factors are wires. The two endpoint maps absorb these
wires without allocating their tensor-product MPO core. The direct source's
collection approximation policy remains part of its target.

Fixed recursive source builders prepare an equivalent exact direct topology
within their collection budget. This does not mutate their assembly policy,
use runtime SVD, or silently fall back to a one-cluster approximation. It is
not a compact recursive planner for graphs whose topology exceeds that budget.

Graph PEPO tree factors project directly into local tensors. Square channel
plans use existing graph residuals and exact routing; through-wire indices
are shared by the endpoint maps, without materializing routed product blocks.
Unprojected square targets are checked against the previous construction on
uniform chains, a square with loops, and short periodic routing. This changes
the reference virtual gauge for newly prepared square channel plans: capped
approximations need not match the initial sparse-block channel implementation.
Ordinary source `exp()` policies are unchanged.

`pack_residuals` / `bind_assembler` expose the complete residual-to-array
kernel. Static native-sector gathers, tree permutations, identity factors,
and contraction paths are bound outside compilation. Dense/native MPO,
graph and square kernels pass Torch `aot_eager`, `fullgraph=True`,
`dynamic=False` forward/backward checks. This does not compile the entire
Torch exponential/residual evaluator. JAX compact operator traces remain
JIT/gradient tested. All replay uses fixed shapes and no numerical SVD/QR.

Native replay validates conservation for each independent coefficient family,
including when a new coefficient-vector binding is supplied. Equal current
XX/YY values cannot justify an independent U1-conserving family. Bound
sector splitting matches the existing fixed factors and derivatives for U1,
Z2, U1U1 and Z2Z2, including repeated physical charges.

## Measurement

Single-thread CPU Torch complex128; reference `(a,b)=(.3,.2)`, held-out
`(.31,.19)`, step `-.1j`, two ordered factors `a sum ZZ` then `b sum X`,
channel cap six. Times include construction, actual operator trace and
backward for `real(trace)+.23 imag(trace)`; median of the last four of six
runs. Both paths use identical prepared bases and their values/gradients are
compared. Preparation and reference QR are outside these timings.

| Geometry / cutoff | packed replay ms | fused replay ms | largest forward tensor entries, packed → fused | saved entries, packed → fused |
| --- | ---: | ---: | --- | --- |
| Eight-site MPO / 4 | 72.8 | 37.4 | 4,056 → 256 | 15,324 → 6,172 |
| Eight-site MPO / 5 | 181.1 | 42.8 | 13,368 → 1,024 | 49,988 → 22,972 |
| 2×3 graph PEPO / 3 | 58.3 | 48.9 | 2,772 → 864 | 17,752 → 10,456 |
| 2×3 square PEPO / 3 | 58.0 | 47.9 | 2,772 → 864 | 17,752 → 10,456 |

Forward sizes were observed with Torch dispatch; saved entries count tensor
save events, including repeats, rather than unique allocated bytes. They are
not peak RSS or total GPU memory. Development timing varies with concurrent
work; another run gave 140.3/31.9 ms for the order-five MPO. These probes
support an assembly improvement, not a universal throughput guarantee or
an optimized compiler benchmark.

The cap is an approximation. Relative dense operator errors against the
uncompressed expansion were 3.07e-3 and 2.91e-3 for the two MPO cases, and
1.69e-3 for both PEPO cases. Absolute errors in the two-parameter gradient
of the **unnormalized** trace objective were 4.74, 3.95, .395 and .395.
Correct derivatives of the projected operator are not automatically accurate
derivatives of the uncompressed target. Increasing chi and/or enriching the
reference snapshots remains an accuracy choice separate from this replay fix.

## Limits and upstream decision

Reference preparation still visits expanded topology and host QR snapshots.
Full symbolic minimization, environment-optimal compression and unrestricted
compact recursive planning remain unimplemented. Local dense residuals and
local fixed factors grow with cluster size, and a dense square site tensor
still scales as d² times the product of its four retained bond dimensions.
`memory_budget` guards explicit projection operands/results and preparation
buffers; it is not a bound on total contraction or autodiff memory.

The same-task upstream audit and installed versions from the initial study
were reused. **Adopt:** public Cotengra `array_contract_tree` / `get_path`
with greedy planning and a static backend einsum tape. Cotengra's Python
fallback remains available. **Narrow compatibility choice:** bind Torch's
native `cat` through Autoray before capture, avoiding its argument-translator
wrapper during compiled native replay. No dependencies or installed libraries
were changed. The APIs and scoped validation are linked in the
[handoff](../../../history/2026-09-29-fused-cluster-channels.md).

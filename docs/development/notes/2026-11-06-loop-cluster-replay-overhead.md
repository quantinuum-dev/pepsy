# Loop-cluster snapshot and norm reuse

The Eli replay avoids two measured redundant operations: constructing an
unused physical network after every pair, and contracting the target norm
again for scalar logging. The [audit](../../../history/2026-11-06-loop-cluster-overhead-audit.md)
retains the original findings and measurements.

## Implementation and compatibility

`materialize_physical=False` reaches the native reduced-gate finalizer from
`apply_reduced_loop_cluster_gate`, `gate_loop_cluster`, and
`PreparedLoopClusterUpdate.apply`. It omits only the separate physical
snapshot: `result.physical_tn=None`. Public defaults remain `True`, preserving
existing result types and materialization. Eli explicitly selects the omitted
snapshot for both prepared and reference replay. Preparation of the numerical
environment still uses the existing full-network gauge-inserted copy.

`ReducedALSSolution.target_norm` stores the Python scalar computed during
Torch, batched Torch, Quimb, and monitored QR/normal solving. Unsupported or
unrecorded cases retain `None`. Eli reuses that scalar, falling back to the
problem diagnostic if absent. The public `problem.target_norm` remains a fresh
contraction, so mutating target/environment arrays cannot expose a stale
property cache. No device tensor or autograd graph is retained for this scalar.

ALS initialization, equations, conditioning guards, stopping, per-pair gauge
extraction, exponents, gate order, source barriers and truncations are unchanged.
Automatic batching and changes to scalar synchronization remain separate work.

## Upstream inspection

Classification: **adopt** the existing public Quimb copy/gauge-insertion
operations only when their physical snapshot is requested. No compatibility
shim, dependency upgrade, installed-source edit or solver-policy change.

Python 3.12, Quimb 1.15.1.dev101+g1b64353d9, Autoray
0.11.1.dev14+g014a3f69a, Cotengra 0.8.3, Cotengrust 0.2.1,
Symmray 0.4.1.dev25+gdc377f033, Torch 2.14.1+cu126.
Installed signatures were inspected: `TensorNetwork.copy(virtual=False,
deep=False)` and `gauge_simple_insert(gauges, remove=False, smudge=0.0,
power=1.0, return_gauges='raw')`. The numerical backend dispatch is unchanged.

Checked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The [Autoray repository](https://github.com/jcmgray/autoray) and
[Symmray array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
were unavailable through the web tool; installed code/signatures and the
official Symmray repository supplied the relevant fallback inspection.

## Validation

CPU/CUDA multi-gate regressions compare physical operators, gradients, gauges,
exponents, costs, iteration counts and fidelity tracking with/without physical
snapshots, across source boundaries. Counted insertions prove one full-network
insertion per pair is removed. A separate mutation regression proves solver
norms are scalar snapshots while problem diagnostics remain current.
The batch suite also checks reported norms against current problems.

The affected package/API selection passed **313 tests**; native Symmray local
gauge/reduced-compression checks passed **3 tests**. Downstream complete-operator,
prefix, backend, fidelity, CLI and disk-cache tests passed **98 tests**.
These are scoped checks, separate from the full-suite result recorded in the
[handoff](../../../history/2026-11-06-loop-cluster-replay-overhead.md).

## Matched D6 CUDA diagnostic

Reuse the audit's saved 5x5 homogeneous depth20 residual, pair
`(2,2)—(3,2)`, diagonal lower-leg gate, complex128, native `gesvd`, one
numerical thread, cap10/tol1e-8 and local fidelity. Each mode starts from
independent copies of that same state. Plan setup and input copies are
outside the timer; snapshot construction, the logging callback and CUDA
synchronization are included. Four ALS sweeps in every sample.
The baseline requests physical snapshots and reads `problem.target_norm`;
the optimized mode omits snapshots and reads `solution.target_norm`.

Alternate baseline/optimized order across fifteen warm pairs per run.
The initial unpinned run had medians **76.65/81.34 ms**. This is retained:
individual samples varied about 49–94 ms, so separate medians are sensitive
to scheduling drift. Two later runs pinned the host to CPU4:

| Repeat | Baseline median, ms | Optimized median, ms | Median paired baseline/optimized |
| --- | ---: | ---: | ---: |
| 2 | 84.26 | 81.97 | 1.04624 |
| 3 | 77.88 | 58.70 | 1.04702 |

These paired results indicate a modest local improvement, with substantial
timing variability. They establish neither stable full-trajectory throughput
nor a 1.33x speedup suggested by comparing repeat 3's separate medians.
Outputs match exactly for core arrays, gauges, exponent, costs, iterations,
solver counts and local fidelity; target norm differences are at most
1.67e-16 relative. No global trace was evaluated.

Reproduction: `/tmp/loop_cluster_overhead_fix_benchmark_20261106.py`.
Raw samples and comparisons are retained in `/tmp/loop-cluster-overhead-fix-20261106/`
and its `-repeat2` / `-repeat3` sibling directories. The two repeats use
`taskset -c 4` and `LOOP_OVERHEAD_OUTPUT` to preserve each result set.

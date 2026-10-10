# FIT sum caching measurements, 2026-10-09

Follow-up to the [sum-target implementation](2026-10-09-fit-sum-targets.md).
User requested attention to caching and tensor construction. Implementation
and measurements are local, uncommitted, on `develop` at baseline `21f883b`.

## Findings and changes

The original sum implementation already reused fixed/moving environments
across sweep reversals and block-size transitions. The review removed three
additional sources of repeated work:

1. Construct the fitted-site bra once per environment update and share it
   across targets. Its numerical data is never cached across state updates.
2. Cache target component selection and original tensor ordering per visited
   block. Retain only references to existing factors, without contracting
   target blocks into larger intermediate tensors. These immutable routing
   caches can survive another run on the same target structure.
3. For optional verbose fidelity, compute the target norm once per run using
   the Hermitian triangle, including all cross terms. Reset it on every run
   to avoid stale numerical values or autodiff graphs.

Regression counters verify one fixed-side build for six alternating 3→2→1
sweeps; five direction reversals reuse environments. Each visited target
block is selected/ordered once. A new run rebuilds fitted-state environments
while retaining target routing. Three terms share one bra conjugation per
site extension. Three verbose sweeps use 18 closed contractions instead of
39: 6 target pairs once and 4 fitted overlaps/norms per sweep.

## CPU measurements

NumPy complex128, one BLAS thread (`threadpoolctl.threadpool_limits(1)`),
six RL sweeps, initial block size 3, one two-site transition sweep,
`max_bond=4`, `cutoff=1e-12`, `rtol=None`. Timed only `run_gate`, excluding
construction. Two warm-up repetitions preceded seven measured repetitions;
mode order alternated and the table reports medians. Final measurements were
run after the regression-test process completed.

| Three-term case | Before review | Updated | Updated, rebuild every sweep |
| --- | ---: | ---: | ---: |
| Ordinary MPO, L=12, target/guess bond 4 | 49.219 ms | 45.502 ms | 53.170 ms |
| Layered MPO products, L=10, factor bond 2, guess bond 4 | 67.704 ms | 62.940 ms | 76.545 ms |
| Ordinary MPO with verbose fidelity | 111.258 ms | 70.402 ms | 78.095 ms |

The updated ordinary-MPO generic contraction route took 60.636 ms, versus
45.502 ms for automatic dense direct contractions. All comparisons checked
normalized fidelity and absolute norm agreement within `1e-10` before
accepting their timings. Output cap and sweep controls were identical.

Ordinary inputs were `qtn.MPO_rand(12, 4, dtype='complex128', seed=s)` for
target seeds 1, 2, 3 and guess seed 70. Layered inputs were six
`qtn.MPO_rand(10, 2, dtype='complex128', seed=s)` operators with seeds 0…5,
paired using `.apply(other, contract=False)`, with a bond-4 guess at seed 70.
The rebuild control set `_allow_sweep_environment_reuse=False`. The before
control loaded the preceding sum implementation saved at the start of this
review. Temporary driver: `/tmp/pepsy_sum_cache_bench.py`; final raw results:
`/tmp/pepsy-sum-cache-bench-final.log` (temporary files are not durable).

These small CPU cases show approximately 7–8% lower normal sweep time and
37% lower verbose time than before this review. They are not a general
optimality claim or a CUDA/compiled benchmark. Large-rank contraction and SVD
costs can dominate these Python/construction savings.

## Compatibility and validation

Rechecked unchanged installed versions: Quimb 1.15.1.dev90+g6a3906cbe,
Autoray 0.11.1.dev14+g014a3f69a, Cotengra 0.8.3.dev8+g8954240f2,
Symmray 0.4.1.dev15+g0374aaa3c. Reused the official-source/signature audit
recorded for this same ongoing FIT task. Classification remains **adopt**:
public Quimb contractions and native additions/splits, no upstream shim,
dependency update, custom decomposition, or dense conversion.

**373 focused tests passed**, including 55 sum tests and native Z2/U1U1
MPS/MPO, Torch gradients, cancellation, window ownership, schedules, and
public API. Ruff (`src tests`) and `git diff --check` passed. No full-suite
run. See the [handoff](../../../history/2026-10-09-fit-sum-cache-review.md).

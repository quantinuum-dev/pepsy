# 2026-11-05 — Dense cluster ALS correctness and speed review

User requested a check of ALS/QR/LQ/private stages and further speed options.
Package baseline: `develop`, `4d50e53`. This is a bounded profiling study;
no package implementation, defaults or running campaign changed. The previous
cluster audit report remains separate and uncommitted.

## Correctness scope

The current dense path inserts SU weights once, QR/LQ-reduces the active
sites, constructs the exact selected-patch environment with SU exterior
closures, initializes the reduced target, and alternates native local solves.
Complete-sweep QR/LQ preserves the joint reduced tensor. Final gauge extraction
uses full exterior SU weights and a bond-sized SVD; per-pair RMS extraction
preserves the operator amplitude in the exponent. These are consistent with
the independent reconstruction/weighted-SVD checks in the preceding audit.
Native U1 and optional-mode defects in that report remain unresolved.

Quimb's public ALS accepts supplied live `tnAA`/`tnAB` networks, so Pepsy's
one-sweep calls retain the solved tensors rather than resetting their values.
The physical indices are batched right-hand sides in Quimb's local solve;
it already avoids explicitly expanding their identity into the norm matrix.
Hermitian-only patch construction skips an environment eigensolve, but
current ALS still uses eigendecomposition for its local positive solves.

## Measured workload

Loaded the corrected 23-spin homogeneous D20 χ6 final residual and gauges
read-only. Replayed one fixed extra lower-leg RZZ gate on interior bonds,
using complex128, one host thread, exact `gesvd`, Hermitian-only six-site
environments, cached contraction paths, ALS cap10/tol1e-8. This is not a
new T=6 run or a full-trajectory speed/convergence claim.

One warmed CUDA pair update took approximately **97 ms**. Separately
synchronized phase instrumentation attributed about 58.5 ms to four native
ALS sweeps, 9.4 ms to initial SVD, 13.0 ms to environment construction and
contraction, 3.1 ms to inter-sweep QR/LQ, and 5.8 ms to final gauge/output
construction. Preparation took 4.4 ms. Instrumented timings include their
own synchronization overhead. The reduced tensor has shape (24,4,4,24),
initial SVD is 96×96, and local ALS matrices are 144×144.

A disposable prototype that omitted only materialization of the unused
fully-gauged output network left core arrays and gauges identical. Its median
CUDA time changed from 97.0 to 95.8 ms, a small difference. Preserving the
public result would require lazy snapshot semantics rather than dropping its
`physical_tn` field; this prototype is not a public API implementation.

## Local-solver opportunity

Using the existing explicit `quimb_opts` to select direct solves instead of
positive eigensolves gave the following warmed median GPU times:

| Interior bond | Current update | Direct-solve prototype | Speed ratio |
| --- | ---: | ---: | ---: |
| (3,1)–(3,2) | 104.3 ms | 58.2 ms | 1.79× |
| (2,1)–(2,2) | 121.2 ms | 66.2 ms | 1.83× |
| (2,2)–(2,3) | 122.8 ms | 68.3 ms | 1.80× |

Both routes took the same 4/5/5 sweeps. Final normalized local squared errors
agreed to approximately 1e-16 absolute; relative reduced-tensor differences
were 2.7e-9 to 1.7e-8. Sampled local eigenvalue ratios were 4.1e-10 to 3.1e-8,
above the current 1e-12 clipping threshold.

The prototype explicitly disables positivity clipping. It is **not ready as
an unconditional default**: near-singular/indefinite cases need a guarded
solve and the existing eigensolver fallback, with matched trajectory and
backend checks. Keep ALS tolerances and iteration cap unchanged during that
comparison. A smaller SVD initialization using exact gate-factor structure
is another candidate; no such initialization speedup was measured here.
Cache topology/layout work, not stale numerical environments after gates.

Temporary scripts and machine-readable measurements are under
`/tmp/pepsy-cluster-speed-20261105/`. The earlier 137-test audit included CPU
and CUDA; this follow-up adds the bounded profiles and solver comparisons,
not a new full-suite result. No commit or push occurred.

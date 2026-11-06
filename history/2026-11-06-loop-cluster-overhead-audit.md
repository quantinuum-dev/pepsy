# 2026-11-06 — Current loop-cluster overhead audit

- Scope: user requested avoidable slowdowns in Eli's loop-cluster implementation.
- Baselines: Pepsy `develop` at `bb5ad4d`; examples `main` at `fa2c63c`.
- Change: this audit record only; numerical code and production outputs unchanged.
- Commit status: included in the documentation commit carrying this record;
  nothing pushed.

## Bounded measurement

Use the saved corrected homogeneous depth20, 23-active-spin, D6 residual,
embedded in a 5x5 PEPO. Apply one lower-leg diagonal gate at
`(2,2)—(3,2)` on independent copies of the same final saved state. This is a
diagnostic, not a continuation or global trace result. Latest prepared replay,
CUDA complex128, native `gesvd`, one numerical thread, cap10/tol1e-8,
Hermitian-only environment and local fidelity tracking. Four ALS sweeps.

Discard one cold call; alternate seven repetitions with/without a callback
that reads `problem.target_norm`. Median wall times are **90.95/88.03 ms**.
Separate CUDA-synchronized, inclusive phase medians over four repetitions:
preparation **7.34 ms**, environment **14.13 ms**, ALS **59.62 ms**,
reconstruction **7.53 ms**, diagnostic norm **2.00 ms**. Included in
reconstruction is **1.92 ms** for the second full-network gauge insertion.
Nested timers perturb execution and independent medians need not add.
One Torch CPU-side profile counts **27 scalar extractions** and **283 clones**;
their recorded wait times include GPU computation and are not removable
overhead estimates. One warm program miss and twenty hits; no evictions.

## Findings

1. `_local_svd_reduced_state` always copies/gauge-inserts `physical_tn`, even
   when prepared replay never consumes that result field. Optional/lazy
   materialization could remove this work while preserving the public result
   contract. Preparation also gauge-inserts a whole-network copy for each
   pair, although its environment uses a local patch.
2. The example callback recomputes the weighted target norm already evaluated
   by Torch ALS. Passing the solver's scalar norm into result diagnostics
   could avoid an extra contraction and synchronization. Do not permanently
   cache the public property across mutable problem tensors/environments.
3. ALS is the largest stage in this D6 case. Its SVD initializer, two guarded
   solves per sweep, full triangular inverse conditioning certificate and
   host scalar decisions are further profiling targets. These guards enforce
   numerical behavior; eliminating checks or changing initialization needs
   independent reconstruction/gradient/convergence validation.
4. Eli's runner still uses `batch_size=1`; experimental auto batching is
   available in Pepsy. Earlier real-stream geometry scans and synthetic gains
   do not establish an Eli full-trajectory speedup. A matched comparison is
   needed before choosing it as the benchmark default.

The earlier [stage profile](2026-11-05-cluster-stage-profile.md) separately
shows larger-D environment contraction dominance and CPU16 small-D ALS
slowdowns. Retain that hardware/workload scope.

Reproduction: `/tmp/loop_cluster_overhead_audit_20261106.py` in the shared
Python 3.12 environment with local Pepsy/Gaugy sources. Evidence:
`/tmp/loop-cluster-overhead-audit-20261106/{summary.json,calls.txt,calls.pstats,run.log}`.
No numerical tests rerun for this read-only profile; documentation whitespace
checks passed. No implementation speedup is claimed.

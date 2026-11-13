# 2026-11-11: full-update CUDA synchronization and gate scheduling

## Scope and numerical contract

Dense Torch CUDA two-site full update batches small SVD-rank and positive
environment diagnostics into host reads. It avoids repeated evaluation of
the same initial local loss and repeated extraction of target/final fidelity
scalars. The zero-floor PSD projection retains the same eigensystem and
clipping, returning device diagnostics for the FU caller to collect. Rank,
PSD checks, adaptive ALS stopping and candidate acceptance remain in place.

Fixed CUDA gates are classified in bounded groups of 128 for `smart`, `row`,
and `column` ordering. The exact Pauli/diagonal tests and dependency graph are
unchanged; only classification transfers are batched. Unsupported/trainable
gates remain barriers. The Eli runner continues to schedule inside source
slices, carrying the PEPO and dependency-checked boundary caches between slices.
Changing gate order, or combining opposite PEPO legs, can change the finite-D
approximation. The later user-requested default is smart for full update;
separate generic PepsOptimizer runs are required for source/depth barriers.

The optional SU warm start uses Quimb's public single-index diagonal
multiplication. This restores execution on installed Quimb 1.15.0, which
does not have the previously called plural-index helper. The separate
standalone Gaugy SU `transpose` incompatibility is not addressed here.

This does not remove all CPU synchronization: rank decisions, convergence
and scalar reports still use the host, and Torch CUDA `eigh` synchronizes.
The explicit L-BFGS alternative still exchanges vectors with SciPy.

## Measured GPU checks

RTX A5000, Torch 2.14.1+cu126, Quimb 1.15.0, Autoray
0.11.1.dev16+g14513fea6, Cotengra 0.8.4.dev7+g5d3558c4f,
Cotengrust 0.2.1, Symmray 0.5.0; one CPU thread, complex128.
No dependency or installed source was modified. The Quimb changelog,
Autoray repository, Cotengra changelog and Symmray repository were checked;
the Symmray array-doc page was unavailable. This adopts existing public APIs;
the single-index warm-start change is a narrow compatibility fix.

In one instrumented 3×2 PEPO replay with two truncated fits, D=2, boundary
cap 8, direct boundaries and four ALS sweeps per fit, a frozen pre-change
source copy and the current source gave:

| Profiler event | Before | After |
| --- | ---: | ---: |
| `aten::item` | 68 | 38 |
| CUDA stream synchronizations | 104 | 66 |
| Device-to-host copies | 102 | 66 |
| Device-to-host bytes | 562 | 398 |
| `linalg_eigh` calls | 18 | 18 |

Both outputs remained on CUDA, used Quimb ALS for both pairs, and differed
from the same CPU result by relative norm 3.35e-15. The trace includes
profiler/runtime activity and explicit profiler synchronization; event counts
are not an end-to-end speed measurement. Largest post-change device-to-host
copy was 40 bytes. The CUDA `eigh` calls and convergence decisions remain.

On the actual 10×10 inhomogeneous D20 schedule, dt=.025, the first source
slice contains 5,130 original actions. Original scalar gate classification
and the current batched version returned **identical original action indices**
for both smart and column ordering. Single-trial gate-order preparation times
on the same GPU were smart 2.04 → 1.26 s and column 0.241 → 0.094 s. This
measures ordering only and does not establish whole-evolution speedup.

## Middle-out and boundary-cache comparison

The existing Eli `time` scheduler balances 240 target steps among 20 source
slices. Each slice is a hard barrier for FU ordering. An older whole-stream
reordering lost the useful finite-bond middle-out path; see the sibling
`pepsy_examples` FU source-ordering record. The current source-bounded path
is also covered by independent small complete-operator tests.

Single-trial instrumented 10×10 prefixes used the same first source slice,
T=6, dt=.025, D=4, boundary cap 12, direct boundaries, 20 ALS iterations,
`gesvd`, Torch CUDA complex128, source barriers and optional opposite-leg
fusion. No final T=6 cost was measured. These runs used the development
version before the final minor classification/fidelity cleanup; CPU tests
overlapped briefly and the host was not isolated.

| Order | Time | Boundary builds | Boundary reuses | Pair updates | Joint updates | Local fidelity product |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smart | 121.78 s | 599 | 16,681 | 2,128 | 160 | 0.9999896414 |
| column | 282.10 s | 3,413 | 13,597 | 2,242 | 46 | 0.9999906461 |

Reported environment phase times were 71.93 s and 233.46 s respectively.
The prefix profiler synchronizes phases, affecting runtime. These single
measurements support a cache-reuse advantage for smart scheduling in this
prefix; they do not establish a portable speedup or final global accuracy.
At measurement time the Eli default was `column`; the subsequent user-requested
change makes `smart` the default while retaining source-slice barriers. Pass
`--full-update-gate-order column` to reproduce the former default.

The sibling benchmark's `eli_global_cost/pepo_profile.py` reproduces the
prefix with `--source-slices 1 --target-dt .025 --bond 4 --boundary-chi 12
--boundary-mode direct --gate-order smart --solver auto --svd-driver gesvd
--iterations 20 --device cuda --threads 1`, the 10×10 D20 source and a fresh
`--out`. Replace `smart` by `column` for the comparison. The profile emits no
full-time cost. Raw profiler and prefix outputs for this machine are under
`/tmp/pepsy-fu-*` and are not Git artifacts.

## Validation

209 CPU full-update/gate-order tests passed, including cached versus uncached
PEPS/PEPO, SU warm starts and dense targets. Separately, 39 accelerator-focused
tests passed (including one CPU reference), covering complex64/complex128, upper/lower/joint legs,
direct/DMRG boundaries, exact-rank thresholds, PSD diagnostics, the two
batched classifiers, smart-order equivalence and both warm starts. Four
CuPy complex128 SU-warm-start tests passed. The sibling Eli source-order
and preflight selection passed 18 tests. Ruff and diff checks passed.
The full suite stopped at **354 passed, 2 skipped, 1 failed** in the previously
recorded `test_fermionic_long_range_hopping_sign_survives_su_and_bp_gauges`:
the exact fermionic readout was zero rather than the expected −0.24383, before
the FU path is entered. Log: `/tmp/pepsy-fu-cuda-full-suite-20261111.log`.
A complete T=6 GPU comparison was not run.

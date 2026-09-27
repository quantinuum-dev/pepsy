# 2026-09-26 — PEPS sampler performance assessment

Status: measured assessment and temporary prototypes. No sampler implementation
was changed during this performance task. The preceding
[backend fix](peps_sampler_backend_fix.md) remains in the working tree.

## Summary

The current sampler is a useful, backend-preserving reference implementation
with prefix sharing. It is not yet an efficient GPU batch sampler. The largest
immediate issue is the row-transfer cache: routing ignores tensor dimensions,
so an optimization can materialize enormous dense transfer tensors.

Priority order:

1. Replace the row-cache heuristic with estimated memory/work checks.
2. Batch probability validation and random draws across active prefix groups.
3. Reuse contraction expressions and topology metadata, reducing Quimb object
   creation and eventually implementing factored row environments.
4. Add effective-sample/weight diagnostics and stable scaled rho norms.
5. Develop actual batched boundary contractions and SVD/QR kernels after the
   previous steps, preserving the existing reference implementation.

## Method and scope

- develop / 80f451a, including the existing uncommitted backend fix.
- NumPy 2.5.2, Torch 2.6.0+cu124, JAX 0.10.2,
  Quimb 1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305,
  Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1,
  Symmray 0.4.1.dev7+g83fb22865.
- Activated the existing py312 environment; checkout src on PYTHONPATH.
  BLAS/OpenMP/Torch threads were one. Torch timings used inference_mode.
- Baseline: finite random 4x4 PEPS, D=4, physical dimension 2, seed 101,
  complex128, sample_chi=16, marginal_chi=32, Quimb future and ket compression,
  greedy contraction planning. DMRG/FIT comparisons use one FIT iteration.
- One warm-up and two measured calls with seeds 800/801. Tables show the
  median of those calls; the initial pilot has one measured call.
- Stage wrappers record exclusive host wall time. CUDA totals synchronize
  before and after sampling; CUDA stage numbers are not kernel-time attribution.
  cProfile observations come from separate calls, not the throughput timings.
- The GPU was already at 100% utilization from the user's production jobs;
  CPU exact evolution was also running. Jobs were not stopped or altered.
  CUDA results describe this shared workstation, not an idle-device comparison.
  Some independent probes overlapped; these are bounded engineering
  measurements, not controlled microbenchmarks or statistically precise rankings.
- 23 timing cases, plus direct route-equivalence, allocation-size, and
  synchronization probes. JAX performance and production 5x6/high-D
  throughput were not measured.

## 1. Dense row-cache routing is the first improvement

Current source:
[sampling route](../../../src/pepsy/sampling/samplers.py),
_sample_batch_boundary and _boundary_sample_or_probability.

Batch routing chooses the transfer cache for up to 32 shots on at most nine
sites, or up to four shots on larger lattices. Serial positive-marginal
sampling chooses its full-center reference only above nine sites. Neither
criterion considers D, the actual conditioned/future bonds, dtype, or
simultaneously live prefix groups.

A temporary subclass selected the existing reference-prefix implementation
for a 3x3 D=4 PEPS, sample_chi=16, marginal_chi=32:

| Eight-shot call | Time |
| --- | ---: |
| Current dense transfer cache | 15.934 s |
| Existing reference-prefix route | 0.07663 s |
| Ratio for this case | 207.9x |

The direct comparison used the same state and seed 812. Configurations matched
exactly; proposal probabilities and amplitudes matched at rtol=1e-11,
atol=1e-14. This is a validated routing prototype, not an installed speedup.

The largest allocated local transfer was **1 GiB** for this 3x3 case.
An allocation estimate from the 4x4 D=4 network metadata found individual
local transfers of **16 GiB** at the third row, before workspace and other
prefix groups. Those large tensors were not allocated in the probe.

The transfer output size contains both left and right products of the
conditioned ket/bra, current PEPS ket/bra, and future-boundary bonds. Small
site count does not imply a small transfer.

**Proposed implementation:** estimate output elements and workspace before
materializing a column; account for dtype and live groups; route to the
existing reference above a conservative memory/work budget. Keep a tested
override for benchmarking. Longer term, keep row environments factored
instead of forming a large dense transfer matrix.

**Current workaround:** use sample_batch with at least 64 shots on these
small D=4 examples. That selects the reference-prefix branch under the current
heuristic. This is a workaround for the measured regime, not a universal
optimal batch size.

## 2. Prefix batching helps, but distinct prefixes are still serial

Random 4x4 D=4, complex128, 32 shots:

| Backend / method | Warm call | Shots/s |
| --- | ---: | ---: |
| NumPy serial | 1.006 s | 31.8 |
| NumPy prefix batch | 0.877 s | 36.5 |
| Torch CPU prefix batch | 1.112 s | 28.8 |
| Torch CUDA serial, shared GPU | 9.105 s | 3.51 |
| Torch CUDA prefix batch, shared GPU | 7.040 s | 4.55 |

For NumPy, batching reduced local-rho evaluations from 512 to approximately
351 per call and improved throughput by about 15%. Final configurations were
almost all distinct. Sharing is strongest near the beginning; it does not
produce one tensor batch calculation for all later shots.

NumPy batch size 8 gave 34.4 shots/s; batch size 64 gave 40.5 shots/s.
Tune batch size using throughput and memory on the actual state. Large batches
can retain many separate conditioned networks, so unbounded batches are not
a general remedy.

CUDA complex64 measured 8.830 s serial / 6.981 s batched, close to complex128.
This does not establish intrinsic GPU performance or a precision speedup.
The current workload consists of many small operations and host decisions,
and the GPU was busy.

## 3. Where time and synchronization go

Exclusive host stage shares for the NumPy 32-shot baseline:

| Stage | Share |
| --- | ---: |
| Local-rho contractions and their temporary networks | 48.5% |
| Projected-row application / conditioned-boundary update and compression | 26.1% |
| Center-network construction | 7.3% |
| Probability validation and diagnostics | 4.0% |
| Exact projected amplitudes | 2.9% |
| Random choices | 2.1% |
| Remaining grouping/control/output overhead | 9.1% |

A separate cProfile call observed 1,584 tensor-network copies for a 32-shot
batch; TensorNetwork.copy accounted for about 20% of the profiled cumulative
wall time. These cumulative entries overlap contraction stages and should
not be added to the table.

A CUDA batch with seed 800 made, within sampler code:

- 349 scalar reads for conditional validation;
- 349 integer-choice array copies to NumPy;
- 32 scalar reads for output proposal mantissas/exponents;
- 32 scalar reads for output amplitudes.

Upstream compression may add scalar reads. No full rho/probability-vector
copies were observed. The dispatch observer reported 45,789 Torch operator
calls, including 8,190 tensordot calls; these are dispatch calls, not a count
of launched CUDA kernels. In inference_mode the observer did not expose
scalar/copy primitives, so those readout counts were taken from direct
sampler/Autoray call instrumentation.

**Proposed near-term GPU improvement:** collect the small local rhos for
all active groups at one site, stack them, validate and draw together, and
copy selected indices once for the site. Keep the sampled-prefix dependency.
This can reduce readback frequency before rewriting boundary contractions.
It has not been implemented or benchmarked end to end.

**Proposed larger improvement:** bucket compatible boundary shapes and use
batched array contractions, QR/SVD, and gathers. Share fixed PEPS tensors.
A repeated ordinary Quimb index is not a valid independent shot axis.
Preserve truncation policy, source tensors, seeds within the new method, and
exact amplitude/proposal separation in regression tests.

## 4. Optimize effective samples per second

Importance weights are proportional to abs(Psi(S))**2 / q(S). The measured
diagnostic is ESS = (sum weights)**2 / sum(weights**2). Use log weights to
avoid overflow. ESS measures weight concentration; a small pilot does not
certify convergence, absence of rare heavy weights, or observable error bars.

The [direct-sampling paper](https://arxiv.org/abs/2109.07356) motivates auxiliary
proposals with importance correction and studies how a future marginal
environment improves sampling efficiency. Its good marginal cutoffs are
state-dependent, not a universal prescription for time-evolved PEPS.

A second benchmark used the user's simple-update workflow: 4x4, D=4,
dt=0.4, five steps (t=2), diagonal initial wall, initial gauge equilibration
only. Sampling used sample_chi=16, complex128, 128 shots, Quimb compression.

| marginal_chi | Warm time | Mean ESS / 128 | Effective samples/s |
| --- | ---: | ---: | ---: |
| 0 | 2.103 s | 30.66% | 18.7 |
| 8 | 1.473 s | 99.91% | 86.8 |
| 16 | 1.441 s | 100.00% rounded | 88.8 |

Two independent measured batches contributed to each ESS average.
The near-equal times for 8 and 16 should not be treated as a precise ordering.
For this snapshot, a modest future environment gave approximately 4.7x the
effective-sample throughput of no future environment. Proposal quality also
changed prefix reuse, so smaller marginal_chi was not automatically faster.

An exact dense contraction of this small SU snapshot gave norm squared
0.9158841738432193 and checkerboard staggered mean Z
-0.0034798221853733803. The two weighted sampled means for marginal_chi=8 were
-0.01926 and 0.00639; for 16 they were -0.02241 and 0.01087. These are finite
sample estimates, not deterministic observable errors. The snapshot's
subunit norm was retained.

On the random state, reducing sample_chi from 16 to 8 lowered pilot ESS from
about 99.9% to 94.1% without a visible throughput gain. Choose the two bond
caps separately, monitor ESS and observable stability, and include setup
cost when only one small batch is drawn per changed state.

## 5. DMRG/FIT and contraction planners

Random 4x4 D=4, 32-shot NumPy calls:

| Future / ket compressor | Warm time | Pilot ESS fraction |
| --- | ---: | ---: |
| Quimb / Quimb | 0.877 s | 99.9% |
| DMRG / Quimb, one FIT iteration | 0.899 s | 99.8% |
| DMRG / FIT, one FIT iteration | 1.064 s | 99.8% |

These results support Quimb ket compression as the initial performance choice
for this regime. DMRG/FIT is an accuracy option to compare when stronger
truncation needs it; no general quality ranking is established by this pilot.

Planner comparison on the same baseline:

| contraction_opt | First sampling call | Warm call |
| --- | ---: | ---: |
| greedy | 0.868 s | 0.877 s |
| auto-hq | 1.486 s | 0.837 s |
| ReusableHyperOptimizer, eight greedy trials | 2.126 s | 0.986 s |

The warm difference between greedy and auto-hq is small relative to shared
machine variability. A new reusable optimizer did not improve this case.
Existing Quimb/Cotengra caches already serve repeated contractions; cProfile
did not identify path search as the dominant warm cost.

[ReusableHyperOptimizer](https://cotengra.readthedocs.io/en/latest/basics.html#reusablehyperoptimizer)
caches paths for matching ordered contraction inputs. Reuse a planner and
stable topology where possible; do not assume adding another planner cache
alone solves tensor-network reconstruction and dispatch costs.

## 6. Numerical diagnostic found during profiling

On the unnormalized random complex64 CUDA PEPS, the first rho had maximum
absolute entry 7.408e23. Entries and trace were finite, and sampling completed,
but the unscaled Frobenius norm overflowed to infinity. The Hermiticity
diagnostic consequently became NaN.

Dividing rho by its maximum absolute entry before evaluating the relative
norm gave a finite defect of 0.00498636 in a temporary calculation.
This is a diagnostic overflow finding, not evidence that the sampled
probabilities themselves overflowed.

**Proposed fix:** compute the diagnostic with safe scaling (including zero
and tiny-norm cases), preserving the current denominator convention. Add
float32 rescaling-invariance and non-finite regressions. Do not change PEPS
norms or gauges to hide the issue. No production fix was made in this task.

## Best practice now

- Explicitly choose boundary mode for practical PEPS sampling; the default
  exact mode is a small-system reference.
- Start from prefix batches of 64–128 and measure the actual state. For the
  tested D=4 snapshots, sample_chi=16 and marginal_chi=8–16 are useful pilot
  settings, not validated settings for all 5x6/t=10 snapshots.
- Start with Quimb ket compression. Compare DMRG/FIT using ESS/time and
  observable convergence if truncation becomes significant.
- Reuse the sampler for an unchanged snapshot; refresh after evolution or
  tensor changes. Reuse shape-compatible contraction plans, but rebuild
  state-dependent future environments when the state changes.
- Apply importance weights to Z, staggered Z, and other observables whenever
  the proposal is approximate; report ESS and Monte Carlo uncertainty.
- Keep complex128 as the accuracy reference. Benchmark complex64 only after
  checking finite diagnostics and observable/proposal accuracy.
- Torch no_grad/inference_mode is appropriate for sampling-only workloads
  when gradients are not needed. Avoid timing import/JIT/path warm-up as
  steady-state throughput.
- Measure CUDA with synchronization or CUDA events, and benchmark it idle
  before selecting it on speed grounds.
  [PyTorch CUDA timing guidance](https://docs.pytorch.org/docs/2.6/notes/cuda.html#asynchronous-execution).
- Preserve exact projected amplitudes for the current importance estimator.
  Replacing them with a compressed boundary amplitude changes the numerical
  contract and needs a separate explicit approximation and validation.

## Classification and evidence

- **Adopt now:** existing public batching, sampler reuse, importance weighting,
  ESS/time tuning, and warm synchronized measurement.
- **Prototype validated in this task:** route to the existing reference path
  instead of dense transfers for the reproduced oversized-cache case.
- **Proposed:** memory/work routing guard, grouped native validation/draws,
  safe scaled diagnostics, reusable array expressions and factored rows.
- **Defer:** general native boundary batch architecture until focused kernels
  have accuracy and memory tests. No speedup is claimed for unimplemented work.
- No new compatibility shim or dependency upgrade is needed for this assessment.

Temporary scripts and logs:

- /tmp/peps_sampling_performance.py and performance_{pilot,cpu,cuda,opt}.log
- /tmp/peps_sampling_performance.jsonl
- /tmp/peps_sampling_cache_probe.py and .log
- /tmp/peps_sampling_su_performance.py and .log
- /tmp/peps_sampling_sync_profile.py and .log
- /tmp/peps_perf_cpu_batch.txt and .prof

The scripts were run with the active py312 environment, PYTHONPATH=src:/tmp,
OPENBLAS_NUM_THREADS=1, and OMP_NUM_THREADS=1. The SU probe additionally adds
the existing sibling roughening benchmark directory to PYTHONPATH and uses
its source read-only. All generated scripts/data remain under /tmp.

No numerical package suite was rerun for this documentation-only assessment.
The new numerical evidence consists of these performance/quality probes and
the direct route-equivalence assertions; previous test results are recorded
separately in the backend-fix note. No commits, publication, dependency
updates, or changes to production jobs were made.

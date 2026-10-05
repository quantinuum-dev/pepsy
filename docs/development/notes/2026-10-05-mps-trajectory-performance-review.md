# 2026-10-05 — MPS trajectory performance and API review

Review of `develop` at `bec773a` plus the uncommitted
[trajectory corrections](2026-10-05-mps-trajectory-corrections.md). These are
recommendations, not implemented optimizations. Production code was unchanged
by this review. “PAI” in the request was interpreted as “API.”

## Measurements

Python 3.12, NumPy complex128, CPU, one OpenBLAS/OMP thread, existing development
environment and dependency versions from the linked correction audit. Timings
exclude imports and use warm-up followed by three repeats (15 for the prototype).
They do not establish accelerator, native Symmray, large-bond, or MPI performance.
Temporary reproduction scripts: `/tmp/pepsy-trajectory-performance-review.py`
and `/tmp/pepsy-trajectory-performance-prototypes.py`, with corresponding logs.

End-to-end case: eight-site computational zero state, H on all sites, seven
successive nearest-neighbor CNOTs, then amplitude damping with probability 0.15
on sites 0–3; direct mode, chi=8, 64 shots, seed=15, retain=none, progress off.
This is a deliberately small, low-entanglement workload: the CNOT prefix acts
on a product of plus states. Coalescing has at most 16 outcome histories.

| Strategy | Workers | Median seconds | Min–max seconds |
| --- | --- | ---: | ---: |
| Independent | 1 | 1.020 | 1.005–1.022 |
| Independent | 4 | 1.254 | 1.236–1.289 |
| Independent | auto (38 here) | 1.275 | 1.202–1.317 |
| Coalesced | 1 | 0.0260 | 0.0259–0.0262 |
| Coalesced | 4 | 0.0308 | 0.0294–0.0355 |
| Coalesced | auto (38 here) | 0.0318 | 0.0293–0.0344 |

Existing coalescing is approximately 39 times faster in this favorable case;
automatic threads are approximately 25% slower than serial independent replay.
This does not support a universal serial-worker policy.

Kraus microbenchmark: normalized random complex vector of eight qubits (seed
142), converted to an exact MPS, chi=16; four computational-basis projectors
on a two-site support. Current probabilities matched dense contraction within
1e-12. Nonadjacent support `(0, 7)` performs eight `swap_site_to_` calls:
two support-placement calls for each of four outcomes, including the no-op
placement of site zero. Thus the expensive gathering is repeated four times.

A temporary NumPy prototype gathers/contracts/normalizes one amplitude block
and applies every Kraus matrix to it. Medians:

| Support | Current probability helper | Shared-block prototype |
| --- | ---: | ---: |
| `(0, 1)` | 0.642 ms | 0.187 ms |
| `(0, 7)` | 7.764 ms | 1.965 ms |
| `(7, 0)` | 7.764 ms | 1.957 ms |

Prototype probabilities agreed with the current implementation within 1e-12;
the input vector remained unchanged within 1e-12. The prototype omits general
dispatch, represented-norm handling, and production diagnostics. Its roughly
fourfold nonadjacent-kernel improvement is an opportunity estimate, not a
production or end-to-end speedup claim. Rare-outcome, backend, native-array,
layout and exact-mode contracts still require implementation tests.

In a separate warmed cProfile run of 32 independent shots, total profiled time
was 1.212 s. Fresh optimizer construction accounted for 0.164 s cumulative
(13.6%), `set_gates` for 0.046 s (3.8%), and trajectory compilation for 0.011 s
(0.9%). These cumulative categories overlap and must not be added. Compilation
was called 193 times, including segment installation and empty constructors;
most calls do not compile the full original stream. Instrumenting 16 shots
counted 97 calls serial and 114 with four workers. Compilation alone is not
the dominant bottleneck in this workload.

For empty replay on a two-site state, Python tracemalloc peak allocations with
retain=none were 0.45/2.21 MiB for 256/2048 serial shots and 0.75/5.34 MiB with
four workers. These are traced allocations, not process RSS or device memory.
Both paths allocate all seed pairs; parallel dispatch also materializes input
work, futures, per-shot results and diagnostic dictionaries before reduction.

## Proposed implementation order

1. **Reuse the Kraus amplitude block per event.** Split support preparation
   from outcome application in `_mps_projected_kraus_probability` and
   `_kraus_probabilities`. Preserve projected amplitudes, untruncated routing,
   physical-leg order, and scoped precision; do not restore cancellation-prone
   Gram probabilities. Later batch outcome norms on-device and transfer the
   small probability vector once. GPU benefit remains unmeasured.
2. **Make local execution bounded in memory.** Generate seeds incrementally
   with the existing seed sequence, bound in-flight work, and fold diagnostics
   as shots complete. Retain all states only when requested. Suppress optional
   history construction for final/none retention while preserving measurement
   records needed by feed-forward and leakage. Reuse an executor across
   coalesced branching stages instead of creating one on every ordered map.
3. **Prepare a canonical shot template once.** `_shot_factory` invokes the
   constructor and canonicalizes each shot. Reuse the trusted branch-clone
   mechanisms after preparing an isolated template. Preserve constructor-state
   restart semantics, subclass initialization, owned arrays and metadata.
4. **Carry compiled plans through every runner.** The parallel runner passes
   raw entries back to `run_trajectory_shots` per shot. Reuse immutable plans
   and validated ordinary segments where safe. `ordinary_segments` is computed
   but not consumed by execution. Keep dynamic control/cap/layout handling and
   external factory compatibility. This is a smaller measured opportunity
   than template preparation for the tested workload.
5. **Improve automatic worker selection.** CPU availability alone is not a
   cost model for small MPS replay or a shared accelerator. Benchmark across
   bond sizes and event mixes before choosing thresholds; expose resolved
   worker count and selected strategy. Existing coalescing should remain the
   first shared-work optimization when branch counts are small.

## Proposed API improvements

- Add explicit `run_trajectories(...)` while retaining `run(...)` compatibility.
  Currently even `workers=1` triggers ensemble dispatch for an otherwise
  ordinary single-shot call. Make restart-from-initial-state semantics explicit.
- Group numerical, trajectory and execution options with typed option objects;
  retain compatibility keywords. Reconcile `workers`/`parallel_workers`, and
  make `run_kwargs` override precedence visible.
- Support local `observable` reduction and `chunk_size`, with the same bounded
  execution model used for MPI. The current MPS facade rejects observable
  callbacks unless MPI is enabled; local retain=none therefore cannot provide
  this standard result-consumption path. Return count/weight-aware uncertainty
  as well as the existing mean and ESS; coalesced leaves are not individual
  equally weighted samples, and ESS is not a standard error.
- Put terminal sampling on the stable `NoisyResult` surface for both strategies.
  Currently `sample_bits` is forwarded only when the raw coalesced result has
  it, so automatic strategy selection changes method availability. Preserve
  logical ordering, ragged-register lengths, counts and importance weights.
- Add requested/resolved strategy, worker count, fallback reason and discarded
  coalescing work to diagnostics. Existing `coalesced` identifies final strategy
  but does not explain why automatic selection or restart chose it.

## Validation and limits

Seeded independent runs with one/four workers produced identical outcome-label
histories and dense final states within 1e-12 for 16 shots. Dense probability
references and prototype state preservation passed as described above.
Documentation links and `git diff --check` passed. No production implementation
changed, and no numerical test suite was rerun for this review. Previous test
successes and unresolved failures remain scoped to the linked correction audit.
Nothing was staged, committed or published.

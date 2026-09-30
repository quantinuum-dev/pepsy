# 2026-09-26 — 9×9 and 10×10 D=4 sampler resource probes

## Scope and method

User requested a larger PEPS corner case to expose memory or CPU problems.
These are measured CPU resource probes, not large-system accuracy certification
or production-throughput benchmarks. No sampler implementation or defaults were
changed. Existing exact CPU and DMRG GPU jobs continued. Two 9×9 probes briefly
overlapped; the timing comparisons are indicative, not isolated benchmarks.

- NumPy complex128, one BLAS/OpenMP thread per probe; CUDA hidden.
- Evolved diagonal product wall, five second-order Ising steps of dt=0.2,
  J=-1, hx=1; simple update capped at D=4, observed maximum bond 4.
- Quimb-MPS future boundaries, chi=16, chi_prime=8, automatic cutoffs,
  rho_positivity="absolute". These finite caps define an approximate proposal.
- Reference row path versus opt-in factored cache with 64 MiB estimated-cache
  budget. Four proposal draws were attempted in two chunks of two, using the
  private boundary-draw method to separate proposal cost from exact amplitudes.
  No placeholder amplitudes or fabricated sample results were returned.
- Successful amplitude probes then ran public `sample_batch(2, chunk_size=1)`.
- Each subprocess had a 16 GiB address-space limit, 150-second soft CPU limit,
  155-second hard CPU limit, and 180-second wall timer. Limits belong to this
  diagnostic harness, **not** to the sampler API.
- Exact plans were inspected before execution; plans exceeding a 2 GiB largest
  tensor or 5e10 estimated contraction operations were not executed. Largest
  planned tensor is not a bound on total workspace or peak process memory.

The initial planner was explicitly `greedy`, matching the existing benchmark.
**PepsSampler's default planner is `auto-hq`, which was not tested here.**
The slow greedy reference path is not evidence that the default planner has
identical cost.

## Results

| Case | Setup | First two proposal draws | One exact amplitude | Public two-shot batch, chunks of one | Peak process RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9×9 reference, greedy | 3.05 s | 69.52 s | Not reached | Not reached | 1325 MiB |
| 9×9 factored, greedy | 2.43 s | 1.00 s | 0.48 s | 2.12 s | 651 MiB |
| 10×10 factored, greedy | 3.61 s | 1.72 s | Preflight declined | Not run | 599 MiB |
| 10×10 factored, alternate exact plan | 3.65 s | 1.74 s | 6.89 s | 15.29 s | 781 MiB |

Setup excludes evolution and imports. Stage timings include Python overhead;
exact amplitude timing includes scaled-leaf/cache preparation. Peak RSS includes
imports, evolution, setup, all draws, and amplitude work in that process; it is
not a CUDA measurement or the retained-cache size.

The reference run completed its first two draws and reached the **150 CPU-second
limit during the second proposal chunk**. No OOM occurred. At interruption,
144.50 seconds had been spent in local conditional contractions and 0.23 seconds
in conditioned-boundary updates. This is a CPU-cost issue under the chosen
planner, not thread proliferation: measured CPU seconds followed wall seconds.

Factored sampling completed all four proposal draws in 1.98 seconds for 9×9 and
3.46 seconds for 10×10. Maximum live prefix groups were two. Estimated factored
cache sizes were 31.43 MiB and 36.19 MiB, respectively; actual process peaks were
higher because the cache budget does not cover all tensors/workspace.

For the first shared-seed 9×9 draws, reference and factored log proposal
probabilities differed by at most **2.85e-14**. Public batch log probabilities,
scaled amplitudes, and weight diagnostics were finite in both successful full
runs. Two-shot ESS is not a convergence or support check. No dense Born oracle
was constructed for 81 or 100 sites.

## Exact-amplitude planning is material at 10×10

The greedy 10×10 plan had width 28, a **4 GiB** largest complex128 intermediate,
and estimated contraction cost 3.62e11. The harness declined to execute it.
For comparison, the 9×9 greedy plan had width 20 and a 16 MiB largest tensor.

An explicit `cotengra.RandomGreedyOptimizer(max_repeats=32, parallel=False,
seed=17)` found a 10×10 width-22 plan in 0.013 seconds, with a **64 MiB** largest
tensor and cost 3.72e9. This exact plan was passed to the initial amplitude
contraction, then reused by the sampler's existing amplitude cache. Proposal
contractions retained the greedy setting. The subsequent public two-shot batch
completed in 15.29 seconds. This changes contraction order, not the target or
truncation. It is a successful probe, not a new planner default or separate
public amplitude-planner option.

## What remains

1. Evaluate the default `auto-hq` planner on these shapes, and consider a separate
   amplitude-planner setting plus cost/memory preflight or exact slicing. Chunking
   alone cannot bound an exact PEPS contraction's workspace.
2. Evaluate conservative automatic factored-cache selection across workloads;
   these wide cases benefit, while earlier small cases regressed. Keep the
   existing opt-in default until that decision is supported more broadly.
3. Repeat large cases on an isolated GPU and with more entangled evolved states,
   larger caps, more chunks, and independent numerical comparisons. These short
   probes do not establish long-run memory stability or GPU throughput.

Diagnostic script and raw results are under `/tmp/peps_large_probe.py` and
`/tmp/peps-large-{9-reference,9-factored,10-factored,10-factored-random-greedy}.json`.
Those temporary files may disappear; the configurations and principal findings
are retained here. The harness added an alternative-planner flag while the
reference process was running; traceback source line text for that temporary
file can therefore differ from the executed reference code. The resource-limit
exception and sampler stack identify the interrupted conditional contraction.

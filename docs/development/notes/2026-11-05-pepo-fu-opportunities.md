# PEPO full-update performance opportunities

**Review/prototype:** preserve sequential full update (FU) and measure its
environment work before extending independent-cluster batching. No maintained
numerical implementation or default changed in this investigation. The earlier
cluster batching change is committed and pushed as `aadd8eb` on `develop`.

## Independence differs from the cluster method

FU reads a global reduced norm environment. Disjoint six-site patches do not
establish independent reads and writes after a truncation. A seeded exact
contraction check on a random complex128 3x8 PEPO, D=2, used active bonds
((1,1),(1,2)) and ((1,5),(1,6)). The normalized remote reduced environment
changed by 3.44e-16 after an exact unitary update and **0.00810** after the
truncated FU update. These are relative Frobenius differences, not fidelity
errors. Thus simultaneous updates against frozen environments would be a
separate approximation, even when the cluster footprints are disjoint.

Independent PEPO trajectories can instead batch compatible contractions without
sharing evolving state. Within one environment, the two transverse boundary
fronts and the strip prefix/suffix fronts are independent until combined.
Their mutable `CompBdy`/cache state needs separate ownership and validated
merging; threading one existing `CompBdy` instance is not safe.

## Mature checkpoint measurements

Read a copy of the archived 10x10 D=6, boundary-chi56 FU checkpoint from the
sibling Eli benchmark. Use complex128 CUDA on the RTX A5000, one host thread,
greedy contractions, native `gesvd`, direct sequential BRA/KET boundary
compression with cutoff 1e-12, Quimb pair ALS capped at ten sweeps and rtol 1e-8.
Disable extra global normalization/fidelity diagnostics. These controls are
explicit: the default FU boundary fitter is DMRG, not direct compression.

Four extra lower-leg diagnostic gates traverse row 4, columns 3--6. This is a
bounded probe on a mature saved state, not the original circuit chronology or
a new production trajectory. Synchronized public-driver timings:

| Gate | Pair preparation (s) | Environment (s) | Pair optimization (s) | Boundary builds / reuses |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.2541 | 19.0663 | 0.2359 | 9 / 0 |
| 2 | 0.0156 | 0.7186 | 0.0829 | 0 / 9 |
| 3 | 0.0135 | 0.7208 | 0.0652 | 0 / 9 |
| 4 | 0.0137 | 0.6312 | 0.0688 | 0 / 9 |

After the first gate, each strip update reuses seven partial contractions and
rebuilds one. Peak allocated CUDA memory is 1.69 GiB. Cold boundary-front
parallelism would not accelerate the already reused transverse cuts in gates
2--4. A cold-only 19-second result must not stand in for steady-state gate cost.

A second four-gate run separates environment stages. On warm gates 2--4,
transverse cut validation/reuse takes 0.0076--0.0091 s, strip prefix/suffix
reduction takes 0.027--0.033 s, and the final norm contraction takes
**0.589--0.677 s**. Whole-strip preparation, including both earlier stages,
takes 0.045--0.049 s. The final contraction is therefore the warm-path target;
it still has fresh pair reduction factors and must be recomputed.

An earlier standalone probe omitted `StripEnvironmentCache` and exhausted the
24-GiB device in an unreduced final strip contraction. The public driver
already supplies this cache; rerunning that path succeeded. This failure is
not evidence of a default-driver memory regression. Separating exact strip
reduction from persistent reuse could improve the standalone/non-reuse path,
but is a separate, unimplemented refinement.

## Guarded Torch ALS probe

A process-local adapter maps the existing reduced FU problem to the shared
guarded Torch solver, keeping the same metric, target, initial factors, sweep
budget and stopping rule. It does not enable a public FU `solver='torch'`
option. The 576x576 metric has minimum eigenvalue about 1.82e-15; the conservative
certificate selects **zero Cholesky solves and four eigensolves** for the
ungauged two-sweep solve.

Two interleaved microbenchmarks give different Quimb timings: ungauged Quimb
pair optimization 0.0807 s versus Torch 0.0868 s in the first; 0.1408 s versus 0.0865 s
in the second. Each reports a median of three warm repetitions. This variation
does not establish a stable backend speedup. Existing environment gauging
reduces both solvers to one sweep in this sample, but adds conditioning work;
guarded Torch still uses its spectral fallback.

Ungauged local fidelity agrees within 3.4e-16, while the contracted local pair
differs by roughly 8e-5--1e-4 in unweighted Frobenius norm. Gauged candidates
differ much more from the ungauged result (about 0.83) despite low weighted
loss. The ill-conditioned local metric makes local fidelity insufficient to
claim equivalent physical outputs. No default solver/gauge replacement is
supported by this check; a physical observable/reference study is still needed.

## Measured final-norm path opportunity

On gate 2 of another copied-checkpoint probe, retain the identical ten-tensor
final norm network and its requested output order. Compare greedy with
Cotengra `RandomGreedyOptimizer(max_repeats=64, parallel=False, seed=941)`.
Choose the candidate only when its estimated FLOPs improve and its largest
intermediate does not exceed greedy's. Alternate the two fixed trees over four
repetitions, discard the first and synchronize CUDA around each contraction.
The evolving diagnostic state still receives the original greedy result.

| Final norm path | Median seconds | Estimated FLOPs | Largest intermediate elements |
| --- | ---: | ---: | ---: |
| Greedy | 0.67103 | 32,912,098,848 | 16,422,912 |
| Randomized greedy, 64 trials | 0.05216 | 1,957,872,384 | 4,105,728 |

Planning both trees costs 0.00170 s in this sample. The contraction kernel is
**12.86x faster**, with one quarter of the largest-intermediate elements.
Normalized environment matrices differ by 3.89e-15 in Frobenius norm. Both
compute the same contraction up to floating-point order; no chi, boundary
fitting policy, target or convergence tolerance changes.

This is a **measured prototype on one warm graph**, not an integrated FU option
or an end-to-end speedup. An integration should select/cache a norm-specific
path separately from the boundary optimizer, check memory limits, and validate
sequential trajectories, both PEPO legs, PEPS, supported backends and numerical
conditioning. Replacing the global `contraction_opt` would also alter boundary
work and is not the change tested here. The near-singular solver comparison
above still motivates physical/reference checks after changing contraction
order.

## Classification and next experiments

- **Adopt:** retain the existing source-validated transverse and strip reuse,
  exact-rank skips and optional same-bond opposite-leg fusion. These are
  already implemented, not new opportunities.
- **Prototype, first priority:** integrate guarded norm-specific contraction
  planning/cache and validate full trajectories. The identical-operand test
  above targets the measured warm-path bottleneck without simultaneous gates.
- **Prototype:** batch shape-compatible boundary kernels across independent
  trajectories, or separate boundary fronts when both must be rebuilt.
  Bound memory and preserve per-member compression/convergence decisions.
- **Defer:** frozen-environment simultaneous FU gates, automatic changes to
  boundary chi/sweeps/tolerance, and replacing the default local solver.
  They need a separate accuracy study and are not drop-in cluster batching.

## Evidence and limits

Temporary scripts/results under `/tmp`:

- `pepsy-fu-opportunities-20261105.py`: exact remote-environment dependency.
- `pepsy-fu-opportunities-assessed-20261105.py`: cached mature environment and
  process-local solver comparison.
- `pepsy-fu-gauge-probe-20261105.py`: fixed-problem gauge/solver comparison.
- `pepsy-fu-stream-probe-20261105.py`: four-gate public-driver timing.
- `pepsy-fu-stage-probe-20261105.py`: the same sequence with environment stages.
- `pepsy-fu-norm-path-probe-20261105.py`: bounded planning and identical-operand
  contraction comparison; `mature-norm-path.json` records its result.
- `pepsy-fu-opportunities-20261105/`: JSON results and fixed-problem tensors.

CUDA probes require device access outside the restricted sandbox. The active
environment and installed upstream versions are unchanged from the
[earlier audit](2026-11-05-independent-cluster-batches.md#upstream-and-validation):
Torch2.14.1+cu126, Quimb1.15.1.dev101+g1b64353d9,
Autoray0.11.1.dev14+g014a3f69a, Cotengra0.8.3 and
Symmray0.4.1.dev25+gdc377f033. No dependency or installed source was modified.
No production job, frozen source or archived checkpoint was changed.

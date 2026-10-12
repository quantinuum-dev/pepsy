# 2026-10-06 — Real PEPS boundary engine comparisons

User scope: check Quimb MPS, one-site DMRG, and mixed DMRG2 boundaries on real
contractions and optimizations. Branch `develop`, baseline `8f7c896`; existing
uncommitted PEPS/solver work preserved. No production behavior changed.
Used the maintainer router and tensor-fitting guidance and reused the
unchanged installed-environment audit from the preceding PEPS reviews.

## Configuration distinction

| Configuration | Settings |
| --- | --- |
| Quimb MPS | `boundary_engine="quimb-mps"` |
| One-site FIT | `boundary_engine="dmrg", fit_mode="dmrg"` |
| Mixed two-/one-site FIT | `boundary_engine="dmrg", fit_mode="dmrg2"` |

`dmrg` canonicalizes to boundary FIT mode `eff`. `dmrg2` is a fit mode, not
a separate boundary engine. Quimb uses its own reusable environment store;
its public norm/fidelity metric route is `method="mps"`. One-site/mixed FIT
metrics use `method="dmrg"` with the corresponding fit mode.

## Real optimization results

CPU Torch complex128, normalized random 3x3 D=2 PEPS seed 311, horizontal RZZ
angle .37 on ((0,0),(0,1)) and vertical RZZ angle .23 on ((1,1),(2,1)). Both
gates form one automatic batch. Retained D=2; NLopt LD_LBFGS maxeval=50;
ten FIT sweeps; SRC seed zero; greedy contractions. Dense target reference
constructed independently by exact gate splitting with cutoff zero.

The exploratory comparison used one y/x cycle with one round trip per axis,
boundary caps (8,10), and normalization/evaluation caps (32,48):

| Configuration | Warm-start infidelity | Exact final infidelity | Seconds |
| --- | ---: | ---: | ---: |
| Quimb MPS | .12578756526312063 | .0562904175916209 | 2.82 |
| DMRG | .12069355695861828 | .05671813610981824 | 3.33 |
| DMRG2 | .12069355695861694 | .05671813610983689 | 3.44 |

All runs improved, preserved the original input, respected D=2, and had dense
output norm error below 2.6e-15. Reported final infidelities matched exact
dense results within 3e-15. Repeating all three with boundary caps (32,48)
gave final infidelities differing by less than 2e-14. Warm starts differ
between engine paths, so these figures do not establish an optimizer ranking.
Timings are individual exploratory observations, not a controlled benchmark.

New permanent regression tests also ran all three configurations with the
**full default four round trips per axis**, unchanged maxeval=50, checking
strict improvement, exact final fidelity, unit norm, retained bond dimension,
Torch dtype, input ownership, and engine-specific FIT diagnostics. DMRG2
diagnostics confirmed two block sweeps followed by eight one-site sweeps.

## Boundary-cap accuracy

CPU NumPy complex128, normalized random 4x4 D=2 PEPS seed 317, target formed
by exact RZZ angle .31 on ((1,1),(1,2)). Exact dense infidelity is
**.08798063763679731**. Tested both x/y directions at caps 4, 16, and 64,
ten FIT sweeps, SRC seed zero, max_separation=0, scaled contractions, and
known target norm one. Quimb uses sequence `(axis + "min",)`; FIT uses its
directional boundary sweep. These are each route's own compression policy,
not an identical intermediate factorization.

Maximum absolute infidelity error across the two directions:

| Configuration | cap 4 | cap 16 | cap 64 |
| --- | ---: | ---: | ---: |
| Quimb MPS | 8.64e-2 | 6.41e-4 | 9.11e-15 |
| DMRG | 2.14e-3 | 7.78e-15 | 7.00e-15 |
| DMRG2 | 2.01e-3 | 7.00e-15 | 8.11e-15 |

At cap 64 all norm errors were below 7e-15. All eighteen corrected probe
cases completed. An initial probe formatting error attempted `complex()`
on the documented `(mantissa, exponent)` result; correcting the probe to
reconstruct that scalar resolved it without library changes. Permanent
tests cover exact norm and known-target fidelity for all three configurations
in both directions at cap 64.

Quimb metrics took roughly .04-.06 seconds per norm-plus-infidelity case;
FIT took .22-.30 seconds. This one fixture shows faster Quimb evaluation but
different accuracy at an equal cap. No universal speed/accuracy ordering is
claimed. An insufficient cap is an approximation limitation, not evidence
of incorrect index contractions.

## Local objective and cache checks

Independent 3x3 D=2 Torch complex128 fixture, seeds 291/293, boundary cap 32,
ten FIT sweeps, SRC seed zero. Across 14 row/column updates per configuration,
including all direction changes and reused boundaries after changing earlier
slices, the local loss matched dense fidelity within 1.11e-16. Complex
directional gradients matched centered finite differences within 4.04e-12.
The probe applies deterministic perturbations, rather than mocking numerical
contractions; it tests objective/cache correctness, not solver convergence.
Real solver convergence is tested by the optimization regressions above.

## Validation and limits

- Existing focused selection from `test_optimize_peps.py` and
  `test_prepare_boundary_inputs.py`, matching Quimb/DMRG2/real boundary-pair
  sweeps/native fermionic tests: **40 passed**, 264 deselected, two warnings.
  Includes real U1/U1U1 Quimb-boundary optimization in both axes.
- New `tests/test_peps_boundary_engine_numerics.py`: **3 full-default-schedule
  optimization cases passed**; **6 exact 4x4 metric cases passed** (separate
  selections). No skipped cases in these runs.
- `python -m ruff check src tests` and `git diff --check`: passed.
- Added tests and this report/history only. Nothing staged, committed, or
  pushed. No full suite, GPU, or broad complex64 engine comparison.

No new engine-specific correctness bug was reproduced. The preceding
[FIT/sweep review](2026-10-06-peps-fit-sweep-review.md)'s negative-loss early
exit and missing local returned-candidate guard remain unfixed; successful
cases here do not resolve those failure-handling issues.

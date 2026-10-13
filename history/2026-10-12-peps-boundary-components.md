# 2026-10-12 — Separate PEPS norm and overlap errors

The saved D=3 low-chi failures are dominated by norm errors, not overlap
errors. Changing the boundary method or initialization alone does not fix
them at `(18,36)`.

- Scope: identify the unreliable contraction and compare boundary methods.
- Branch / baseline: `develop` / `287bff6`; diagnostic record only.
- Fixtures: frozen failed metric inputs from the ongoing
  [TFIM comparison](2026-10-12-peps-tfim-zero-state-review.md), at the attempted
  ALS step t=1.2 and global L-BFGS step t=1.8.
- Publication: this record accompanies the session's authorized commit/push.

## Actual routes

ALS uses the Pepsy `dmrg` boundary engine, `fit_mode='eff'`, joint layers,
`fit_init_strategy='reuse-direct'` and ten boundary FIT iterations. Its moving
boundaries reuse compatible guesses; standalone metric calls construct fresh
boundaries. It does not use `guess-src` by default.

The global trajectory uses Quimb direct compression with explicit BRA/KET
layers **inside its differentiable objective**. Its independent normalization
and acceptance checks use Pepsy `eff` FIT with `guess-src`. The engine name
`dmrg`, FIT mode and initialization are separate controls.

## Component errors at `(18,36)`

Evaluate candidate norm N, target norm T, and squared overlap |O|² separately
against exact dense contractions of the same frozen tensors. Entries below
are signed relative errors in percent; the metric divides |O|² by |N| |T|.

| Fixture | N error | T error | Norm-product error | Squared-overlap error |
| --- | ---: | ---: | ---: | ---: |
| ALS failure | −0.088415% | −0.107839% | −0.196158% | −0.009935% |
| Global failure | −0.500478% | +0.067315% | −0.433500% | −0.005678% |

The underestimated denominator drives fidelity above one. Substituting only
the exact norms makes both infidelities positive; substituting only the
exact overlap leaves both negative. Neither physical state is non-finite.

## Frozen-state method comparisons

| Method / caps | ALS-fixture infidelity | Global-fixture infidelity |
| --- | ---: | ---: |
| Exact dense reference | 0.000454642231 | 0.002467794688 |
| Actual FIT policy, `(18,36)` | −0.001410395370 | −0.001818448935 |
| Pepsy direct BRA/KET, `(18,36)` | −0.001837233985 | −0.004056015963 |
| Quimb MPS BRA/KET, `(18,36)` | −0.002374327271 | −0.012893253401 |
| Actual FIT policy, `(36,144)` | 0.000351184125 | 0.002745319094 |
| Actual FIT policy, `(81,144)` | 0.000454612675 | 0.002467794688 |
| Quimb MPS BRA/KET, `(81,144)` | 0.000454642231 | 0.002467794688 |

Quimb agrees with dense references to roundoff at `(81,144)` on both inputs.
Pepsy FIT's ALS-fixture residual is about 3e-8. Increasing the norm cap again
to 162 gives the same results. Raising only the norm cap from 18 to 81 while
keeping overlap cap 36 already removes both negative estimates, but leaves
measurable overlap error.

At low caps, switching from `reuse-direct` to `guess-src`, or selecting
`fit_mode='dmrg'` with `guess-src`, does not resolve the failures. Increasing
`eff` iterations from 10 to 40 with direct initialization also does not resolve
them. The direct/Quimb methods perform worse at those same low caps on these
fixtures. Comparisons use each route's default sweep geometry, so they are not
an isolated algorithm ranking or a derivative/trajectory validation.

Some cap pairs have compensating errors: `(36,36)` looks more accurate than
`(36,144)` on the ALS fixture because norm and overlap errors cancel. Check
the three contractions separately rather than selecting chi from fidelity
alone. These measurements do not prove 81 is sufficient for all D=3 states.

## Evidence and limits

Forty-four fixed-input contractions completed across two fixtures, seven
boundary configurations and additional independent norm-cap scans. Scripts
and JSON are `/tmp/pepsy-tfim-d34-pCJAS1/compare_boundaries.py`,
`boundary-comparison.json` and `norm-cap-scan.json`. The environment/upstream
audit is unchanged. No solver defaults, installed dependencies or background
queue settings were changed. No new full-suite claim; `git diff --check`
is the documentation check.

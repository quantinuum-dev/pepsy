# 2026-11-17 — Eli FU ALS convergence audit

- Scope: explain and investigate whether the 23-spin D71 FU ALS is trapped in
  poor local solutions, following the lower accumulated fidelity at chi=5.
- Branch / baseline: `develop`, `328979f`; initially clean working tree.
- Commit status: audit record included with the subsequent local ALS restart
  feature commit. Numerical code, saved campaigns and live workers were
  unchanged during this audit.

## Implementation findings

`optimizers/peps/_full_update.py` prepares an SVD start and a positive reduced
norm environment, then uses `bp/reduced_update.py` to call Quimb's native
two-factor ALS. Each full sweep uses dense Hermitian eigensolves with a
relative spectral floor of 1e-12 and then QR/LQ balancing. The tested files
match the frozen D71 campaign sources byte for byte, including the shared
reduced solver. Installed Quimb is 1.15.0; Torch is 2.14.1+cu126.

The downstream Eli runner explicitly sets `rtol=1e-8`, overriding Pepsy's
complex128 automatic 1e-9. It stops when either the squared residual divided
by the target norm or its absolute full-sweep change is at most that
tolerance. `cost_tolerance` establishes small improvement, not stationarity
or global optimality. The wrapper retains the SVD guess if the final ALS
candidate is worse; the shared ALS loop does not restore the best intermediate
sweep. No meaningful nonmonotonicity appeared in the controlled probes.

The production local JSONL retains only the wrapper's initial/final
`loss_history`, although the optimizer also supplies per-sweep `solver_costs`.
It does not save the original intermediate pair inputs, so exact historical
restart comparisons are unavailable from these files alone.

## Historical evidence

D71 chi=4 and chi=5 each performed 88,956 matching pair updates: 72 exact SVD
updates and 88,884 ALS fits. Neither reached the 20-sweep cap. Chi=5 had
84,932 residual-threshold stops and 3,952 cost-change stops. The latter
account for 87.47% of the sum of local infidelities. This motivates testing
the harder fits but does not establish trapping. Raw norm environments had
positive minimum eigenvalues and no negative spectral weight in both runs.

## Controlled experiments

Loaded a copy of the completed D71 chi=5 final PEPO. Applied a new upper-leg
ZZ gate with angle 6/71 on three bonds selected from bonds with large
historical errors. These are representative new fits, **not replays of the
original intermediate updates**. Each pair used a freshly built fixed
boundary-cap-64 DMRG environment with three sweeps, complex128 Torch CPU,
one thread, and the campaign's exact SVD policy. Each reduced metric is 100x100.

| Bond | Production residual / sweeps | Tight residual / sweeps | Relative improvement |
| --- | --- | --- | --- |
| (3,2)-(4,2) | 6.36331232e-7 / 3 | 6.35829463e-7 / 8 | 0.07885% |
| (2,3)-(3,3) | 4.58045152e-7 / 3 | 4.58044932e-7 / 5 | 0.0000479% |
| (3,2)-(3,3) | 5.34543187e-7 / 3 | 5.34358709e-7 / 7 | 0.03451% |

Tight controls use `rtol=1e-13`, with a 100/200-sweep cap. On every bond,
three 10% perturbations of the SVD start and three independent random starts
converged to essentially the same tighter objective. Weighted-QR ALS and
environment-gauged ALS agreed. L-BFGS after tight ALS added negligible
improvement. Forcing 200 sweeps reduced normalized gradient norms to
approximately 1.3e-14–1.6e-14 without a material further objective reduction.

An identity-metric check matched the independently computed truncated-SVD
global optimum: 0.00401355535368106 versus 0.004013555353681059.
This validates that control case; the environment-weighted rank-constrained
objective remains jointly nonconvex.

Evidence and reproducible probe/replay scripts are under
`/tmp/pepsy_als_audit_20261117/`, especially `summary.json`,
`historical-stats.json`, `probe.py`, and `replay.py`. The scripts use final
saved inputs under the benchmark's `store/eli_global_cost/` D71 campaigns.

## Validation and limits

Shared Python 3.12 environment, local Pepsy source, one CPU thread:

`pytest -q -o addopts='' tests/test_peps_full_update.py -k 'shared_solver_routes or can_disable_automatic_stopping or reduction_preserves_exact_gate'`

Result: 12 passed, 55 deselected, one existing Quimb default-SVD warning.
The full suite was not run. This is evidence against poor ALS minima in
these three probes, not a certificate for all historical pairs or the global
PEPO. Tighter full-run tolerances and larger boundary caps remain unmeasured.
Capturing original difficult pair inputs and exporting per-sweep costs would
make a broader audit possible without rerunning entire trajectories.

## Follow-up: joint L-BFGS from SVD and random starts

The user requested joint L/R L-BFGS and two types of initialization. Compared
ALS and joint reduced L-BFGS on the same three frozen targets/environments,
using the original SVD factors and two independent complex Gaussian factor
pairs (seeds 201/202), each rescaled to its SVD factor's Frobenius norm.
Measured medians of three one-thread CPU solves after untimed solver warmups;
environment and initialization construction are excluded.

| Solver / start | Iterations | Solver time | Normalized residual |
| --- | --- | --- | --- |
| Tight ALS / SVD | 5–8 | 17.8–25.5 ms | 4.58045e-7–6.35829e-7 |
| Tight ALS / random | 6–10 | 17.1–31.8 ms | Same per-bond objectives within stopping precision |
| Tight L-BFGS / SVD | 1,000 cap | 554–557 ms | 0.028%–0.181% above tight ALS |
| Tight L-BFGS / random | 1,000 cap | 548–560 ms | 2.85e-6–1.25e-5 |

Tight settings use 1e-13 for the respective solver's tolerance; equal numeric
tolerances are not equivalent stationarity/error criteria. All initial
ungauged tight L-BFGS comparisons hit their 1,000-iteration cap. At the
production 20-iteration cap and tolerance 1e-8, SVD-start L-BFGS returned
1.78e-6–2.34e-6 versus ALS's 4.58e-7–6.36e-7; two L-BFGS fits stopped on
function-change tolerance, one hit its cap. No measured random ALS start
meaningfully improves on tight SVD-start ALS.

Also checked the existing environment gauges while holding the *physical*
starting factors fixed. SVD-start joint L-BFGS improved to 348–430 iterations
and 188–226 ms, approximately 7–14 times the corresponding tight ALS solve.
Its residuals remained slightly above tight ALS (about 0.002%–0.004%).
Random-start L-BFGS still hit its cap; gauges did not help those cases.
This supports a conditioning/speed concern for this joint solver; it does
not demonstrate that its capped random runs have reached local minima.

Raw results and reproducible scripts:
`/tmp/pepsy_als_audit_20261117/{solver-comparison,gauged-comparison}.json`,
`compare_solvers.py`, and `compare_gauged.py`. Numerical package code and
production jobs remain unchanged. The joint analytic-gradient/autograd check
and reduced L-BFGS normalization regression both passed (2 tests, 3.76 s).
These CPU reduced-pair measurements do not establish full-trajectory or
full-site-tensor L-BFGS runtime or accuracy.

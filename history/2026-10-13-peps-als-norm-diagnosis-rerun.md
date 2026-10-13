# 2026-10-13 — PEPS ALS norm diagnosis and larger-cap rerun

- Scope: report the overnight D/chi grid, diagnose ALS failures and state-norm
  stability, and launch the requested larger-chi D=3 ALS trajectory to t=5.
- Working branch: `develop`; numerical baseline: isolated `31e9dab` checkout.
- Commit status: this handoff is uncommitted. No package implementation changed.

## Overnight grid

The original worker is absent from the host process table. Its queue status
remains stale at D4-low-sweep. Individual status files show four failed runs,
one completed run and three unstarted D=4 configurations. D3-high-global
reached t=5 with exact circuit fidelity 0.482438571595382.

## Norm and overlap checks

Exact squared norms were recorded at every completed time step. Maximum
absolute deviation from one was 0.001706611 for D3-high-sweep, 0.000923367
for D4-low-sweep, and 0.001097298 for the completed D3-high-global run.
These checks do not show a divergent physical norm. They do show a small
normalization error, consistent with approximate norm-based normalization.

Recontracted all three frozen ALS failure inputs against dense references,
separating candidate norm, target norm and squared overlap. Signed relative
errors at each run's original caps:

| Fixture | Norm-product error | Squared-overlap error |
| --- | ---: | ---: |
| D3-low-sweep, attempted t=1.2 | -0.196158% | -0.009935% |
| D3-high-sweep, attempted t=2.2 | -0.425161% | roundoff |
| D4-low-sweep, attempted t=1.9 | -0.247834% | -0.010718% |

All three negative acceptance infidelities are dominated by underestimated
norm products. Substituting exact norms fixes their sign; substituting only
the exact overlap does not. At the D3-high-sweep failure, exact candidate and
target squared norms are 1.002022447 and 1.001999051, and exact infidelity is
0.003189948659. Caps `(81,144)`, `(162,324)` and `(324,648)` agree with that
exact infidelity to roundoff. On D3-low-sweep, all three pairs agree with
each other but retain about 3e-8 absolute infidelity error. D4 `(256,512)`
reduces the frozen-fixture infidelity discrepancy to about 2e-9.

Evidence: `/tmp/pepsy-tfim-d34-pCJAS1/als-failure-components-20261013.json`.
These are fixed-input contraction checks, not full-trajectory convergence.

## New run

Launched `D3-chi162-324-sweep-20261013` from the zero state, preserving the
original circuit, ALS budgets, single-strip batches, exact layered targets,
SU guesses, measured target norms and strict fixed caps. Fitting/evaluation
caps are `(162,324)` and normalization cap is 162. All 50 steps target t=5;
exact dense fidelity and squared norm are recorded every completed step.

The cap choice was presented to the user as an optional preference. After
allowing a response and completing the component checks, proceeded with the
stated default `(162,324)`. Full trajectory results remain pending.

Persistent user service: `pepsy-tfim-d3-chi162-324-20261013.service`, initially
PID 571453, confirmed active with CPU quota 100% and MemoryMax 16 GiB.
The service owns the run independently of the foreground tool session.
Configuration and log live under the original root; individual live status,
observations and checkpoints are under
`/tmp/pepsy-tfim-d34-pCJAS1/runs/D3-chi162-324-sweep-20261013/`.
The original grid's stale queue status does not describe this standalone run.

Validated the unchanged isolated source path and all 51 exact references
before launch. No numerical implementation tests were needed for this
configuration-only rerun. No commit or push was performed.

## User-requested stop

The user subsequently requested that running jobs stop. Sent SIGTERM to the
rerun's user service; the main process exited. Host checks found no remaining
benchmark Python processes, and the service cgroup has zero tasks. Updated
the individual status file to `stopped`, preserving all logs and checkpoints.
Last completed step is 4 (t=0.4), with exact fidelity 0.999999998210283 and
exact squared norm 1.000000001405726. Step 5 was interrupted. No run was
restarted. The full higher-cap trajectory remains unvalidated.

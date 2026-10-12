# 2026-10-12 — 4×4 TFIM zero-state trajectory review

The requested physical trajectory exposes failures that the earlier random-PEPS
checks did not cover. **Default sweep and JAX global runs do not reach t=5.**

- Scope: run the FullUpdate physical setup, review accuracy and timings.
- Branch / baseline: `develop` / `03568a4`.
- Changes: this validation record only; no numerical implementation changed.
- Publication: included in the session's authorized review commit and push.

## Setup and reference

Use `pepsy_examples/FullUpdate/run.py`'s `geometry`, `step_gates`, and
`dense_z` helpers: 4×4 open boundaries, J=h=1, dt=0.1, 50 requested steps,
initial state `ps_to_peps(4, 4, theta=0)` = |0⟩¹⁶, D=2, CPU complex128.
The Hamiltonian is −Σ ZZ −Σ X. Each second-order step has 16 half-step X
rotations, 24 full-step ZZ gates, and 16 half-step X rotations.

Check the helper's gates against independent analytic matrices. Evolve a
65,536-element vector with `audit_fidelity.apply_dense` in the same snake site
order, without truncation. This is the exact **Trotter circuit**, not exact
continuous-time evolution. All 50 reference steps completed; final mean Z is
0.7379204280665801. Every optimizer's initial dense vector equals the same
all-zero basis vector and has fidelity 1. The first step agrees to roundoff.

Sweep uses its default strip/SU/ALS policy. Global uses its default whole-step
batching and 20 L-BFGS-B iterations per fit, separately with Torch and JAX
autodiff. All use boundary caps `(8, 16)`, greedy contraction and one BLAS/OpenMP
thread. These are different optimizer budgets and batching policies, so the
table is a correctness/accuracy audit, not an equal-work solver ranking.

## Measured results

Fidelity is the squared normalized dense overlap with the independent circuit
reference, not the approximate boundary score. Higher is better.

| Run | Last completed time | Exact fidelity | Outcome |
| --- | ---: | ---: | --- |
| Torch sweep | 1.4 | 0.900336272196050 | Failed while advancing to 1.5 |
| Torch global | 5.0 | 0.326340540500710 | Completed all 50 steps |
| JAX global | 0.1 | 1 to roundoff | Failed while advancing to 0.2 |

At the shared time t=1, sweep fidelity is 0.986999155827626 and Torch global
fidelity is 0.985636129574204. Torch global's final mean Z is
0.809047049163749; its largest site-Z error is 0.231526987804940.

Torch global spent 211.55 s inside optimizer runs and 212.07 s including
measurements/checkpoints, excluding process imports. Sweep spent 276.97 s
through its last completed step; JAX spent 19.29 s through its first step.
Those incomplete timings exclude time spent inside the subsequent failed
steps. Jobs ran concurrently, so these are not isolated speed benchmarks.
Successful checkpoints remained finite and had D≤2. The maximum exact
squared-norm error was 0.006757 for Torch global and 0.001760 for sweep.
Fidelity measurements normalize the dense vectors separately.

## Failures and controlled diagnosis

Sweep raises `ValueError: PEPS infidelity is substantially negative
(-1.063e-03) at evaluation chi=(8, 16)`. It had already rejected four failed
fits across completed steps and retained their warm starts. Replaying the
first three steps with detailed records reproduced two
`no_valid_local_updates` failures at t=0.3. The remaining completed steps
contain valid updates. This does not establish accuracy of the default caps
through the requested trajectory; larger/adaptive caps remain unverified here.

JAX global raises `ValueError: PEPS infidelity must be finite`. Capture the
second-step candidate and fixed target before optimization and evaluate both
backends on these same arrays. The initial approximate loss is finite,
1.20934833e-6, but all 576 JAX gradient entries are non-finite. Disabling
canonicalization still leaves 48 non-finite entries and is not a fix.

The same captured fit with **exact contraction** has finite gradients in both
backends. For one seeded unit direction and centered difference step 1e-5:

| Contraction / backend | Autodiff derivative | Finite difference |
| --- | ---: | ---: |
| Boundary / Torch | −8.76818398e-4 | 4.55009641e-5 |
| Exact / Torch | 4.55009231e-5 | 4.55009197e-5 |
| Exact / JAX | 4.55009231e-5 | 4.55008919e-5 |

This isolates the reproduced gradient problem to approximate boundary
differentiation on this early-time state. Torch reports rank-deficient QR
inputs; the precise decomposition-level cause and a valid remedy are not
established by this review. No backend policy, clipping tolerance, state, or
test expectation was changed to hide a failure. A complete JAX trajectory
and a complete sweep trajectory remain **unvalidated**, not passing.

## Evidence and scope

Temporary scripts, exact vectors, checkpoints, per-step observations, failure
tracebacks and captured fit live under `/tmp/pepsy-tfim-4x4-Lbdhux/`:
`run_validation.py`, `capture_jax_failure.py`, and `probe_gradient.py`.
The compatible isolated environment and upstream audit from the
[publication review](2026-10-12-peps-strip-publication.md) are unchanged.
The [earlier random-state checks](2026-10-12-peps-4x4-review.md) remain valid
for their own fixtures and do not establish this zero-state trajectory.
No example notebook, production output, shared environment or implementation
was edited. No new passing full-suite claim accompanies this numerical audit.

## Follow-up: why ALS stops while Torch global advances

Resume from the saved sweep state at t=1.4, with a fresh driver and the same
next Trotter step. This drops the old driver's cached boundary guesses, so
the reproduced negative estimate is −0.001039 rather than −0.001063; the
failure remains at the post-fit metric check.

The first **ALS fit succeeds and improves exact local infidelity** from
0.003296932518130 to 0.002504557175305. Its summary is
`success=True, termination_reason='partial_local_updates'`: six local
updates apply and ten are rejected. The subsequent approximate postcheck,
not an ALS linear solve, raises. Both candidate and target remain finite.

Two errors combine. The incoming state's exact squared norm is
1.001759938421066, while the unitary shortcut assumes target norm one.
The small boundary caps also introduce contraction error. Holding the failed
candidate/target tensors fixed separates these effects:

| Norm/overlap caps | Target norm policy | Raw infidelity |
| --- | --- | ---: |
| `(8,16)` | Assume one | −0.001038587496458 |
| `(8,16)` | Measure at these caps | −0.001143267913472 |
| `(8,16)` | Supply exact norm | 0.000720083621780 |
| `(16,64)` | Assume one | 0.000749026620631 |
| `(16,64)` | Measure at these caps | 0.002504557175295 |
| Exact dense reference | Normalize both vectors | 0.002504557175299 |

The `(32,64)` result agrees with `(16,64)`. Supplying target norm one
disables automatic metric retries by design, so the default stops when its
raw estimate crosses −0.001. Raising the clipping allowance would hide the
accuracy problem rather than resolve it.

Torch global, started from the **same saved sweep state**, also underestimates
the local error: its post-fit boundary loss is 0.015230112361629 versus exact
0.022152548878851. It compresses the whole step in one batch, whereas sweep
fits strips. Its larger local error stays positive despite the bias, so it
passes the same guard and advances. Completion does not establish better
accuracy or a more reliable metric. Its internal global objective also differs
from the driver's boundary postcheck; the postcheck governs acceptance.

Controlled one-step results from that common t=1.4 checkpoint:

| Sweep configuration | Completed t=1.5? | Exact circuit fidelity | Fit summaries |
| --- | --- | ---: | --- |
| Default | No | — | First fit improves; postcheck raises |
| Measure target norm, default caps/retries | Yes | 0.857397804901607 | All eight have partial updates |
| Accurate normalization/evaluation, default fit caps | Yes | 0.851177120343672 | Seven partial, one failed |
| Accurate normalization/evaluation **and fit caps** | Yes | 0.883113824503349 | All eight completed and accepted |

The last configuration takes 26.82 s with concurrent probes and uses:

```python
PepsOptimizer(
    state, gates, chi=2, mode="sweep", contraction_opt="greedy",
    normalize_initial=False,
    boundary_chi=(16, 64), normalize_chi=16, evaluation_chi=(16, 64),
    infidelity_kwargs={"norm_target": None},
)
```

This validates the remedy for the failing step, not a complete corrected
t=5 trajectory or general sufficiency of these caps at larger D. The ALS
algorithm, tolerances and acceptance guard were unchanged. JAX's early-time
gradient failure is a separate issue and is not addressed by this check.

New temporary evidence: `diagnose_sweep.py`, `check_failed_metric.py`, and
`diagnosis/{default,global,measured,metrics,all,metric-controls}.json` under
the same run directory. This follow-up changes only this record;
`git diff --check` passes. No implementation test suite was rerun for the
documentation-only change.

## Corrected full D=2 rerun

Starting afresh from |0⟩¹⁶ with the corrected settings above completes
**all 50 steps to t=5**, taking 1096.07 s inside the driver (18.27 min).
The ALS schedule, D=2, gates and dt are unchanged. This run does not resume
the old t=1.4 checkpoint, so its t=1.5 result differs from the one-step probe.

| Time | Exact circuit fidelity |
| --- | ---: |
| 0 | 1 |
| 1 | 0.987006297992 |
| 1.5 | 0.915209878846 |
| 2 | 0.823425425727 |
| 3 | 0.762532403641 |
| 4 | 0.723221498507 |
| 5 | 0.725660291295 |

Of 400 strip batches, 378 accept an optimized candidate, 21 skip optimization
because the precheck is below tolerance, and one at t=0.3 retains its warm
start after `no_valid_local_updates`. There are no later failed fits or
trajectory stops. Every saved state is finite with D≤2; the maximum exact
squared-norm error is 2.51e-9. Final mean Z is 0.933212049765 versus exact
0.737920428067, so numerical completion does not establish an accurate
long-time D=2 approximation for all observables.

Per-step exact vectors, full optimizer records, checkpoints, aggregate JSON,
and PNG/SVG figures are in `sweep-torch-accurate/` under the existing temporary
run directory. `run_validation.py --mode sweep --backend torch
--accurate-boundary --label sweep-torch-accurate` specifies this run; its
output directory must be new. No package implementation changed.

## D=3,4 background comparison launched

The user explicitly requested every configuration target **t=5**, with exact
fidelity recorded alongside approximate results. Eight sequential CPU runs
are queued under `/tmp/pepsy-tfim-d34-pCJAS1/`:

| D | Low norm/overlap caps | High norm/overlap caps | Solvers |
| --- | --- | --- | --- |
| 3 | `(18,36)` | `(36,144)` | ALS and Torch global L-BFGS-B |
| 4 | `(32,64)` | `(64,256)` | ALS and Torch global L-BFGS-B |

All start from |0⟩¹⁶ with the same circuit and use single-strip batches,
column gate order, SU warm starts and exact layered targets. Global L-BFGS-B
varies every candidate tensor; its target batch matches the ALS batching
policy. Subsequent targets differ when the trajectories diverge. ALS retains
its default budget; L-BFGS-B uses at most 100 iterations per batch and
`tol=1e-10`, with direct BRA/KET boundary compression. Compare measured
fidelity versus cost, not iteration counts or nominal convergence alone.

Each configuration uses its specified caps for fitting and metrics, measures
target norms, and disables cap retries/adaptive growth. Every acceptance
metric and optimized candidate is also audited with a normalized exact dense
overlap; every time step records exact circuit fidelity, mean Z, normalization,
driver/audit/measurement times, peak RSS, warnings and optimizer records.
The first global fit also records a directional finite-difference check.
A four-step D=2 runner check completed, but its global directional derivative
error was 3.06e-7, exceeding the chosen 2e-7 absolute allowance. That failed
gradient check is retained; global gradients are not certified by this launch.

The worker runs one configuration at a time with one CPU thread, a 16 GiB
RSS guard and no whole-run time limit. Failed configurations retain their
checkpoints/tracebacks and the worker proceeds to the next case. A `STOP`
file in the run root stops the worker and active process. The source is
isolated in a detached worktree at `31e9dab`; FullUpdate helpers and exact
reference vectors were copied. A runner filename collision with Python's
`queue` module was caught during setup, stopped and corrected before the
production launch; its artifacts remain in `bootstrap-failure/`.

`manifest.json` defines the grid; `queue-status.json` reports live progress;
`runs/<name>/observations.json` and `metrics.jsonl` hold the comparisons.
The foreground tool session owning the background worker is `8719`.
**D=3,4 results are pending**, not established by the completed D=2 run.

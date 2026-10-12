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

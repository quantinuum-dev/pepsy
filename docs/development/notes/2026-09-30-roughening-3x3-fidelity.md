# 3×3 roughening: fidelity, BP norms, boundary observables, and sampling

User-requested validation of the exponent-preserving SU implementation.
This is measured evidence for these settings, not a general convergence claim.
No numerical implementation or default was changed during this follow-up.

## Configuration and references

- The live downstream `PepsSimpleUpdate.step` calls Pepsy `gate_simple` with
  `renorm=False, strip_exponent=True`, `contract="reduce-split"`, and
  `cutoff_mode="rsum2"`. Initial gauges are created with `gauge_all_simple_`.
- Open 3×3 lattice, snake site order, diagonal x+y mean-field wall,
  Δθ=0 (physical paper angle −0.25268025514207865), J=−1, hx=1, hz=0.
- D=2 and D=4, dt=0.25, 24 steps through t=6, no periodic gauge refresh.
  SU cutoff 1e−12, boundary cutoff 1e−10, NumPy complex128 on CPU.
- Boundary χ=2D. PepsSampler future χ and conditioned-ket χ′ are also 2D;
  1,200 Z-basis configurations per D/time, seeds `20260930 + 100*D + step`.
- Exact full-Hamiltonian evolution uses independent 512×512 Hermitian
  diagonalization of H=JΣZZ+hxΣX+hzΣZ. A separate dense Strang gate sequence
  matches the runner timestep, allowing timestep and SU errors to be separated.
- The physical PEPS snapshot contains each external gauge exactly once and
  retains the core exponent. Full double-layer contraction uses a Cotengra
  HyperOptimizer, independently checked against the dense 512 amplitudes.
- Pepsy `two_norm_bp` uses tolerance 1e−10; `loop_cluster_expand` with its
  fixed-point messages supplies C=0, 4, and 6. C denotes maximum generalized
  loop cluster size. All four BP solves converged in 31–49 iterations.

Here N=⟨ψ|ψ⟩ is the **squared norm**, and fidelity is the normalized squared
overlap F=|⟨ψ_ref|ψ⟩|²/(N_ref N). A BP norm is not a fidelity estimate.

## Norms and fidelities

| D | t | N, full contraction | N, C=0 | N, C=4 | N, C=6 | F, exact H | F, exact Trotter |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4 | 0.107673964 | 0.101534115 | 0.107250120 | 0.108860644 | 0.01718807 | 0.02560132 |
| 2 | 6 | 0.071738408 | 0.068356172 | 0.070066220 | 0.071327871 | 0.01585143 | 0.01481801 |
| 4 | 4 | 0.170464751 | 0.155899160 | 0.173556792 | 0.162756358 | 0.04769693 | 0.04347017 |
| 4 | 6 | 0.088003923 | 0.080479729 | 0.085813110 | 0.081773583 | 0.02075271 | 0.01748189 |

Absolute relative norm errors, in the same row order:

| D | t | C=0 | C=4 | C=6 |
| --- | --- | --- | --- | --- |
| 2 | 4 | 5.7023% | 0.3936% | 1.1021% |
| 2 | 6 | 4.7147% | 2.3310% | 0.5723% |
| 4 | 4 | 8.5446% | 1.8139% | 4.5220% |
| 4 | 6 | 8.5498% | 2.4894% | 7.0796% |

C=4 improves these four BP estimates; increasing C to 6 is not monotonic.
Exact Trotter/full-H fidelity is 0.96183750 at t=4 and 0.93435433 at t=6.
Thus timestep error alone does not explain the much lower SU fidelities.
The exponent change fixes the previous overflow, but does not establish
accuracy of low-D SU with unre-equilibrated gauges.

## Observables

I=Σs_i⟨Z_i⟩/9 uses the initial wall signs. Mean Z is Σ⟨Z_i⟩/9.
The PEPS column is an exact contraction of the truncated PEPS, distinct from
exact Hamiltonian evolution. Sampling uncertainties are one standard error
from the self-normalized importance estimator, including site covariances
for the aggregate observables.

| D | t | Observable | Exact H | PEPS exact contraction | Boundary χ=2D | 1,200 weighted samples |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 4 | I | −0.015705 | 0.380830 | 0.381379 | 0.377037 ± 0.008830 |
| 2 | 6 | I | 0.005412 | 0.204426 | 0.204163 | 0.198519 ± 0.007773 |
| 4 | 4 | I | −0.015705 | 0.311392 | 0.329384 | 0.319398 ± 0.011985 |
| 4 | 6 | I | 0.005412 | 0.249167 | 0.253110 | 0.253943 ± 0.008230 |
| 2 | 4 | Mean Z | −0.051881 | −0.524285 | −0.523261 | −0.515556 ± 0.009575 |
| 2 | 6 | Mean Z | 0.069533 | −0.423531 | −0.423540 | −0.421852 ± 0.016252 |
| 4 | 4 | Mean Z | −0.051881 | −0.497848 | −0.520214 | −0.509469 ± 0.012115 |
| 4 | 6 | Mean Z | 0.069533 | −0.493665 | −0.502191 | −0.493178 ± 0.012968 |

Maximum absolute local-Z boundary errors relative to exact PEPS contraction
are 0.00610, 0.00331, 0.06196, and 0.02335 in row order. Maximum imaginary
residuals are 0.00411, 0.00651, 0.04790, and 0.01254. The runner appropriately
flags these as `check_boundary_chi`; χ=2D is not converged for these readouts.
Among the 36 sampled site estimates, the largest deviation from exact PEPS
is 3.08 estimated standard errors (D=2,t=4,site (2,0)); no assertion that
every local estimate lies within one or two standard errors is made.

## Sampling failure and explicit repair

The strict/default sampler failed at D=4,t=4,χ=χ′=8 with
`Conditional density matrix at site (2, 1) has a substantially negative diagonal.`
This is a finite-environment proposal issue; evolution and exact norm
contraction had already succeeded.

Only that diagnostic snapshot was retried with the existing explicit
`rho_positivity="absolute"` option. It takes absolute eigenvalues of the
Hermitian conditional matrix and includes the changed proposal in the saved
probabilities and weights. No live PEPS tensor, gauge, exponent, package
default, or runner default was changed. Its largest relative positivity
correction was 0.4827, so this was not roundoff clipping.

All four actual proposals were enumerated over all 512 configurations:
each sums to one within 4e−16, has full support, and agrees with returned
batch proposal probabilities within 6e−17. D=2 proposals equal exact PEPS
Born probabilities within floating-point error. D=4 proposal total variation
distances from PEPS Born probabilities are 0.07241 at t=4 and 0.02140 at t=6.
The strict D=4,t=6 proposal passed full enumeration without a repair.

Sample ESS values are 1200, 1200, 1125.8, and 1195.8. The theoretical ESS
fraction from all configurations is 0.8778 for the repaired snapshot, versus
the finite-sample estimate 0.9382; neither figure alone proves accuracy.
Every saved amplitude agrees with the exact PEPS amplitude within 9e−17.
Sampling leaves the evolution tensors, gauges, and exponent exactly unchanged.

## Reproduction and artifacts

Artifacts are under `/tmp/roughening_3x3_fidelity_bp_samples_20260930/`:
`check.py`, `audit_proposals.py`, `plot_results.py`, logs, `results.json`,
`summary.csv`, `local_z.csv`, four sample/state NPZs and four enumerated-proposal
NPZs, plus `comparison.pdf` and its two PNG pages. The original strict failure
is retained in `strict_run.log` and partial `strict_results.json`.

From the examples magnetization benchmark directory, activate
`~/envs/py312/bin/activate`, prepend the local Pepsy source and `$PWD` to
PYTHONPATH, set `CUDA_VISIBLE_DEVICES=''`, and run the three scripts in order.
The diagnostic used OMP/OPENBLAS/MKL thread counts of two. The primary
evolution/readout/sampling run completed in 21.6 seconds, excluding the
subsequent complete proposal audit and plotting.

No test files or implementation files were changed in this follow-up.
Checks inside the scripts establish first-step agreement with independent
Trotter gates, full/dense norm agreement within 6e−16, BP convergence,
proposal normalization/support, amplitude correctness and state isolation.
Earlier regression suites are recorded in the separate
[implementation note](2026-09-30-su-exponent-tracking.md); they were not rerun.

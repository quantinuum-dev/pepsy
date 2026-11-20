# L=16 Ising DMT versus MPS dynamics

**Measured:** strict DMT preserves its gated input's protected local densities,
but energy preservation alone does not establish accurate local dynamics.
At the tested small timestep and bond caps, appreciable local errors and nonpositive
three-site densities occur. Ordinary complex-basis MPO DMT can also lose
Hermiticity. These are distinct from the two MPS reference-path compatibility
bugs corrected during this experiment.

[Follow-up: Torch complex128, first-negativity traces, and paper-style
benchmarks](2026-11-19-dmt-complex128-positivity.md) qualifies the accuracy
comparison: larger-step protocols favor DMT at the tested caps. It also
records an unresolved published-curve normalization mismatch and an L=64
trace-guard failure; neither experiment is claimed as a curve reproduction.

## Protocol

- Baseline: `develop`, `f5d31006d02821028c890a04b17e4b15b9683d0b`;
  this session's changes are uncommitted. No dependency was changed.
- Open chain, L=16, double precision, unitary evolution to t=20,
  dt=0.05, samples every 0.5, bond caps 8/16/32.
- Common symmetric forward/backward nearest-neighbor bond sweep for every
  method. Onsite fields are apportioned over adjacent bonds, with full
  endpoint weights. State/output cutoff is explicitly zero; DMT constraints
  are strict, with the default paper connected-residual rule.
- MPS uses `MpsOptimizer` in both `svd` and `direct` modes, with unitary
  working-norm stabilization. MPO uses `MpoOptimizer(mode="dmt")`.
  Native Pauli DMT is tested with complex and real coefficient storage.
- Initial pure product state follows the period-eight near-Y construction
  in White et al., Eq. (32). Each site is proportional to
  `i(1+g_j)|0> + |1>`, where one-based `j mod 8` in `{1,2,7,0}` has
  `g_j=-0.1`, and the other sites have `g_j=+0.1`.

Our explicit Pauli-matrix Hamiltonian convention is

$$
{\color{blue}H}=\sum_{j=0}^{14}Z_jZ_{j+1}
+h_x\sum_{j=0}^{15}X_j+h_z\sum_{j=0}^{15}Z_j.
$$

| Case | hx | hz | Role |
| --- | ---: | ---: | --- |
| TFIM | 1.05 | 0 | Integrable control |
| Tilted Ising | 0.9045 | 0.8090 | Paper coupling ratios; integrability broken |

[White et al.](https://arxiv.org/pdf/1707.01506), Secs. III–IV, motivate
protecting local densities and compare pure-state near-Y evolution. This
experiment uses their coupling ratios and initial-state pattern, not their
L=64/128 sizes. In the spin convention S=Pauli/2, their Eq. (30) is H/4,
so paper time is four times the time reported here. The finite size,
Trotter step, and truncation/gauge choices preclude claiming reproduction
of their published curves.

## Independent references and measurements

1. **Exact Hamiltonian:** sparse bit-basis construction, evolved with SciPy
   `expm_multiply`. Exact energy drift is below 7e-14 in total energy; norm
   drift is below 1.4e-14. No dense density matrix is formed.
2. **Exact gate stream:** independent full state vector with precisely the
   same gates, isolating compression error from Trotter error.
3. **Compression audits:** contract physical ket/bra indices directly,
   independently of the DMT protected panels. For each checked gate, obtain
   the input marginal on the union of a triple and the gate support, apply
   the small dense gate, trace extras, and compare with the compressed
   output. This needs at most a four-site density matrix.

Readout includes all-site X/Y/Z, every nearest-neighbor ZZ, every contiguous
three-site density, total/bond energy, and a normalized cosine energy-density
mode at k=pi/4. Imaginary observable components and Hermiticity defects are
retained. Eigenvalue diagnostics use the Hermitian part of each three-site
density; they do not certify positivity of the full L=16 operator.

For all six real-Pauli runs, **all 69,600 gates** were independently audited.
The maximum three-site Frobenius preservation defect was **1.21e-13**,
the largest total-energy change due to compression was **1.10e-13**, and
the largest per-gate trace defect was **8.07e-14**. Trace drift over the
sampled trajectories was at most **4.12e-12**. Actual lossy splits occur;
this is not an untruncated or bond-one pass-through test. Ordinary MPO and
complex-Pauli runs audit twelve gates each, spread across the trajectory,
in addition to their native per-split guards.

The tested invariant is

$$
\operatorname{Tr}_{\bar R}{\color{purple}\rho_{\mathrm{DMT}}}
=\operatorname{Tr}_{\bar R}{\color{blue}U\rho_{\mathrm{current}}U^\dagger},
\qquad R=\{j,j+1,j+2\}.
$$

It does not equate the current compressed input with the exact evolved state.

## Results

The [machine-readable summary](2026-11-19-dmt-ising-summary.json) retains
all methods, caps, diagnostics, and timings. Its `xyz_rmse_vs_exact` averages
over all sites, three Pauli axes, and the 41 sample times. The three-site
metric is RMS Frobenius error, not trace distance.

| Model | chi | MPS direct XYZ RMSE | MPO DMT XYZ RMSE | Real Pauli DMT XYZ RMSE |
| --- | ---: | ---: | ---: | ---: |
| TFIM | 8 | 0.10194 | 0.15689 | 0.15680 |
| TFIM | 16 | 0.05870 | 0.12185 | 0.11983 |
| TFIM | 32 | 0.02551 | 0.08295 | 0.08134 |
| Tilted | 8 | 0.07071 | 0.12374 | 0.12544 |
| Tilted | 16 | 0.03351 | 0.06173 | 0.06297 |
| Tilted | 32 | 0.01510 | 0.04320 | 0.04322 |

MPS direct also has smaller Pauli RMSE when restricting to t>=10 in every
tested model/cap comparison. This finite-size experiment does not demonstrate
a DMT advantage in these local observables. The
[CSV table](2026-11-19-dmt-ising-results.csv) includes energy, RDM, late-time,
Hermiticity and timing columns. All 32 baseline comparison rows completed;
maximum sampled trace drift over all DMT storage choices was 4.78e-12.

At chi=16 in the tilted case, real-Pauli DMT has maximum energy drift per
site 2.5904e-4 versus MPS direct's 2.7346e-2. However, its all-site Pauli
RMSE is 0.06297 versus 0.03351, and three-site density RMSE is 0.17735 versus
0.07933. Energy conservation improves substantially while local accuracy
is worse in this comparison. Real-Pauli chi=32 improves Pauli RMSE to
0.04322 and the most negative sampled local eigenvalue to -3.09e-4;
this remains an approximation, not a positivity guarantee.

The integrable TFIM is a useful stress control, not evidence for or against
the paper's thermalizing-system claim. For real-Pauli DMT its Pauli RMSE
decreases from 0.15680 to 0.11983 to 0.08134 for chi=8/16/32; the energy
mode error is not monotone in chi. MPS direct gives 0.10194/0.05870/0.02551.

With real Pauli coefficients, Hermiticity defects and imaginary observables
are exactly zero in these runs. Positivity still fails: the tilted chi=16
run reaches a three-site eigenvalue -0.05729. Ordinary complex MPO storage
can produce large Hermiticity defects, despite passing local preservation.
Thus reporting only real parts or only energy would conceal important failures.

[TFIM chi=16 plot](2026-11-19-dmt-ising-tfim-chi16.png) ·
[Tilted chi=16 plot](2026-11-19-dmt-ising-tilted-chi16.png)

### Timestep check and optional refinement

For the tilted chain, halving dt to 0.025 reduces exact-gate-stream maximum
energy drift per site from 2.5949e-4 to 6.4876e-5, a factor of four. Real
Pauli DMT at chi=16 changes from 2.5904e-4 to 6.4641e-5. This is consistent
with second-order Trotter error dominating energy drift.

At fixed chi, smaller dt also means more truncations: DMT's Pauli RMSE
**increases** from 0.06297 to 0.10093 in this check. This is not a converged
joint dt/chi limit. Preservation through individual truncations does not
control accumulation of finite-bond dynamics error.

Supplemental chi=16 runs with `truncation="optimized"` and zero alternating
refinement sweeps fail on both models with
`FloatingPointError: DMT refinement cannot canonicalize its feasible factors
at this precision.` They are **failed experiments**, not accuracy results;
no complete optimized trajectory is included in the comparisons. The
failure is in `_dmt_refine.refine_bond`'s final canonicalization guard.
The optional refiner was not changed in this task.

## MPS fixes and validation

- **Compatibility shim:** NumPy/Python norm scalars have no gradient tape.
  Avoid an unavailable Autoray `stop_gradient` dispatch in initial unitary
  norm tracking and nonunitary overlap diagnostics.
- **Adopt exact local construction:** NumPy `direct` with cutoff=0 now uses
  the existing fixed operator-bond two-site construction. Quimb 1.15.0's
  `gate_nonlocal` otherwise splits the gate with a separate default cutoff,
  creating a roughly 2e-6 small-system error even with sufficient state chi.
  Nonzero-cutoff behavior and Torch/JAX paths are unchanged.
- Final focused selection: **263 passed, one skipped** across ordinary-MPO
  DMT replay, ordinary MPS, MPS normalization, gate precision, and the new
  benchmark regressions. After the last energy-readout assertion, six NumPy/
  Hamiltonian regressions passed again. Ruff and `git diff --check` pass.
- Earlier broader selection: 321 passed, one skipped, two failures in existing
  MPS rank-expectation and nonunitary-timing assertions. Both failures were
  reproduced with baseline compression/norm source loaded and only a test-side
  NumPy stop-gradient compatibility registration. They were not weakened.
  Standalone mixed-backend tests also exposed the checkout's existing
  namespace-cache ordering sensitivity; the final combined selection above
  exercises Torch/JAX successfully. No full-suite or hardware-GPU claim.

Installed versions: NumPy 2.5.2, SciPy 1.18.0, Quimb 1.15.0, Autoray 0.8.4,
Cotengra 0.8.3.dev8+g8954240f2, Cotengrust 0.2.1, and Symmray
0.4.1.dev15+g0374aaa3c. Reviewed official Quimb/Autoray/Cotengra documentation
and Symmray source; the linked Symmray array page was unavailable. Native
Symmray DMT and new GPU validation remain deferred. Computation used CPU
NumPy with one BLAS thread per process. Timing samples are single runs,
sometimes concurrent with independent checks; they are not controlled
throughput measurements.

## Reproduction

Activate the environment selected by `AGENTS.md`, then run from the repository
root. The common reference and cutoff settings are explicit in the script.

```bash
python examples/dmt_ising_dynamics.py --time 20 --dt .05 --sample-every 10 \
  --models tfim tilted --chis 8 16 32 \
  --methods mps-svd mps-direct mpo-dmt pauli-dmt \
  --output /tmp/dmt-ising-l16-main
python examples/dmt_ising_dynamics.py --time 20 --dt .05 --sample-every 10 \
  --models tfim tilted --chis 8 16 32 --methods pauli-dmt-real --audit-all \
  --output /tmp/dmt-ising-l16-real
python examples/dmt_ising_dynamics.py --time 20 --dt .025 --sample-every 20 \
  --models tilted --chis 16 --methods pauli-dmt-real \
  --output /tmp/dmt-ising-l16-halfdt
```

Per-sample NPZ arrays and logs remain under those temporary paths. Durable
summaries and selected figures are linked above. Equal bond caps are the
requested comparison; MPS and MPO bonds represent different objects and
do not imply equal cost or equal approximation power.

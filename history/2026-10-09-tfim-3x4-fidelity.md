# 2026-10-09 — 3x4 TFIM full-update fidelity benchmark

- Scope: user requested second-order TFIM evolution versus `MpsOptimizer`
  exact mode, D=4, boundary chi=2 D^2=32, dt=0.1, J=h=1. User selected
  t=5 (50 steps) after the proposed open boundaries and initial |+> state.
- Branch / baseline: `develop`, `8d53933b95341690693187047d03a58f617dfa22`.
- Commit status: new evidence files only, uncommitted; no algorithm edits,
  staging, commit or push. Preserved unrelated workspace changes.

## Method

Executed an isolated `git archive HEAD` snapshot, verifying its imported
Pepsy path. Torch CPU complex128, one BLAS/Torch thread, seed 20261009.
Hamiltonian H=-sum_nn Zi Zj-sum_i Xi, Pauli operators. Each step applies
X(dt/2), all 17 ZZ(dt) bonds (horizontal rows, then vertical columns),
X(dt/2). Gate order is `input`. Both engines use precisely this stream.
`MpsOptimizer(mode='exact')` is dense exact gate replay, without truncation.

PEPS uses committed defaults: reduced ALS, gauge off, maximum 50 local
iterations, automatic tolerance, cached one-site DMRG boundaries initialized
by SRC, one chi=32, no convergence probes, no independent acceptance checks
and no refinement. Default initial/per-gate output normalization is retained.
Contraction planning uses `build_optimizer(max_repeats=8, parallel=False)`.
Repeated `run(reset_traces=False)` preserves accumulated local fidelity.

External benchmark hooks contract the PEPS exactly before and after every
pair update, measuring its normalized squared overlap with the dense gated
pre-update state. These observations do not affect updates or acceptance.
At every complete timestep, measure exact PEPS overlap against the exact
MPS engine. Independently verify that engine against NumPy gate application.
A sparse Hamiltonian exponential also measures continuous-time fidelity,
separately from the primary same-Trotter comparison.

## Measured result

| t | True same-gate fidelity | Reported local product | Exact local product |
| --- | ---: | ---: | ---: |
| 1 | 0.998744 | 0.999581 | 0.999581 |
| 2 | 0.549995 | 0.872550 | 0.872070 |
| 3 | 0.289242 | 0.596685 | 0.592625 |
| 4 | 0.119774 | 0.417927 | 0.414437 |
| 5 | 0.094333 | 0.300149 | 0.296961 |

The local product substantially overestimates final fidelity in this run.
Even a product of exact local fidelities does so, isolating the accumulation
assumption from boundary-estimation error. This is consistent with correlated
truncation errors; a local product is neither an exact whole-circuit fidelity
nor a certified bound. This run does not establish chi convergence or optimal
ALS solutions. D=4 with this default update does not maintain high global
fidelity through t=5 for this initial state.

- All 850 pair updates completed: 34 exact-SVD shortcuts (first two steps),
  816 ALS updates. No ALS iteration-limit exits; maximum 20 iterations.
- Maximum absolute local-fidelity estimate error: 3.32392e-4; mean 1.54269e-5.
  Largest discrepancy between the two accumulated products: 0.00413119.
- Maximum exact MPS/NumPy statevector discrepancy: 8.97e-16. PEPS bonds
  stayed <=4; maximum measured norm-squared deviation from one: 9.77e-15.
- At t=5, exact Trotter versus continuous-time fidelity is 0.996211;
  PEPS versus continuous time is 0.097555. Trotter error is distinct from
  the large PEPS error in the primary comparison.
- PEPS wall time excluding dense hooks: 155.03 s; exact engine replay:
  0.268 s; dense local audits: 1.72 s. Full loop: 157.81 s. These are
  instrumented CPU run timings, not general scaling claims.
- Boundary cache counters at the final gate show reuse; exact norm and
  fidelity audits remain independent of those cached environments.

## Evidence and validation

- [Configuration, summary and all 50 steps](../docs/development/notes/2026-10-09-tfim-3x4-fidelity.json)
- [Figure PNG](../docs/development/notes/2026-10-09-tfim-3x4-fidelity.png)
- [Figure PDF](../docs/development/notes/2026-10-09-tfim-3x4-fidelity.pdf)
- Temporary reproducibility bundle: `/tmp/pepsy-tfim-3x4-t5/benchmark.zip`.
  Contains scripts, all 850 local audits, CSV trajectory, final statevectors,
  figures and logs. Temporary files can disappear; numerical conclusions
  and trajectory above are retained in the repository evidence files.
- Benchmark assertions checked gate counts, D/chi caps, absence of outer
  metrics and independent exact-reference agreement at all 50 steps.
  Script Ruff checks and `git diff --check` passed. No implementation changed,
  so no new unit/full-suite run was needed. Nothing was committed or published.

## Follow-up — skeptical correctness audit

The user questioned whether the large gap could be a fidelity or engine bug.
Ran additional independent numerical checks against the same isolated commit;
production defaults and implementation remain unchanged. Detailed evidence:
[audit JSON](../docs/development/notes/2026-10-09-tfim-3x4-fidelity-audit.json).
Scripts and full local records are in the temporary
`/tmp/pepsy-tfim-3x4-t5/fidelity-audit.zip`.

Independent reference and ordering checks:

- Rebuilt the Trotter reference using sparse sum-X exponentials and a ZZ
  diagonal phase, without the gate replay implementation. Its state differs
  from the exact MPS reference by at most 1.36e-14 per component.
- Direct overlap, Torch overlap, phase-aligned vector distance, and inverse
  sparse evolution back to the initial state all reproduce the original
  final fidelity 0.094332960318366 to roundoff.
- Direct products of saved local fidelities match accumulated records at
  every timestep, maximum discrepancy 5.55e-16.
- A nonuniform complex product state with distinct general complex gates at
  every site/edge and reversed pair ordering on alternate edges agrees with
  independently bit-embedded sparse matrices: maximum statevector errors
  3.29e-16 (PEPS) and 5.87e-17 (exact MPS). All 17 pair gates fit D=4 without
  truncation. This avoids symmetry masking ordering/conjugation mistakes.

Full 50-step diagnostic evolutions:

| Environment | True final fidelity | Local product |
| --- | ---: | ---: |
| Original cached chi=32 | 0.09433296 | 0.30014860 |
| Fresh chi=32 at every pair; both caches disabled | 0.10444125 | 0.29984916 |
| Exact reduced Gram environment; all boundary calls bypassed | 0.07092354 | 0.27942436 |

The exact metric is E.H @ E from the independently contracted single-layer
exterior map E, retaining its two QR legs and all other physical legs. This
does not use the production double-layer/strip/boundary metric builder.
Normalization in this diagnostic uses exact dense norms and changes only the
TN exponent. Reported and exact local fidelities agree within 1.60e-14 over
all 850 updates, and every reconstructed target agrees with the independently
gated pre-update state within 6.66e-16 in infidelity.

At eight interior horizontal/vertical updates at t=1,2,3,5, an independent
NumPy/SciPy weighted rectangular least-squares ALS implementation with a
stricter 1e-13 stopping threshold agrees with the production exact-environment
solve within 2.71e-10 in fidelity. No exact-environment or fresh-environment
update worsens the true warm-start fidelity beyond roundoff. These checks
verify sampled local solves, not global optimality of the D=4 trajectory.

Tracked the missing terms explicitly in a second exact-environment run:
for normalized pre-update target y and retained state x, define
alpha=y.H x and residual e=x-alpha*y. With exact circuit state psi,
c_new=alpha*c_old+psi.H e. The local product retains only the product of
|alpha|^2. Propagating all residual contributions reproduces the actual
overlap with maximum error 2.26e-14. At t=5 its squared modulus decomposes as
0.27942436 + 0.39963516 - 0.60813598 = 0.07092354: a substantial destructive
cross term. The missing coherent contributions are measured, not merely a
proposed explanation.

Conclusion: no fidelity-formula, gate-reference, ordering, or accumulation-
bookkeeping bug was found in these checks. The large product/global gap
survives exact environments and cache removal. Fresh and cached approximate
fits do produce different trajectories (maximum absolute global-fidelity
difference 0.01120), so this does not establish exact cache/fresh equivalence
at finite chi or certify all boundary code. The exact-environment control
shows that such differences are not required for the large discrepancy.
No claim of globally optimal reduced ALS or chi convergence is made.

All diagnostic runs completed, script Ruff and `git diff --check` passed.
Only evidence/journal files were added or updated; nothing was staged,
committed or pushed. No fresh full-suite run was needed for these read-only
implementation audits.

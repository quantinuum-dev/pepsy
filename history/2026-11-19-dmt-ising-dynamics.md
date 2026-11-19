# 2026-11-19 — L=16 Ising DMT/MPS comparison

- Scope: user requested ordinary MPO and Pauli DMT time evolution at L=16,
  equal-chi MPS comparison, energy/local-density checks, and comparison with
  White et al., arXiv:1707.01506.
- Branch / baseline: `develop`, `f5d31006d02821028c890a04b17e4b15b9683d0b`.
- Status: uncommitted working-tree edits; nothing staged, committed or published.

## Work and findings

- Added a reproducible transverse/tilted Ising benchmark with a period-eight
  near-Y product state, exact Hamiltonian and identical-gate state-vector
  references, physical-index local-density audits and exported plots/NPZs.
- All 69,600 gates across native real-Pauli L=16 chi=8/16/32 runs preserve
  three-site targets to 1.21e-13. This verifies the local contract, not exact
  future dynamics. Energy is much steadier than MPS, but local accuracy need
  not improve and positivity can fail. Complex ordinary MPO DMT can also
  lose Hermiticity. No production DMT rule was changed.
- Corrected NumPy norm-diagnostic stop-gradient dispatch and zero-cutoff
  direct gate factorization in the MPS reference paths, with regressions.
- Optional optimized DMT fails its canonicalization guard in supplemental
  long runs. Recorded as unresolved; did not silently fall back or weaken it.
- [Protocol, measured results, sources and reproduction](../docs/development/notes/2026-11-19-dmt-ising-dynamics.md).

## Validation

- Focused MPO replay/MPS/normalization/precision/new benchmark selection:
  263 passed, one skipped. Six final NumPy/Hamiltonian regressions passed.
- Earlier broader selection: 321 passed, one skipped, two baseline-reproduced
  MPS rank/timing assertion failures. See the report for the baseline probe
  and standalone mixed-backend ordering limits. No full-suite claim.
- Ruff and whitespace checks pass; no dependency edits or new hardware-GPU
  execution. Full-region checks remain enabled and actual lossy splits occur.
- Raw arrays/logs in `/tmp/dmt-ising-l16-*`; durable numerical summaries and
  selected plots accompany the report.

## Follow-up — Torch complex128 and positivity mechanism

- Completed L=16 chi=16 t=20 Torch CPU complex128 runs for MPS direct,
  ordinary MPO DMT and native Pauli DMT on both models. Gates and final
  cores are dtype-checked. GPU hardware is unavailable. No production DMT
  rule or tolerance was changed.
- Added independent first-negativity tracing. At L=6 chi=16 a truncation
  creates a global eigenvalue -6.63e-8 while protected triples remain
  positive to roundoff. Later a four-site negative input generates an
  already-negative gated triple target, which DMT preserves to 4.37e-16.
  L=16 Torch complex128 independently exhibits the latter mechanism.
- Added local transfer-environment MPS measurements, independently checked
  against dense partial traces with nontrivial gauge and represented scale.
  Registered the existing public Torch stop-gradient helper explicitly in
  the benchmark; standalone startup otherwise depended on earlier imports.
- Completed paper-style L=16 chi=16/32 larger-step protocols with the exact
  energy Fourier observable and untruncated identical-gate references.
  Both explicit spin/Pauli conventions favor native Pauli DMT over MPS in
  the measured aggregate mode error. The small-step ordering was opposite.
  Published-curve reproduction is not claimed: the printed-equation
  initial energy differs from Fig. 7, and no author implementation/raw data
  was found in the arXiv source.
- L=64 chi=16 real-Pauli DMT fails at step 3 gate 60 with relative trace
  defect 1.21e-6 > 4.77e-7. Separately reproduced, saved with context, and
  left unresolved. The incomplete MPS-only result is not a comparison.
- New validation: focused selection **269 passed, one skipped**; after
  final example changes **16 passed**; Ruff and whitespace checks pass.
  Prior baseline failures and no-full-suite limits above still apply.
- [Detailed follow-up, 25 summary rows, configurations, diagnostics and plots](../docs/development/notes/2026-11-19-dmt-complex128-positivity.md).
  Branch/baseline unchanged; nothing staged, committed or published.

# 2026-11-19 — Focused White et al. Fig. 7/8 comparison

- User scope: focus on the paper's Figs. 7 and 8 using Pepsy's methods.
- Branch/baseline: `develop`, f5d31006d02821028c890a04b17e4b15b9683d0b.
  Earlier MPS reference fixes and benchmark artifacts remain in the working
  tree. Nothing staged, committed or published.
- Recovered balanced positive-Y state, time/energy mapping and Fourier phase
  from vector PDF data. The final exported sample is an endpoint state at
  x=99/39; retain raw fixture coordinates and align explicitly for comparison.
- Independent dense Fig. 8 L=16 gate reference matches all 100 digitized
  semiexact samples to 3.26e-9. Early eight-point reference and published
  initial spin/energy values have deterministic regressions.
- Completed Torch complex128 MPS svd/direct, ordinary MPO DMT and Pauli DMT
  at chi=16/32/64 through figure time 100. Native Pauli mode RMSE is
  7.27e-4/4.73e-4/5.18e-4, 31–48x smaller than MPS; ordinary MPO still
  loses Hermiticity. Native Pauli chi=32/64 triples have roundoff negativity.
- Fig. 7 L=64 MPS completes and follows the published baseline closely.
  Both DMT paths stop at preservation guards; recorded as failures. NumPy
  ordinary MPO cross-check also fails. No production DMT kernel or guard
  changed. Isolated trace-aware prototypes failed physical audits and were
  not adopted. Long-chain numerical conditioning remains unresolved.
- Added reproducible PDF extraction fixture and plotting/aggregation example;
  updated the energy example's default to the data-supported figure protocol.
  Literal printed-equation initialization remains separately selectable.
- New validation: 18 example regressions pass; Ruff, whitespace and local
  report-link checks pass. No new full-suite claim; previous broader-suite
  baseline failures are recorded in the preceding handoff.
- [Full protocol, evidence, artifacts and commands](../docs/development/notes/2026-11-19-dmt-white-figures.md).

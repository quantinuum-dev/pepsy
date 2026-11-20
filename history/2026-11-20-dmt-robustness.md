# 2026-11-20 — Ordered DMT robustness work and positive-factor control

- Scope: user authorized the six robustness/efficiency/coverage/convergence/
  positivity work items in order.
- Branch / baseline: `develop`, `eaedf789`; preceding uncommitted region-review
  changes preserved.
- Publication: uncommitted working-tree changes; no new commit/push requested.

Isolated the chi192 failure to real native QR heap corruption: a QR call
changed an unrelated live factor, and synthetic Quimb/NumPy QR reproduces a
late native crash. This matches NumPy issue 33161. Added a platform/BLAS/dtype/
size-gated pivoted-QR shim for managed DMT and native Pauli paths; retained
native behavior elsewhere and exposed the exact gauge change by warning.

Implemented exact-fit map elision, immutable bounded rank-aware candidate
plans, and public `complete_dmt_regions` support coverage. Added matched
MPS/DMT/FIT chi/dt studies and a separate locally purified factor prototype.
The prototype guarantees PSD algebraically but explicitly changes marginals;
simultaneous bounded-rank positivity and hard DMT constraints remains open.

New 6×6 chi96/128/192 processes finish t=0.3 with no preservation errors;
chi192 final trace defect is 7.12e-11 and canonical Gram defect 6.21e-14.
Inter-chi plaquette gaps remain about 0.01, so this is stability evidence,
not converged large-lattice dynamics. On 3×3, the matched chi256 calculation
agrees with the untruncated circuit; finite-chi negativity remains visible.

Validation: 558 DMT passes/one skip, 128 native-Pauli passes/one skip;
the final new-feature selection has 42 passes. Full-suite first failure
reproduces on archived clean HEAD (BP cluster / Autoray namespace, after
104 passes and seven skips). Public API/layout has the pre-existing installed
version mismatch. See the [evidence record](../docs/development/notes/2026-11-20-dmt-robustness.md)
for methods, scope, measurements and remaining work.

User steering after the first pass: prioritize Torch/CuPy, retain MPO/Pauli-MPO
DMT, and do not pursue XX† or constrained positive-factor fitting. Updated the
active plan accordingly; prior positive-factor results remain historical.
The local environment has Torch CPU but no CUDA and no installed CuPy, so the
NumPy large-lattice results do not establish GPU-backend convergence.

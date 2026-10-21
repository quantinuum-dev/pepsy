# 2026-10-21 — JAX QR and periodic direct boundary fixes

- Scope: fix the dependency causes found by the requested Gaugy GPU review.
- Branch / baseline: `develop` / `2aacd87`.
- Commit status: this record accompanies the local implementation commit;
  final full-suite validation is in progress. Nothing is pushed.
- Existing solver changes and their documentation/tests belong to another
  task and are preserved separately.

## Changes and evidence

Adaptive JAX QR now reserves its regularized pullback for exactly zero primal
pivots, preserving native derivatives at small nonzero pivots and linearity
for implicit FIT. Periodic direct boundaries pass the actual public Quimb
direct compressor through callable dispatch. See the
[numerical rationale and upstream audit](../docs/development/notes/2026-10-21-qr-periodic-boundary-fixes.md).

Focused final validation passed: 50 QR/FIT tests (CPU/GPU and zero pivots),
25 periodic-boundary/compatibility tests, four existing flat-backend tests,
and the five original downstream Gaugy MPS/boundary failures. Gaugy's added
six device/style cases check every parameter at two states against both Torch
and finite differences. Full cross-subsystem validation is running; no
full-suite success is claimed yet. Logs: `/tmp/gaugy-numerical-fixes-ZSzg6D`.

The full run found an additional failure in
`test_netket_flat_z2.py::test_fermionic_boundary_input_scaling_preserves_native_signs_and_log_values[False]`:
NumPy `take` dispatch reaches a JAX array with unsupported `mode='raise'`.
The same file fails identically against untouched Pepsy `2aacd87` (14 passed,
one failed). This predates the QR and periodic-boundary changes and remains
visible in the validation report.

No push or release is authorized or performed. Singular-chart and rank-crossing
limitations remain explicit; original downstream tolerances are unchanged.

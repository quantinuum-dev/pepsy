# 2026-10-21 — Numerical review and backend fixes

- Scope: fix the dependency causes found by the requested Gaugy GPU review.
- Branch / baseline: `develop` / `2aacd87`.
- Commit status: implementation commits `c2e645a`, `1149b2b`, and `8069fdd`
  are local. This record finalizes their validation; nothing is pushed.
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
and finite differences. Full cross-subsystem results are recorded below.

The full run found an additional failure in
`test_netket_flat_z2.py::test_fermionic_boundary_input_scaling_preserves_native_signs_and_log_values[False]`:
NumPy `take` dispatch reaches a JAX array with unsupported `mode='raise'`.
The same file fails identically against untouched Pepsy `2aacd87` (14 passed,
one failed). This predates the QR and periodic-boundary changes and remains
visible in the initial validation report. The follow-up fixes normalization
with native Autoray norm/log dispatch, avoiding a JAX scalar that converted
NumPy blocks while leaving their sector metadata unchanged. The unchanged
15-case flat-Z2 suite and the corresponding bosonic small-norm TDVP regression
pass. Final VMC reruns passed 30 boundary-compression and 135 TDVP tests.

The full boundary-input suite also caught ten compatibility failures in the
new cyclic probe: existing contraction wrappers need not expose cyclic-axis
methods. The probe now checks for those methods before calling them and
preserves the original dispatch otherwise. Existing tests are unchanged;
the complete boundary-input and periodic numerical suites pass all 190 cases.

The tail of the full run exposed a Quimb shared-RNG race in
`test_stim_importance_sampling_and_parallel_seed_streams`. Ordinary
`ps_to_mps` had created a seeded random MPS before overwriting it with known
product vectors. It now calls the public product-state constructor with
independent site arrays. All 160 constructor/trajectory tests (including 16
new deterministic cases) and all 365 stabilizer-state consumer tests pass.
The audit note above records the adopted upstream API and preserved contracts.

## Final validation

The complete collection contains **9,246 cases in 232 files**. The final
aggregate is **9,243 passed, zero failed, three skipped**, with no missing or
unexpected collected node IDs. Files ran in fresh processes with all test
profiles enabled (`python -m pytest -o addopts=''`) and the RTX A5000 visible.
This is a full collection plus focused revalidation, not a single monolithic
pytest run: files started before the final QR correction were rerun, as were
the complete affected VMC, boundary, constructor, trajectory, and stabilizer
suites after their follow-up changes. Initial failures and rerun logs remain
available separately.

Supplemental configurations exercised previously skipped JAX cases with
64-bit mode and two logical CPU devices. The three remaining skips require
Metal, two CUDA devices (this machine has one), or CuPy float32 subnormals
that this hardware flushes to zero. CPU multi-device checks are not presented
as multi-GPU coverage. Explicit Torch CPU/CUDA and JAX CPU/GPU numerical
regressions supplement the GPU-visible full run.

Whole-tree Ruff, the 12-skill catalog check, and `git diff --check` pass.
The pre-existing solver working-tree edits were present during validation
but are excluded from these commits. Gaugy's separate full run retains two
compressed-trace failures; these results do not certify that downstream path.

Raw evidence is in `/tmp/gaugy-numerical-fixes-ZSzg6D`, with a retained copy
under sibling Gaugy's ignored `runs/numerical-fixes-2026-10-21-ZSzg6D/`.
`final-aggregate.json` records the final source XML for each failing case and
the collection-coverage audit; original per-file logs and focused XML reports
are retained alongside it.

No push or release is authorized or performed. Singular-chart and rank-crossing
limitations remain explicit; original downstream tolerances are unchanged.

## Subsequent publication

After this report, the user explicitly requested commit and push. All four
local commits through `472715d` were pushed successfully to `origin/develop`.
The separate solver working-tree changes remain uncommitted and untouched.
The subsequent compressed-trace repair is confined to sibling Gaugy; this
publication note adds no Pepsy numerical changes or new validation claims.

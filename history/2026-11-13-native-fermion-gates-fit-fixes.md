# Etienne gate and native one-site DMRG repairs

## Scope and implementation

Follow-up to the [initial audit](2026-11-13-native-fermion-dmrg-hubbard-audit.md),
now explicitly authorized to fix the gate, one-site FIT and Autoray issues.
Baseline: `develop`, HEAD `1d0a894`. Changes are uncommitted; nothing was
staged, committed or pushed. Existing PEPS/full-update changes were preserved.

- `tensors/symmetric.py`: fill an owned native generator's missing zero
  blocks before fusion/exponentiation. Filling only the fused matrix misses
  physical component sectors already removed by fusion.
- `fitting/local.py`: correct every dual environment leg independently.
  Reversed gates invalidate the old assumption that exactly one virtual leg
  is dual. The minimal reproducer uses hopping on `(0,3)`, `(1,2)`, then
  `(1,0)` on four half-filled spinful sites.
- `optimizers/mps/optimizer.py`: initialize native canonical form without
  Quimb's dense `eye(d)` scan, which cannot infer charge-index metadata.
- Added zero-angle, imaginary-time, sparse-observable ownership and reversed
  native-gate regressions. Updated sparse MPO/PEPO reference fusion to keep
  the full physical charge space, preserving their assertions and tolerances.
- Updated the symmetric tensor/MPS API guides, Symmray audit, changelog and
  [detailed repair record](../docs/development/notes/2026-11-13-native-fermion-gates-fit.md).

Hubbard Hamiltonian signs, onsite/chemical-potential conventions, source
angles, charge sectors, native array types and compression mode meanings are
unchanged. No direct fallback or block warm-up was added to one-site DMRG.

## Upgraded environment

The shared environment changed during the task. Final checks confirmed Git
Autoray `0.11.1.dev17+g7a474cab3`, Quimb `1.16.1.dev1+g3371ff40c`, Symmray
`0.5.1.dev2+g20035beb7`, and Cotengra `0.8.4.dev8+g646239731`. Exact VCS
provenance is saved in `/tmp/pepsy_fermion_fix_environment.json`.

The installed GitHub Autoray fixes NumPy `stop_gradient`; 36 previously
blocked native replay checks now pass. No local dispatch patch is needed.
The existing requirement `autoray>=0.9` already excludes the previous 0.8.4
environment. No dependency installation or installed-library edit was
performed by these source changes.

## Validation

All numerical checks used the selected Python 3.12 environment and local
source. CPU domain runs used one BLAS/OMP thread, `CUDA_VISIBLE_DEVICES=''`
and `JAX_PLATFORMS=cpu`. No CUDA result is claimed.

| Check | Result |
| --- | --- |
| Exponential, zero-angle and spin-gate checks | 19 passed |
| Native parity/writeback/cache and new reversed-gate cases | 42 passed |
| Native PEPO unitary reference sandwich | 8 passed |
| Non-slow tensor/native-MPS/MPS/MPO selection | 662 passing cases, 1 skip, 30 slow cases deselected; see rerun detail below |
| Default smoke suite | 112 passed |
| `python -m ruff check src tests` | Passed |
| `git diff --check`, local links in touched API/repair docs | Passed |

The non-slow command was:

```bash
python -m pytest -q -o addopts='' -m 'not slow' --tb=short \
  tests/test_symmetric_tensors.py tests/test_mps_fermions.py \
  tests/test_optimize_mps.py tests/test_optimize_mpo.py
```

That run recorded 661 passes and one final sparse-reference shape mismatch
in `test_fermion_to_mpo_accepts_arbitrary_neutral_term_support`. Completing
the oracle's zero components before fusion fixed it; its focused rerun
passed. The other cases' source/controls were unchanged. Logs:
`/tmp/pepsy_fermion_fix_nonslow_domain_tests.log`,
`/tmp/pepsy_native_fermion_fit_fixed_tests.log`,
`/tmp/pepsy_fermion_gate_fixed_tests.log`, and
`/tmp/pepsy_fermion_fix_smoke_tests.log`.

The generated Etienne preparation plus real-time Strang pulse was checked
through the actual sibling helper, without production sources or dependency
overrides:

- 2x2, chi=16, 68 gates: direct/dmrg/dmrg2/dmrg3 match native exact replay
  to floating-point precision; raw norm error below 2e-11.
- 4x2, chi=256, 136 gates: one-site DMRG/direct fidelity
  `.9999999999999868`; DMRG2/direct `.9999999999999875`; raw norm error
  below 4e-11. Every output remains a native fermionic tensor network.

Script/report: `/tmp/pepsy_etienne_fixed_validation.py` and
`/tmp/pepsy_etienne_fixed_validation.json`. Original serialized author
boundary files are absent; this is generated-stream validation, not a paper
benchmark result or a claim of finite-chi/Trotter convergence.

## Broader failures and limits

The full-suite attempt was interrupted after 622 passes, 66 skips and two
BP failures:

- `test_fermionic_open_rho_sweep_supports_native_symmetries[U1-2-None]`
- `test_fermionic_open_rho_sweep_supports_native_symmetries[U1U1-4-_site_charge]`

Both reach an empty native density term and fail in Symmray dense conversion.
A private copy of the current source, with the three implementation files
restored to HEAD while preserving other user edits, reproduces both failures.
See `/tmp/pepsy_fermion_fix_full_suite.log` and
`/tmp/pepsy_fermion_fix_bp_baseline.log`. These are existing BP failures,
not repaired as part of this task; full-suite success is not claimed.

An initial extended domain run included slow 3x4 native/exact stress tests
and was interrupted there. Its six earlier PEPO-reference fusion failures
were repaired and all eight corresponding cases passed separately. The
bounded non-slow selection above supplies the completed domain scope.

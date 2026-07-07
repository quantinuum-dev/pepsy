# 2026-07-07 -- SymDMRG2 TeNPy stability and Lanczos speed parity

- Milestone: Close the main SymDMRG2-versus-TeNPy stability gap on the hard
  mapped-2D Fermi-Hubbard DMRG controls, then start reducing local Lanczos work
  with TeNPy-style stopping criteria.
- Branch: `develop`, after `ec8984a`.

## What changed

- Added TeNPy-style random-unitary MPS constructors:
  `SymMPS.random_unitary_evolution(...)` and
  `SymMPS.random_unitary_for_model(...)`.
  - These grow a charge-correct product state with charge-preserving two-site
    random unitaries, canonicalize, and normalize.
  - This gives a well-conditioned random Symmray start for DMRG. Raw
    `SymMPS.random(bond_dim=chi)` remains available as a lower-level stress
    test.
- Kept product-like starts on the automatic gentle bond ramp path. Product
  starts are now grown through the same kind of warmup that made TeNPy robust
  on the 6 by 6 mapped-PBC case.
- Added a TeNPy-style Ritz convergence gate to native block Lanczos.
  - Local solves now track Ritz energy deltas, the residual/gap state-error
    estimate `(RitzRes / gap)**2`, the gap floor, stop reason, and matvec
    counts.
  - Compatibility mode is preserved: setting `local_eig_energy_tol=None`
    disables the automatic P-error gate and keeps the previous full-cap
    behavior.
- Added TeNPy-style adaptive `local_eig_p_tol_to_trunc`.
  - When `local_eig_p_tol="auto"` is active, the default factor is `0.05`.
  - After each sweep, the next sweep's P-error tolerance is derived from the
    measured maximum SVD truncation error and clamped by min/max bounds.
  - Update records are stored in per-sweep convergence diagnostics and exposed
    through `summary()`.

## Benchmark notes

- The corrected 6 by 6 U=8 mapped-PBC controls show the original large gap was
  not a Hamiltonian or truncation bug. Product-grown SymDMRG2 reaches the TeNPy
  energy scale at the same `chi`, and the random-unitary start avoids the raw
  random-block bad basin.
- The first 6 by 6 random-unitary control with the Ritz gate reached
  `E/N - U/4 = -2.425038069` with average native-Lanczos matvec count
  `16.74`.
  - The previous strict local solve used about `19.51` matvecs on the same
    control, so the Ritz gate removed roughly 14 percent of local matvecs while
    preserving the energy.
  - Stop reasons were dominated by the Ritz gate, with remaining max-step
    windows marking harder local subproblems.
- Wall-time comparisons should be treated as noisy on shared heavy runs.
  For speed decisions, use algorithmic counters from `profile_summary()`:
  `num_lanczos_matvecs`, `avg_lanczos_matvecs`, `max_lanczos_matvecs`, and
  `lanczos_stop_reasons`.

## Validation

```text
pytest -q tests/test_sym_dmrg.py
  62 passed

sphinx-build -W -b html docs docs/_build/html
  passed

python -m py_compile src/pepsy/optimizers/sym_dmrg.py tests/test_sym_dmrg.py
  passed
```

## Remaining work

- Re-run the 6 by 6 random-unitary control on `ec8984a` and compare counters
  against the Ritz-gate baseline:
  - final energy density,
  - `avg_lanczos_matvecs`,
  - `lanczos_stop_reasons`,
  - `num_local_eig_p_tol_updates`,
  - `last_local_eig_p_tol_update`.
- If adaptive P-tolerance lowers matvec count without energy drift, keep it as
  the default TeNPy-style speed setting.
- If matvec count is already low but wall time remains high, the next target is
  per-matvec cost: fused/projected-problem contraction routing, cache reuse, and
  block-native hot-loop profiling.

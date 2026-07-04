# 2026-07-04 -- SymDMRG2 quimb-style API and progress

- Milestone: Align the public `SymDMRG2` control surface and progress display
  with `quimb.tensor.DMRG2`.
- Branch: `develop`, after `d627220`.

## What changed

- Added quimb-style initialization aliases.
  - `p0` is accepted as an alias for the initial MPS.
  - `bond_dims` and `cutoffs` are accepted as scalar or sequence controls.
  - The sequence behavior repeats the last entry, matching quimb's DMRG
    schedule convention.
- Added a public `sweep(direction, canonize=True, verbosity=0, **update_opts)`.
  - Directions accept `"R"`/`"L"` as well as `"right"`/`"left"`.
  - `max_bond`, `cutoff`, `method`, and `cutoff_mode` are accepted as
    quimb-style update options.
  - Symmray sweeps record `local_energies` and `total_energies`, matching the
    quimb DMRG object shape.
- Updated `solve` to accept quimb-style controls:
  - `tol`
  - `bond_dims`
  - `cutoffs`
  - `sweep_sequence`
  - `max_sweeps`
  - `verbosity`
  - `suppress_warnings`
- Kept Pepsy's chainable optimizer convention: `solve` still returns `self`.
  The quimb-style boolean result is available as `converged`.
- Added quimb-style sweep progress.
  - `verbosity > 0` prints pre/post sweep energy lines.
  - Per-site sweeps are wrapped in `quimb.utils.progbar(..., ncols=80)`.
- `summary()` now reports `bond_dims`, `cutoffs`, the default sweep sequence,
  and local/total energy trace counts.

## Validation focus

- Added a regression for `p0`, `bond_dims`, `cutoffs`, `max_sweeps`, and
  `sweep_sequence`.
- Added a regression that monkeypatches `quimb.utils.progbar` and verifies
  `verbosity > 0` uses it for one Symmray sweep.
- Updated two-direction DMRG audits to request `sweep_sequence="RL"` and
  `max_sweeps=2`, because `max_sweeps` now counts individual quimb-style
  directional sweeps.

Focused result during implementation:

```text
pytest -q tests/test_sym_dmrg.py
  18 passed
```

Final validation before commit:

```text
pytest -q tests/test_sym_dmrg.py tests/test_public_api.py tests/test_package_layout.py
  484 passed

python -m pyflakes src/pepsy/optimizers/sym_dmrg.py tests/test_sym_dmrg.py
  passed

sphinx-build -W -b html docs docs/_build/html
  passed
```

## Notes

- The Symmray default sweep sequence is now `"R"` like quimb. Use
  `sweep_sequence="RL", max_sweeps=2` for the old one-right-plus-one-left
  audit pattern.
- The dense/quimb backend still delegates to quimb's own `DMRG2`; Pepsy only
  normalizes the wrapper controls and copies the resulting state and energy
  traces back to `SymDMRG2`.

# 2026-07-03 -- SymDMRG2 canonical diagnostics and ED regression

- Milestone: OBC FH U1U1 Lanczos sweeps now leave an auditable record of each
  canonical-window norm check and local solver call.
- Branch / commit: `develop`, after `44f4a9c`.

## What changed

- Added `norm_identity_diagnostics` to `SymDMRG2`.
  - Each normal dense/Lanczos local solve records the site, sweep direction,
    theta dimension, sample count, tolerance, measured `N_eff ~= I` error, and
    pass/fail status.
  - The existing guard still raises if the effective norm is not identity-like
    after OBC canonicalization.
- Added `local_solve_diagnostics` to `SymDMRG2`.
  - Each two-site solve records the requested solver, resolved solver,
    theta-space dimension, local energy, sweep direction, and resolved matvec
    backend.
  - Tiny forced-Lanczos windows that fall back to dense locally are visible as
    `solver="dense"` with `requested_solver="lanczos"`.
- Exposed `last_norm_identity_diagnostic` and `last_local_solve_diagnostic` in
  the public object state and `summary()`.

## Validation focus

- Extended the four-site forced-Lanczos OBC FH U1U1 sweep test to assert that
  every SVD writeback has a matching norm-identity diagnostic and local-solver
  diagnostic.
- Added a dense Jordan-Wigner fixed-sector ED oracle to `test_sym_dmrg.py`.
- Added a four-site OBC FH U1U1 regression where forced Lanczos reaches the
  `(N_up,N_down)=(2,2)` ED ground energy to numerical precision when the
  initial MPS has enough bond-sector support.

## Important learning

The current Krylov space is exactly the current two-site theta block layout.
That is the intended stable implementation point, but it also means the solver
does not invent missing U1U1 sectors. Small initial sector support can improve
energy but remain far above the full fixed-sector ED ground state. A future
noise/sector-enrichment layer should be treated as a separate convergence
feature, not as part of the canonical Lanczos matvec itself.

Focused result during implementation:

```text
pytest -q tests/test_sym_dmrg.py
  13 passed
```

Final validation before commit:

```text
pytest -q tests/test_sym_dmrg.py tests/test_public_api.py tests/test_package_layout.py
  479 passed

pytest -q tests/test_symmetric_tensors.py
  84 passed

python -m pyflakes src/pepsy/optimizers/sym_dmrg.py tests/test_sym_dmrg.py
  passed

sphinx-build -W -b html docs docs/_build/html
  passed

pytest -q
  1116 passed
```

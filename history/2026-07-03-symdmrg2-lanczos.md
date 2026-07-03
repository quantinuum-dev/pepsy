# 2026-07-03 -- SymDMRG2 canonical Lanczos local solver

- Milestone: First quimb-style matrix-free local eigensolver for Symmray
  `SymDMRG2`.
- Branch / commit: `develop`, after `81201ed`.

## What changed

- Added a theta-space adapter that records the exact current two-site block
  sectors, shapes, dtype, and flat dimension. Krylov vectors are flattened and
  unflattened only through this layout.
- Added a projected two-site Hamiltonian `LinearOperator` whose matvec calls
  the existing sector-preserving `two_site_matvec(...)`.
- Added local solver controls:
  `local_solver`, `dense_threshold`, `local_eig_tol`, `local_eig_ncv`,
  `local_eig_maxiter`, `local_eig_backend`, and `norm_check_tol`.
- Added canonical sweep setup: right sweeps call quimb/Symmray
  `right_canonize`, left sweeps call `left_canonize`, then local H-only dense
  or Lanczos solves are allowed only when `N_eff` is identity-like.
- Kept the explicit dense generalized `H_eff theta = E N_eff theta` solve as a
  fallback when the norm check fails for small local spaces.
- Generalized dense alignment of MPS virtual legs. Symmray canonicalization can
  shrink visible charge maps differently on the two sides of a virtual bond, so
  dense reference contractions embed those legs into a union charge map before
  calling NumPy einsum.

## Why

- This follows the stable pattern used by quimb, TeNPy, and ITensor: optimize
  the current two-site tensor in a projected Hamiltonian, reuse the current
  theta as the Krylov starting vector, then split/truncate and move the
  canonical center.
- Keeping the norm check and generalized fallback prevents an H-only Lanczos
  solve from silently running when canonical form is not reliable.

## Validation plan

- `tests/test_sym_dmrg.py` now checks:
  - the projected `LinearOperator` matches the dense effective Hamiltonian;
  - `N_eff` acts like identity after canonicalization;
  - Hermiticity probes pass;
  - Lanczos local energy matches dense local energy;
  - forced Lanczos sweeps on a four-site FH U1U1 chain match
    `MpsEnergyOptimizer`.

Focused result during implementation:

```text
pytest -q tests/test_sym_dmrg.py
  7 passed
```

Final validation before commit:

```text
pytest -q tests/test_sym_dmrg.py tests/test_public_api.py tests/test_package_layout.py
  473 passed

pytest -q tests/test_symmetric_tensors.py
  84 passed

python -m pyflakes src/pepsy/optimizers/sym_dmrg.py tests/test_sym_dmrg.py
  passed

sphinx-build -W -b html docs docs/_build/html
  passed

pytest -q
  1110 passed
```

## Current limitation

- The matrix-free operator still uses dense embedded contractions internally.
  It avoids forming the full dense local Hamiltonian matrix, but it is not yet a
  fully block-sparse/tensor-core matvec.
- Next step: reduce dense embedding inside the matvec, add torch-native block
  contractions where useful, and later mirror quimb's moving-environment
  segment compression for larger periodic chains.

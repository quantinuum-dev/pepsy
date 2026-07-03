# 2026-07-03 -- SymDMRG2 effective norm and longer-chain sweeps

- Milestone: First safe `L > 2` Symmray DMRG2 correctness-reference sweeps.
- Branch / commit: `develop`, after `5f57b21`.

## What changed

- Added dense left/right norm environments for `<psi|psi>` alongside the
  existing Hamiltonian environments for `<psi|MPO|psi>`.
- Added `two_site_norm_matvec(...)`, which applies `N_eff` in exactly the same
  current two-site `theta` block layout used by `two_site_matvec(...)`.
- Added a dense generalized local eigensolver for
  `H_eff theta = E N_eff theta`, with small-norm directions removed by a
  relative `norm_rcond`.
- Replaced the `L=2`-only Symmray solve guard with small-chain two-direction
  sweeps. The sweep updates the appropriate left or right Hamiltonian and norm
  environment as it moves, and rebuilds the full environments between half
  sweeps for correctness.

## Why

- The earlier dense local eigensolver was safe for a whole `L=2` chain because
  the active theta basis has no external MPS environment.
- On longer chains, an H-only local solve assumes the MPS is exactly canonical
  around the two-site window. Symmray SVD can preserve the represented state
  while changing block charge maps, so the safer first implementation is an
  explicit generalized problem with `N_eff`.

## How it was validated

- `tests/test_sym_dmrg.py` now checks:
  - Hamiltonian environment energy equals `MpsEnergyOptimizer`;
  - norm environment value equals the current MPS norm;
  - both `H_eff` and `N_eff` matvecs preserve theta block sectors;
  - a three-site FH U1U1 solve completes and its final energy matches
    `MpsEnergyOptimizer` and the rebuilt environment energy.

Focused result during implementation:

```text
pytest -q tests/test_sym_dmrg.py
  4 passed
```

Final validation before commit:

```text
pytest -q tests/test_sym_dmrg.py tests/test_public_api.py tests/test_package_layout.py
  470 passed

pytest -q tests/test_symmetric_tensors.py
  84 passed

python -m pyflakes src/pepsy/optimizers/sym_dmrg.py tests/test_sym_dmrg.py
  passed

sphinx-build -W -b html docs docs/_build/html
  passed

pytest -q
  1107 passed
```

## Current limitation

- This is intentionally a dense reference path. It is useful for correctness,
  API shape, and small regression tests, but not for production bond
  dimensions.
- Next step: replace dense matrix construction with a block-sparse Lanczos or
  `LinearOperator` matvec that uses the same sector-preserving `theta` layout,
  then add torch-native kernels for the block contractions where they matter.

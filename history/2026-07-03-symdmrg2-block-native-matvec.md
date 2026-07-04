# 2026-07-03 -- SymDMRG2 block-native projected matvec

- Milestone: First block-native Symmray `H_eff` matvec for the OBC
  `SymDMRG2` path.
- Branch / commit: `develop`, after `15874f2`.

## What changed

- Added `matvec_backend` to `SymDMRG2`.
  - `matvec_backend="auto"` resolves to the Symmray block-native matvec for
    Symmray MPOs.
  - `matvec_backend="dense_reference"` keeps the older NumPy dense-aligned
    matvec as a selectable validator.
- Added cached block-sparse left/right environments for the projected
  `<psi|MPO|psi>` problem. These use separate bra virtual indices internally
  and expose the active cut as output bra/input-ket legs for the local matvec.
- Added a safe pairwise contraction helper for projected local contractions.
  It contracts one shared index with Symmray `tensordot(mode="blockwise")`,
  renames any remaining shared indices on the right tensor, then traces those
  pairs inside the resulting Symmray tensor. This avoids the fused multi-leg
  shape-mismatch path observed in direct `qtn.tensor_contract(...)`.
- `two_site_matvec(...)` now dispatches to the block-native Symmray path by
  default and projects the output back to the exact current theta block sectors.
- The sweep loop refreshes block environments incrementally alongside the
  existing dense energy and norm environments.

## Validation focus

- Added regression coverage comparing block-native and dense-reference matvecs
  on every active site of a four-site FH U1U1 OBC chain, including random
  theta-space vectors.
- Added coverage that the dense-reference matvec backend remains explicitly
  selectable.
- Existing Lanczos and dense local-solve tests now exercise the block-native
  matvec by default.

Focused result during implementation:

```text
pytest -q tests/test_sym_dmrg.py
  12 passed
```

Final validation before commit:

```text
pytest -q tests/test_sym_dmrg.py tests/test_public_api.py tests/test_package_layout.py
  478 passed

pytest -q tests/test_symmetric_tensors.py
  84 passed

python -m pyflakes src/pepsy/optimizers/sym_dmrg.py tests/test_sym_dmrg.py
  passed

sphinx-build -W -b html docs docs/_build/html
  passed

pytest -q
  1115 passed
```

# 2026-07-03 -- SymDMRG2 SVD diagnostics and Lanczos small-space fix

- Milestone: Auditable Symmray SVD sector tracking for the OBC `SymDMRG2`
  Lanczos path.
- Branch / commit: `develop`, after `6f9688c`.

## What changed

- Added `svd_diagnostics` to `SymDMRG2`. Every Symmray two-site writeback now
  records the sweep direction, site pair, bond name, `chi`, cutoff, and the
  left/right charge sectors kept by `theta.split(..., method="svd")`.
- Added `last_svd_diagnostic` and exposed the diagnostic count/latest record
  through `summary()`.
- Fixed the Lanczos `ncv` clamp for tiny local spaces. With `k=1`, ARPACK
  requires `k + 1 < ncv <= dim`; a three-dimensional theta space now uses
  `ncv=3` rather than an invalid `ncv=2`.
- Added a six-site OBC forced-Lanczos regression that checks final MPO energy
  against `MpsEnergyOptimizer` and verifies every SVD diagnostic respects
  `chi`.

## Block-native matvec probe

- A direct Symmray-native projected-matvec probe was attempted before replacing
  the dense-aligned matvec.
- Pairwise contractions such as MPO-with-environment and MPO-with-theta worked,
  but all-at-once or fixed-order contractions could fail after intermediate
  tensors pruned charge maps on shared virtual/MPO legs.
- Conclusion: keep the stable LinearOperator path for now. It already restricts
  Krylov vectors to the exact current theta blocks, but its matvec still embeds
  those blocks into dense charge-ordered arrays internally. A safe sparse
  index-expansion adapter is needed before making the matvec fully
  block-native.

## Validation

Focused result during implementation:

```text
pytest -q tests/test_sym_dmrg.py
  10 passed
```

Final validation before commit:

```text
pytest -q tests/test_sym_dmrg.py tests/test_public_api.py tests/test_package_layout.py
  476 passed

pytest -q tests/test_symmetric_tensors.py
  84 passed

python -m pyflakes src/pepsy/optimizers/sym_dmrg.py tests/test_sym_dmrg.py
  passed

sphinx-build -W -b html docs docs/_build/html
  passed

pytest -q
  1113 passed
```

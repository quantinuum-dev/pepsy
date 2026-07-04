# 2026-07-04 -- SymDMRG2 sector enrichment

- Milestone: First sector-enrichment/noise convergence layer for OBC Symmray
  `SymDMRG2`.
- Branch / commit: `develop`, after `158c994`.

## What changed

- Added opt-in `sector_enrichment="template"` to `SymDMRG2`.
  - The optimizer builds a same-charge random template MPS with
    `sector_enrichment_bond_dim`.
  - It merges the template virtual-bond charge maps into the current OBC MPS.
  - Existing tensor blocks are copied into the expanded layout.
  - Newly valid blocks are initialized with zero or small `sector_noise`.
- Added `sector_enrichment_diagnostics` and
  `last_sector_enrichment_diagnostic`.
  - Diagnostics record template bond sectors, per-site block counts, copied
    blocks, and the number of newly added valid blocks.
- The enrichment runs once before the first Symmray sweep. The local dense or
  Lanczos eigensolver itself remains unchanged: it simply sees a larger current
  theta block layout.

## Validation focus

- Added an L=4 OBC FH U1U1 regression starting from narrow `bond_dim=2`
  support.
- With `sector_enrichment="template"`, `sector_enrichment_bond_dim=12`, and
  small `sector_noise`, forced Lanczos reaches the same fixed
  `(N_up,N_down)=(2,2)` dense-Jordan-Wigner ED energy that previously required
  a rich initial MPS.
- The canonical `N_eff ~= I` diagnostics still pass on every two-site window
  after enrichment.

Focused result during implementation:

```text
pytest -q tests/test_sym_dmrg.py
  14 passed
```

Final validation before commit:

```text
pytest -q tests/test_sym_dmrg.py tests/test_public_api.py tests/test_package_layout.py
  480 passed

pytest -q tests/test_symmetric_tensors.py
  84 passed

python -m pyflakes src/pepsy/optimizers/sym_dmrg.py tests/test_sym_dmrg.py
  passed

sphinx-build -W -b html docs docs/_build/html
  passed

pytest -q
  1117 passed
```

## Remaining work

- Profile enriched runs on larger chains.
- Study repeated/adaptive enrichment schedules instead of one-shot template
  expansion.
- Add torch-native block matvec support later; this slice remains NumPy/SciPy
  eigensolver oriented.

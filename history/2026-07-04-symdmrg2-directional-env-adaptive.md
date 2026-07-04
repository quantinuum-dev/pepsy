# 2026-07-04 -- SymDMRG2 directional environments and adaptive enrichment

- Milestone: First performance/scale pass after the Symmray `SymDMRG2`
  correctness and sector-enrichment milestones.
- Branch: `develop`, after `7505933`.

## What changed

- Added directional sweep environment setup.
  - A right sweep now prebuilds the right dense/norm/block environments and
    seeds only the left boundary.
  - A left sweep now prebuilds the left dense/norm/block environments and
    seeds only the right boundary.
  - The moving side is refreshed incrementally after each two-site writeback.
- Removed the full post-sweep dense/norm environment rebuild from the Symmray
  solve loop.
  - At the end of each sweep direction, Pepsy completes the scalar boundary
    environment from the moving side.
  - Normalized energies now divide by `norm_environment_value()` instead of a
    separate full `<psi|psi>` contraction.
- Added `sector_enrichment="adaptive"` as an alias for repeated template
  enrichment before every sweep.
  - This reintroduces valid charge sectors that SVD truncation may prune between
    sweeps.
  - Diagnostics include the sweep index and distinguish `adaptive_template`
    from the one-shot `template` mode.
- Template enrichment now preserves the current physical-sector map when
  constructing the same-charge random template, instead of collapsing the
  current physical index to a bare dimension. This matters for restricted U1U1
  layouts.

## Validation focus

- Added a directional-environment regression that counts dense left/right
  environment steps and verifies the unused side is not prebuilt.
- Added an adaptive-enrichment regression that runs two sweeps and records two
  enrichment diagnostics.

Focused result during implementation:

```text
pytest -q tests/test_sym_dmrg.py
  16 passed
```

Final validation before commit:

```text
pytest -q tests/test_sym_dmrg.py tests/test_public_api.py tests/test_package_layout.py
  482 passed

python -m pyflakes src/pepsy/optimizers/sym_dmrg.py tests/test_sym_dmrg.py
  passed

sphinx-build -W -b html docs docs/_build/html
  passed
```

## Remaining work

- Profile block-native matvec timing on larger target chains.
- Compare one-shot and adaptive enrichment schedules on production-like FH
  workloads.
- Add torch-native block contractions later where they materially help.

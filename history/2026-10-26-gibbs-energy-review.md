# 2026-10-26 — Gibbs MPS benchmark review

- Scope: review `GibbsMps` against the corner-removed ITF benchmark and fix
  justified package API gaps.
- Branch / baseline commit: `develop` / `aa750d7`.
- Commit status: working-tree edits only; nothing staged or published.

## What changed

- Added normalized local `GibbsMps.energy()` for the Hamiltonian coefficients
  used by the latest successful preparation. It contracts the interleaved
  purification with a reusable Pepsy contraction optimizer, without building
  a density MPO.
- Reject nonfinite `beta` and `dt` before Trotter construction.
- Updated [the Gibbs API guide](../docs/api/optimizers/gibbs_mps.md),
  [implementation note](../docs/development/notes/gibbs_mps.md), and changelog.

## Validation

- `python -m pytest -q -o addopts='' tests/test_gibbs_mps.py
  tests/test_public_api.py tests/test_package_layout.py
  tests/test_quimb_compat.py` → 105 passed, 2 deprecation warnings.
- Small NumPy dense reduction, parameter rebinding, and Torch CPU value and
  analytic-gradient checks cover local energy; a 3×4 benchmark replay
  reproduced the previously saved local energy. A regression checks that the
  Pepsy optimizer is passed into Quimb's local readout and reused.
- A 10×10 β=0 Torch CPU `float64` probe measured 0.93 s for energy, including
  first optimizer construction; it does not measure the entangled χ=128 case.
- `python -m ruff check src tests` and `git diff --check` passed. The full
  numerical suite and production CUDA run were not performed.

## Findings and limits

- Preparation dominates the measured 10×10 runtime; local energy readout is a
  small fraction. Explicit canonicalization is also a small fraction of the
  inclusive direct replay stage. See the dated implementation note for the
  exact benchmark settings and timings.
- Reusing factorizations of repeated thermal gates remains an unverified
  performance idea. No GPU speedup or χ convergence is claimed.

# 2026-07-03 -- SymDMRG2 API scaffold

- Milestone: First DMRG2 implementation slice after the Fermi-Hubbard U1U1 MPO
  fix.
- Branch / commit: `develop`, after `cff4741`.

## What changed

- Added `pepsy.SymDMRG2` / `pepsy.optimizers.SymDMRG2`.
- Dense/quimb MPOs now delegate to `quimb.tensor.DMRG2`.
- Symmray MPOs auto-select the Pepsy block-sparse path, infer `total_charge`
  from `SymMPS.overall_charge()`, and compute the initial direct MPO energy.
- Symmray `solve()` intentionally raises `NotImplementedError` until the local
  two-site sector eigensolver is implemented.

## Why

- The FH U1U1 MPO path is now clean enough to start DMRG work.
- A stable API lets examples and tests target one entry point while the
  Symmray-specific internals are filled in incrementally.
- Quimb should remain the implementation reference and execution backend for
  ordinary dense/quimb MPOs.

## How it was validated

- `pytest -q tests/test_sym_dmrg.py` -> 3 passed.
- `pytest -q tests/test_public_api.py tests/test_package_layout.py` -> 466
  passed.
- `pyflakes src/pepsy/optimizers/sym_dmrg.py tests/test_sym_dmrg.py` -> passed.

## Decisions / findings

- Do not pass Symmray arrays directly to quimb DMRG2 for now. The wrapper raises
  a clear error for `backend="quimb"` with Symmray data.
- The first Symmray DMRG target should remain the bosonic JW MPO plus U1U1
  block-sparse MPS representation, not a native fermionic MPO.

## Next step (do this first next time)

- Implement left/right DMRG environments for the Symmray MPO sandwich and check
  that the environment energy reproduces `MpsEnergyOptimizer`.

## Open questions / blockers

- Exact internal representation for the two-site effective Hamiltonian matvec.
- Whether the first local solve should use dense per-sector eigensolves or go
  straight to a block-sparse Lanczos wrapper.

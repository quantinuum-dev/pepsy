# 2026-07-03 -- SymDMRG2 environments and dense local solve

- Milestone: First Pepsy-native Symmray DMRG2 internals after the API scaffold.
- Branch / commit: `develop`, after `b85cbe5`.

## What changed

- `SymDMRG2` now converts fermionic `SymMPS` inputs to the bosonic JW/U1U1
  representation before Symmray DMRG work.
- Added dense left/right environments for `<psi|MPO|psi>`.
- Added environment-energy validation via `environment_energy()`.
- Added `two_site_theta(...)`, `two_site_matvec(...)`, and
  `dense_local_eigensolve(...)`.
- Enabled exact dense local solve plus Symmray SVD writeback for whole-chain
  `L=2` correctness runs.

## Why

- The repaired FH U1U1 MPO is bosonic JW, so the DMRG state should be optimized
  in the same bosonic block-sparse representation.
- The dense local solve is a correctness reference before adding block-sparse
  Lanczos or torch-native local eigensolvers.

## How it was validated

- `tests/test_sym_dmrg.py` checks:
  - quimb delegation for dense MPOs;
  - Symmray environment energy equals `MpsEnergyOptimizer`;
  - the two-site matvec preserves exactly the current theta block sectors;
  - `L=2` local solve/writeback produces an MPO energy matching
    `MpsEnergyOptimizer`.

## Decisions / findings

- Symmray SVD can split the optimized theta and preserve the state for MPO
  contraction, but it may redistribute the tensor charge and shrink physical
  charge maps. Dense environment and matvec code therefore embeds MPS physical
  legs into the MPO physical charge map at contraction time.
- Longer-chain local updates are still disabled by default. They need canonical
  center / effective norm handling before the H-only dense local solve is safe.

## Next step (do this first next time)

- Add canonical-center movement or an explicit effective norm environment so
  the same two-site dense solve can be used safely for `L > 2` sweeps.

## Open questions / blockers

- Whether to canonicalize the Symmray MPS around each two-site window or solve a
  generalized local eigenproblem with both H and N environments.
- How soon to replace the dense reference matrix with a block-sparse Lanczos
  matvec.

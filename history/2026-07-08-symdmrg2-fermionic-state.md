# 2026-07-08 -- SymDMRG2 fermionic_state() (debosonize) + leg-order fix

- Milestone: Let a `SymDMRG2` (Jordan-Wigner bosonic) ground state drive native
  fermionic gate streams and observables, then fix an off-diagonal sign bug in
  the conversion.
- Branch: `develop`, commits `1b064cd` (feature) and `b82b902` (fix).

## What changed

- Added `MpsEnergyOptimizer._debosonize_fermionic_tn`, the exact inverse of
  `_bosonize_fermionic_tn`. It undoes the self-inverse bosonization sign flips
  and rebuilds each site as a fermionic array via
  `FermionicArray.from_blocks(..., label=site)`, so odd-parity tensors regain
  their dummy mode. Bond dimensions, charges, block sectors, and shapes are all
  preserved (a per-site gauge change, so the fermionic MPS keeps the DMRG bond
  dimension).
- Added the public `SymDMRG2.fermionic_state()`, which converts the converged
  bosonic `state` into a native fermionic `SymMPS` using the fermionic
  `init_mps` as the metadata template.
- Fixed a leg-order bug in the debosonize. quimb's `DMRG2` leaves the physical
  leg first on the boundary tensor (`('k0','b0-1')`), whereas the bosonization
  and `SymMPS.for_model` assume physical-leg-last (`('b0-1','k0')`). Because
  `from_blocks` is leg-order sensitive, reconstructing from the raw DMRG order
  produced wrong fermionic swap phases. The debosonize now normalizes each site
  to physical-leg-last (a phase-free bosonic transpose) before reconstruction.

## Why it was subtle

The wrong-phase state stayed correct for every diagonal observable (doublon
density), for the bosonic-MPO sandwich energy, and for even-parity operators
(the eta-pairing `Delta^dag Delta` correlator). Only the native-fermionic term
energy and odd (single-fermion) correlators were wrong, so `terms != MPO`. The
first "involution" check passed only because it used a fresh `for_model` state,
which already has the physical leg last; states from DMRG or imaginary-time
evolution are in quimb's canonical form and expose the leg-order sensitivity.

The correctness criterion that catches it is the terms-vs-MPO check from
`fh_mps.ipynb`: for a genuine fermionic state, `SymMPS.energy(ham.terms)` must
equal `<psi|H_mpo|psi>`.

## Validation

- Dense Jordan-Wigner ED (L=4 chain, U=6): after the fix,
  `SymMPS.energy(ham.terms) == <psi|H_mpo|psi> == E_dmrg == E_ED = -1.431936`;
  doublon densities and eta-pairing correlators match ED exactly; overlap with
  an independent native imaginary-time ground state is 0.9999.
- `tests/test_sym_dmrg.py::test_symdmrg2_fermionic_state_roundtrip_and_observables`
  now also asserts the leg-order-sensitive `SymMPS.energy(ham.terms) == E_dmrg`
  guard. Full `tests/test_sym_dmrg.py` passes (74 tests); `pyflakes` clean.

## Notes / follow-ups

- Standalone single-fermion `<c^dag_i c_j>` measured via `measure([Cdag, C])`
  still shows a small gap versus ED plus an "odd parity, no label" Symmray
  warning. That is a measurement-operator artifact (odd-parity operators need
  dummy-mode labels), not a state bug; it affects a native reference state
  identically. Even-parity operators and the energy are exact.
- Downstream: `pepsy_examples/fermi_hubbard/sim_henrik_pairing.ipynb` uses
  `fermionic_state()` for the light-pulse start. Its numbers are unchanged by
  the fix because it only measures even-parity observables, which were already
  correct.

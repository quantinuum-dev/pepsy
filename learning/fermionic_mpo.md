# Fermionic Hamiltonian MPOs in Pepsy

Living record for the spinful Fermi-Hubbard `U1U1` MPS/MPO work. This note is
the conceptual "what did we learn?" companion to
`docs/development/fermi_hubbard_u1u1_mpo_notes.md`, which keeps the practical
debug log and validation details.

## Two Conventions

Fermion signs are safe only when the state, operators, and contraction all use
one coherent convention.

- **A. Bosonic Jordan-Wigner (JW).** The tensors are ordinary bosonic tensors.
  Fermionic anticommutation is represented by explicit parity strings in the
  operator network. Contracting `<psi|H_mpo|psi>` is then a plain bosonic tensor
  contraction.
- **B. Native fermionic / graded.** The tensors carry fermionic metadata. Swap
  and contraction phases are supplied by the graded tensor algebra itself.
  Symmray's fermionic arrays live in this convention: `label`, `dual`,
  `dummy_modes`, charge parity, and lazy block phases are part of the data.

The main bug class was mixing these conventions. Pepsy evolves and measures
local terms using native fermionic Symmray tensors, but its direct MPO energy
path currently evaluates a bosonic JW MPO. That bridge is valid only if the MPS
is bosonized into the same JW gauge as the MPO.

## Current Pepsy Bridge

The repaired finite-chain route is:

1. Build the state as a fermionic Symmray MPS.
2. Build the spinful Fermi-Hubbard MPO as a bosonic Symmray MPO with explicit
   JW parity operators.
3. Before the MPO sandwich, copy the MPS into bosonic Symmray arrays and apply
   the missing sector phases from Symmray's left-to-right fermionic contraction.
4. Contract the bosonized MPS with the bosonic JW MPO.

Relevant code:

- `src/pepsy/tensors/symmetric.py`: `_fh_u1u1_jw_local_ops` and
  `SymHamiltonian.to_mpo(model="fermi_hubbard_u1u1")`.
- `src/pepsy/optimizers/energy/peps.py`:
  `MpsEnergyOptimizer._bosonize_fermionic_tn(...)`.

## MPO Sign Convention

For a hopping edge `i < j`, the implemented site-major JW convention is:

- forward term: `c_i^dag P_i P_{i+1} ... P_{j-1} c_j`
- Hermitian conjugate: `P_i c_i P_{i+1} ... P_{j-1} c_j^dag`
- coefficient: `-t_sigma`

Equivalently, the parity string spans the left endpoint through the site before
the right endpoint. In the MPO code this means the left endpoint is folded as:

- `first @ parity` for the forward `c_i^dag ... c_j` channel
- `parity @ first` for the Hermitian-conjugate channel

Interior sites carry parity. There is no parity prefix from site 0, no
alternating `(-1)^i` patch, and no source-term fallback in the direct MPO
expectation.

This corrects an earlier investigation note that had concluded "bare endpoint
plus interior parity only." That was an intermediate false trail. The passing
implementation and tests now use endpoint parity.

## Bosonization Phase

Naively re-wrapping fermionic Symmray blocks as bosonic blocks loses phases that
are produced by the native fermionic contraction. The bridge now:

- calls `phase_sync()` before reading the block dictionary;
- applies a bond-sector phase from moving the outgoing virtual mode past the
  local physical leg;
- includes the contracted-mode phase flip;
- includes the next-tensor dummy-mode crossing when that tensor has odd parity.

This is the part that makes high-bond evolved states agree, not just small
product-state or two-site checks.

## Validation Status

Committed implementation:

- `1ebf5d4 Fix fermionic FH MPO energy path`
- `affefad Quiet expected test warnings`
- `7d1dec4 Refine tensor schematic visuals`

Numerical validation:

- `pytest -q tests/test_symmetric_tensors.py` passed with 84 tests after the MPO
  regression additions.
- Full suite passed with 1100 tests after warning cleanup.
- A 4 by 3 periodic square-lattice `fh_mps`-shaped scratch run matched local
  terms and direct MPO energy through the first two imaginary-time schedule
  blocks:
  - initial diff: `0.0`
  - after `tau=0.1`, 10 steps: about `-2.6e-15`
  - after `tau=0.03`, 10 steps: about `-6.1e-16`

The final `tau=0.01` notebook block was not completed in that scratch run
because Symmray QR/canonicalization became slow; no mismatch appeared before
the interruption.

## External Lessons

- **TeNPy / ITensor / DMRG++:** Keep the fermionic string rule in the operator
  algebra. Do not repair a bosonic MPO by later sprinkling signs across graph
  edges.
- **MPS-FQE / pyblock3 / block2:** Useful as an external oracle and architecture
  reference. It converts basis/sign conventions explicitly and delegates
  fermionic MPO construction to a backend that owns the algebra.
- **Grassmann tensor-network papers:** Support the Symmray interpretation: a
  native fermionic tensor network is not the same object as a bosonic network
  with guessed local matrices. The bridge must be explicit and tested.

## Open Items

- Re-run the full `fh_mps` notebook through the final `tau=0.01` block and
  update saved output when runtime is acceptable.
- Keep optional external-oracle checks on the table, especially MPS-FQE or
  block2/pyblock3, but do not add those as Pepsy dependencies.
- A fully native fermionic MPO is still future work. It would avoid bosonizing
  the MPS, but Symmray does not currently provide a ready-made MPO class for
  this route.
- `SymDMRG2` now exists as the API and quimb-adapter scaffold: dense/quimb MPOs
  run through `quimb.tensor.DMRG2`, while Symmray FH MPOs initialize the Pepsy
  block-sparse path, infer the fixed charge, and record the initial MPO energy.
- The Symmray DMRG internals follow the ITensor/TeNPy/quimb pattern more
  closely now: canonicalize the MPS center, view the projected two-site
  Hamiltonian as a linear operator in the exact current `theta` block layout,
  use the current theta vector as the Krylov/Lanczos initial vector, and split
  the optimized theta with Symmray SVD.
- The Symmray DMRG path assumes an OBC MPS/MPO chain. Periodic lattice edges
  should be represented as long-range terms in that OBC MPO.
- The explicit effective norm remains in the code as a validator and
  diagnostic. After OBC canonicalization, `N_eff` should be identity-like; if
  it is not, the normal dense/Lanczos path raises a canonicalization/alignment
  error rather than quietly solving a generalized problem.
- The real remaining DMRG work is performance and scale: reduce dense embedding
  in the matvec and add torch-native block contractions where useful.

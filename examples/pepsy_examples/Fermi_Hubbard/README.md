# Direct Fermionic Fermi-Hubbard Examples

This folder contains Pepsy/Symmray examples for Fermi-Hubbard simulations that
work directly in fermionic tensor-network space. These examples do not use the
fermion-to-qubit encoding from arXiv:2511.02125.

## Main methods reference

Use Gao et al., "Fermionic tensor network contraction for arbitrary
geometries", Phys. Rev. Research 7, 023193 (2025),
https://doi.org/10.1103/PhysRevResearch.7.023193 as the main Pepsy/Symmray
reference for direct fermionic Fermi-Hubbard tensor networks.

Implementation cues from that paper:

- Store parity, Abelian charge labels, and leg order in the fermionic array
  object, not in a separate fermion-to-qubit mapping layer.
- Prefer local edge/order metadata for arbitrary graphs; square-lattice PEPS,
  snake MPS embeddings, checkerboards, bilayers, and random graphs should all
  flow through the same fermionic tensor path when possible.
- Let quimb/cotengra optimize the tensor-network graph contraction order; the
  fermionic signs are handled by Symmray's swap/parity algebra.
- Keep dense fallbacks as small-system references only, and verify that gates,
  observables, and evolved states remain Symmray/FermionicArray-backed.

## Current notebook

- `half_filled_4x4_direct_fermions.ipynb`

The notebook builds both MPS and PEPS versions of a 4 x 4 spinful
Fermi-Hubbard model with:

- symmetry: `U1U1`
- local sectors: `(0, 0)`, `(0, 1)`, `(1, 0)`, `(1, 1)`
- half-filled charge: `(N_up, N_down) = (8, 8)`
- parameters: `t = 1`, `U/t = 8`
- no Jordan-Wigner, compact, or Octagon fermion-to-qubit mapping

The energy and imaginary-time cells are off by default so the notebook opens
quickly. Turn them on after the construction and block summaries look right.

## Paper targets to test next

For the doped checkerboard case in arXiv:2511.02125:

- lattice: 6 x 6 split into 2 x 2 plaquettes
- direct-fermion charge: `(15, 15)`
- doping: 30 particles on 36 sites, i.e. six holes from half filling
- weak-coupling point: `t_prime = 0`, `U/t = 2`
- exact shifted energy target: `<H>/N - U/4 = -1.27367`
- d-wave average target: `0.108` theory, `0.079 +/- 0.005` experiment

The next implementation step is adding weighted checkerboard Hubbard edges and
fermionic singlet-pair observables so those numbers can be tested directly.

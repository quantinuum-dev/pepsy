# Pepsy/Symmray Fermi-Hubbard Simulation Plan

## Goal

Build Pepsy examples that use fermionic Symmray tensor networks for direct
Fermi-Hubbard simulations. The main methods reference for the Pepsy/Symmray
path is Gao et al., "Fermionic tensor network contraction for arbitrary
geometries", Phys. Rev. Research 7, 023193 (2025).

Use arXiv:2511.02125, "Superconducting pairing correlations on a trapped-ion
quantum computer", as the physics benchmark source for square-lattice,
checkerboard, and bilayer settings.

The examples should prioritize reproducible classical checks over reproducing
the trapped-ion circuit encoding. Use Pepsy's native fermionic tensor-network
path whenever possible so that the examples test the physics directly, without
the Octagon fermion-to-qubit overhead used in the hardware experiment.

Paper sources:

- Main direct-fermion TN reference:
  https://doi.org/10.1103/PhysRevResearch.7.023193
- Open arXiv text for the methods reference:
  https://arxiv.org/abs/2410.02215
- Physics benchmark paper:
- https://arxiv.org/pdf/2511.02125
- Version checked while writing this plan: arXiv v3, dated 2026-02-18.

## Implementation Notes From PRR 7, 023193

- Keep the fermions native: parity, Abelian charge sectors, and leg-order
  metadata should stay with Symmray fermionic arrays rather than being encoded
  through Jordan-Wigner, compact, or Octagon qubit mappings.
- Prefer graph-level geometry metadata. The PRR paper supports both globally
  ordered and locally ordered conventions, with local ordering most natural for
  arbitrary graphs; Pepsy examples should record site maps, edge orientations,
  and any MPS snake embedding explicitly.
- Contraction planning can remain graph based. quimb/cotengra should choose
  exact or approximate contraction trees from the tensor-network graph, while
  Symmray handles fermionic swap/parity signs during transposition,
  contraction, and decomposition.
- The PRR benchmarks use fermionic PEPS for half-filled Hubbard models on 3D
  diamond lattices and random 3-regular graphs. Those are not the immediate
  square-lattice targets below, but they are useful future regression tests for
  Pepsy arbitrary-graph support.
- The paper's cluster-energy idea suggests an eventual Pepsy smoke benchmark:
  compare full approximate contraction against small-radius graph clusters
  with simple-update gauges on a fixed fermionic PEPS.

## Existing Pepsy Pieces To Reuse

- `pepsy.SymMPS` and `pepsy.SymPEPS` for Symmray-backed MPS/PEPS states.
- `pepsy.SymHamiltonian.from_edges("fermi_hubbard", ...)` for local
  Symmray Fermi-Hubbard terms.
- `pepsy.default_physical_sectors(model="fermi_hubbard_u1u1")`, which gives
  spin-resolved spinful local occupation sectors
  `{(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}`.
- `pepsy.site_charge_from_occupations(...)` to fix `U1U1` sectors such as
  `(N_up, N_down)`.
- `state.time_evolve(...)`, `state.apply_gates(...)`, and
  `state.measure(...)` for gate application and local-observable checks.
- `pepsy.symmray_mps_summary(...)` and `pepsy.symmray_peps_summary(...)` for
  block-size diagnostics.

Use spin-resolved `U1U1` for the paper-style direct-fermion Hubbard examples.
The older `model="fermi_hubbard"` preset remains useful when only total
particle-number `U1` is desired.

## Target Paper Settings

### 1. Half-filled square-lattice Hubbard and light-induced eta pairing

Paper setting:

- Lattice: periodic 6 x 6 square lattice.
- Model: nearest-neighbor spinful Fermi-Hubbard, `t = 1`.
- Direct-fermion symmetry target: spin-resolved `U1U1` charge `(18, 18)` for
  the paper's 6 x 6 half-filled setting.
- Boundary treatment: average over periodic/anti-periodic choices in x and y
  where feasible; otherwise start with periodic and document the difference.
- Benchmark limit: `U = 0`, 36 fermions, track imbalance `I_A = n_A - n_Abar`
  under hopping evolution with Trotter step `tau = 0.5`.
- Interacting low-energy setting: `U/t = 8`.
- Best shallow state-preparation depth in the paper:
  `(D_Heisenberg, D_Hubbard) = (1, 1)`.
- Paper reference values for sanity checks:
  energy density `<H>/N - U/4` is about `-2.36` raw, `-2.43` mitigated, and
  periodic-boundary ground-state benchmark is about `-2.5278`.
- Light pulse:
  `A(s) = pi * (1 - cos(omega * s)) / 2`, aligned with the lattice diagonal,
  `omega = 4*pi/3 ~= U/2`.
- Pulse evolution: evolve to `pi/omega` using two second-order Trotter steps
  with `tau = pi/(2*omega) = 0.375`.
- Relaxation check: apply two additional Hubbard Trotter steps at `tau = 0.375`
  with the field off.
- Observables:
  onsite doublon-pair correlations
  `P_eta(x, y) = (1/N) sum_i <Delta_i^dag Delta_{i+(x,y)} + h.c.>`,
  staggered average `P_eta_stag`, spin-spin correlations, energy density, and
  the free-fermion imbalance benchmark.

Example target:

- `half_filled_eta_pairing.py`
- First make a 4 x 4 exact or high-bond reference.
- Then run 6 x 6 SymPEPS with a small sweep over PEPS bond dimension and
  boundary/environment `chi`.
- Save compact JSON/CSV summaries: energy, norm drift, max bond, block density,
  `P_eta(x, y)`, and `P_eta_stag`.

### 2. Doped checkerboard Hubbard and d-wave bond pairing

Paper setting:

- Lattice: 6 x 6 square lattice split into 2 x 2 plaquettes.
- Doping: 1/6 hole doping.
- Direct-fermion symmetry target: spin-resolved `U1U1` charge `(15, 15)`,
  i.e. 30 particles on 36 sites and six holes relative to half filling.
- Initial weak-coupling point: decoupled plaquettes, `t_prime = 0`, `U/t = 2`.
- Strong intra-plaquette hopping: `t`.
- Weak inter-plaquette hopping: `t_prime`.
- Effective preparation model: ferromagnetic 3 x 3 XXZ model with
  anisotropy approximately `delta = -1` and `sum_i Z_i = -3`.
- Plaquette injection:
  `|0> -> |s>` in the quarter-filled 2 x 2 plaquette sector,
  `|1> -> |d>` in the half-filled 2 x 2 plaquette sector.
- Paper reference values:
  exact weak-coupling energy density `<H>/N - U/4 = -1.27367`;
  d-wave average is about `0.108` in theory and `0.079 +/- 0.005` in the
  experiment.
- Adiabatic target check:
  one second-order Trotter step of size `tau = 0.3` toward
  `U/t = 8`, `t_prime/t = 1/2`.
- Observables:
  strong-bond singlet pair correlations
  `P_b(x, y) = (4/N) sum_<ij> <Delta_ij Delta_{ij+(x,y)}^dag + h.c.>`,
  d-wave signed average, and energy density.

Example target:

- `checkerboard_dwave_pairing.py`
- Implement edge grouping for strong and weak bonds.
- Build exact 2 x 2 plaquette states `|s>` and `|d>` in dense form, then
  convert to fermionic Symmray-compatible tensors or initialize a product PEPS
  over plaquettes.
- Validate 2 x 2 plaquette energies before the 6 x 6 run.
- Run the weak-coupling state first, then the one-step ramp to the target
  checkerboard parameters.

### 3. Bilayer Hubbard nickelate-inspired s-wave rung pairing

Paper setting:

- Lattice: two coupled 4 x 4 square Hubbard layers, i.e. 4 x 4 x 2 sites.
- Filling: two quarter-filled layers in the bilayer construction.
- Model:
  `H_bilayer = H_Hubbard^A + H_Hubbard^B + H_exchange`,
  where
  `H_exchange = J sum_i (S_iA . S_iB - n_iA n_iB / 4)`.
- Perturbative limit: `t/J -> 0`, `U >= 0`; ground subspace is rung singlets
  and hole pairs.
- Effective preparation model: ferromagnetic 4 x 4 XXZ with
  `delta = -2/3`.
- Rung injection:
  `|0> -> |vac>` and
  `|1> -> (c_Aup^dag c_Bdown^dag - c_Adown^dag c_Bup^dag) |vac> / sqrt(2)`.
- Target ramp:
  `(t/J, U/J) = (0, 0) -> (0.7, 7)` with `J = 1`, so target
  `t/J = 0.7`, `U/t = 10`.
- Trotter settings: `M = 1, 2` steps, `tau = 0.4`.
- Observables:
  rung-rung singlet pairing
  `P_r(x, y) = (1/N) sum_i <Delta_AiBi Delta_AjBj^dag + h.c.>`,
  optional slanted-pair correlations, and target-Hamiltonian energy density.

Example target:

- `bilayer_rung_pairing.py`
- Represent bilayer sites as `(layer, x, y)` or flatten them with a reversible
  map stored in the output metadata.
- Add bilayer Hamiltonian term construction for intra-layer hopping, onsite
  `U`, and inter-layer exchange.
- Validate on 2 x 2 x 2 or 3 x 3 x 2 before running the paper's 4 x 4 x 2
  setting.

## Implementation Phases

1. Infrastructure and metadata
   - Add a small `settings.py` with lattice maps, boundary-condition helpers,
     output paths, and deterministic seeds.
   - Add `observables.py` for number, spin, onsite eta-pair, bond-singlet, and
     rung-singlet observables.
   - Add `hamiltonians.py` for weighted Fermi-Hubbard and bilayer exchange
     term dictionaries that can be wrapped in `pepsy.SymHamiltonian`.

2. Small exact checks
   - 2 x 2 spinful Hubbard plaquette at `U/t = 2`; verify `|s>` and `|d>`.
   - 4 x 4 or smaller free-fermion imbalance at `U = 0`.
   - 2 x 2 x 2 bilayer rung-singlet injection and exchange energy.
   - Use dense NumPy/quimb exact diagonalization only at these small sizes.

3. Symmray tensor-network checks
   - Recreate the same small systems with `SymMPS` or `SymPEPS`.
   - Assert that tensors and gates remain Symmray/FermionicArray-backed after
     gate application.
   - Track `overall_charge`, `max_bond`, block density, norm, and energy.

4. Paper-size exploratory runs
   - Half-filled 6 x 6: PEPS bond dimensions such as `D = 2, 4, 6` and
     boundary `chi` sweeps.
   - Checkerboard 6 x 6: weak-coupling product/plaquette state first, then
     one Trotter ramp step.
   - Bilayer 4 x 4 x 2: start with MPS snake-order references, then compare to
     PEPS or layered PEPS if Pepsy supports the needed geometry cleanly.

5. Result comparison
   - Produce tables with paper reference values beside Pepsy/Symmray values.
   - Plot or save `P_eta`, `P_b`, and `P_r` versus displacement.
   - Report truncation sensitivity by `D`, `chi`, and cutoff, not just one
     number.

## Pepsy Gaps To Resolve Before Full Reproduction

- Fermionic graph metadata: keep site ordering, edge orientation, graph
  distance, and any local/global ordering convention explicit in outputs and
  helpers, especially for non-square, bilayer, and snake-MPS embeddings.
- Weighted Hubbard edges: the checkerboard model needs per-edge `t` values,
  and the light pulse needs time-dependent Peierls phases.
- Bilayer exchange: the current uniform Fermi-Hubbard builder is not enough for
  `S_iA . S_iB - n_iA n_iB / 4`; add a local exchange-term builder.
- Pairing observables: add tested dense-to-Symmray constructors for charge-zero
  two-site/four-site pair correlators, including fermionic signs.
- Geometry: decide whether bilayer examples should use a flattened MPS,
  a custom PEPS-like graph through Symmray-from-edges, or a Pepsy lattice
  wrapper extension.
- Boundary conditions: implement periodic/anti-periodic sign choices and
  mixed-periodic averaging explicitly in metadata.

## Validation Checklist

- Run focused API and Symmray checks:
  `pytest -q tests/test_symmetric_tensors.py tests/test_public_api.py`
- For new Hamiltonian/observable helpers, add example-local unit tests or
  promote them to `tests/` if they become package APIs.
- For each example, write a small mode that finishes quickly:
  `--size smoke`, `--bond-dim 2`, `--chi 8`, and fixed seeds.
- Store generated numeric outputs under an ignored results directory, not in
  the notebook or source tree.
- Document any mismatch from the paper caused by boundary conditions,
  finite-bond truncation, or using native fermionic TN instead of the
  hardware compact encoding.

## Suggested File Layout

```text
examples/pepsy_examples/
  plan.md
  Fermi_Hubbard/
    README.md
    half_filled_4x4_direct_fermions.ipynb
  fermihubbard_symmray/  # future script/module form
    README.md
    settings.py
    hamiltonians.py
    observables.py
    half_filled_eta_pairing.py
    checkerboard_dwave_pairing.py
    bilayer_rung_pairing.py
    tests/
      test_small_hubbard_checks.py
      test_observables.py
```

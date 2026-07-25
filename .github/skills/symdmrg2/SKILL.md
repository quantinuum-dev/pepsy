---
name: symdmrg2
description: 'Run, debug, tune, extend, or validate SymDMRG2 — pepsy''s symmetric (symmray block-sparse) two-site DMRG ground-state solver for MPS. Use when the user asks to find a ground state / run DMRG with a conserved symmetry (U1, U1U1, Z2) in pepsy; build a symmetric MPO via SymHamiltonian.to_mpo; pick or debug the initial state (product ramp vs random), bond-dimension schedule/ramp, or the mixer (density_matrix vs subspace); compare pepsy DMRG against TeNPy or exact diagonalization; run Fermi-Hubbard (U1U1), spinless Fermi-Hubbard / t-V (U1), Heisenberg (U1), or transverse-field Ising (Z2); diagnose non-monotonic energy vs chi, a stuck/locked DMRG, or a plateau above the reference; or add a new symmetric model. Also use for questions about SymDMRG2 conventions, snake 2D->1D mapping, fermionic MPO Jordan-Wigner signs, or the tenpy_vs_pepsy_dmrg benchmark workspace.'
argument-hint: 'e.g. "run Heisenberg U1 DMRG and check vs ED" or "why is my chi=32 run stuck above TeNPy?"'
---

# SymDMRG2 (symmetric block-sparse DMRG) in pepsy

`pepsy.SymDMRG2` is a two-site DMRG ground-state solver that runs on **symmray
block-sparse arrays** (native symmetry: `U1`, `U1U1`, `Z2`), driven by a symmetric
MPO from `SymHamiltonian.to_mpo`. It is the symmetric analog of `quimb`'s `DMRG2`.

Use the **Python 3.12 env** (`~/envs/py312/bin/python`); the system python has an
old `symmray` that crashes `to_mpo` (`build_local_fermionic_dense` missing).

## When to use
- Ground state of a lattice model with a conserved charge, as an MPS, in pepsy.
- Build/validate a symmetric MPO (`SymHamiltonian.from_edges(...).to_mpo(...)`).
- Choose/debug **initial state**, **bond-dim ramp**, or **mixer**.
- Compare pepsy DMRG vs **TeNPy** or **exact diagonalization**.
- Diagnose a stuck/locked run, non-monotonic energy vs `chi`, or a plateau.

## Do NOT use for
- Non-symmetric / dense MPS DMRG → use `quimb.tensor.DMRG2` or `pepsy.MpoOptimizer`.
- Imaginary-time / gate evolution of an MPS → `SymMPS.time_evolve_mps_optimizer` /
  `pepsy.MpsOptimizer` (`mode="mpo"`), not DMRG.
- PEPS/2D variational optimization → `pepsy.PepsOptimizer` / boundary tools.

## Quickstart (minimal, correct pattern)
```python
import pepsy
from pepsy.tensors import SymHamiltonian, SymMPS, site_charge_from_occupations

L = 8
# Product initial state (bond 1) in a fixed charge sector (Sz=0 half filling):
state = SymMPS.for_model(
    "heisenberg", L, bond_dim=1,
    site_charge=site_charge_from_occupations([1, 0] * (L // 2)),
    seed=7, dtype="complex128",
)
ham = SymHamiltonian.from_edges("heisenberg", "U1", [(i, i + 1) for i in range(L - 1)])
mpo = ham.to_mpo(L=L, compress=False)

opt = pepsy.SymDMRG2(
    mpo, state,
    bond_dims=[1, 2, 4, 8, 16, 32],   # gentle ramp (or "auto"/"product_ramp")
    cutoffs=[1e-10],                  # independent, repeat-last
    local_solver="auto", which="SA",
    mixer="density_matrix",           # White/Hubig DM mixer (best)
    compute_initial_energy=False,
)
opt.solve(max_sweeps=30, sweep_sequence="RL", tol=1e-11)
E = float(opt.energy)          # ground energy; opt.state = solved SymMPS
assert opt.backend == "symmray"
```

## Supported models (`SymHamiltonian` registry)
| model string | symmetry | fermionic | phys_dim | notes |
|---|---|---|---|---|
| `heisenberg` (`heis`) | U1 | no | 2 | isotropic S·S (singlet −3/4) |
| `tfim` (`ising`,`itf`) | Z2 | no | 2 | params `jx`, `hz` |
| `fermi_hubbard_u1u1` (`fh_u1u1`) | U1U1 | yes | 4 | per-spin N; params `t`,`U`,`mu`,`V` |
| `fermi_hubbard_spinless` (`tv`,`t-v`) | U1 | yes | 2 | t-V; params `t`,`V`,`mu` |
| `fermi_hubbard` (`fh`) | U1 | yes | 4 | **`to_mpo` NOT implemented — use `fermi_hubbard_u1u1`** |
| **t-J** | — | — | — | **NOT in registry**; symmray has no `ham_t_j`. Needs a new builder. |

`to_mpo` dispatch: `fermi_hubbard_spinless` → fermionic MPO; `fermi_hubbard_u1u1`
→ specialized JW U1U1 path; everything else (spin models) → generic bosonic path
(`_generic_symhamiltonian_to_mpo`).

## Building blocks
- **Hamiltonian**: `SymHamiltonian.from_edges(model, symmetry, edges, **params)`.
  `edges` are lattice edges (1D pairs, or 2D coord edges). `ham.terms` = dict
  `{edge: 2-site array}` (a complete sum-of-2-site-ops representation of H).
- **MPO**: `ham.to_mpo(L=, mapper=OneDMap(...), compress=, cutoff=)`. For 2D pass a
  `mapper`; it snake-maps coord edges to chain indices and emits long-range terms
  (with fermionic parity strings on the fermionic path).
- **Initial state**: `SymMPS.for_model(model, L, bond_dim=, site_charge=, seed=, dtype=)`
  builds a raw random MPS in the given charge sector; `bond_dim=1` gives a
  **product** state, `bond_dim=chi` gives a full-`chi` **block-fill** (random-rich)
  state. For a well-conditioned **random-unitary** state use
  `SymMPS.random_unitary_for_model(model, L, bond_dim=, site_charge=, seed=, dtype=)`.
  `site_charge=site_charge_from_occupations([...])` fixes the charge sector (its
  length must equal `L`). For odd L, Sz=0 is impossible — use the minimal sector.
- **2D→1D**: `idx2coo, coo2idx = OneDMap(Lx, Ly, mode="snake").build()`
  (also `"folded-snake"` to shorten the worst torus edge).

## Initialization — the critical lesson (verified, decisive)
- **Default to product-grown**: a **product** state (`for_model(..., bond_dim=1)`) +
  a **gentle bond ramp** (`bond_dims="auto"`/`"product_ramp"`, or an explicit
  `[1,2,4,8,16,...,chi]`). This is monotonic in `chi` and matches/beats TeNPy at
  matched `chi`.
- **Full-`chi` block-fill (`for_model(..., bond_dim=chi)`, random-rich) LOCKS** into
  a poor basin. This is **fundamental, not sweep-limited**: doubling sweeps (30→60)
  moved the energy by ~2e-6 while it sat ~0.04/site above the product-grown result;
  energy can even get **worse at larger `chi`** (deeper wrong basin). Mechanism:
  block-fill commits bond dimension to the wrong charge sectors with a near-flat
  Schmidt spectrum; two-site DMRG only re-allocates sectors locally and cannot escape.
- For a **random warm start**, use `SymMPS.random_unitary_for_model(...)`: a
  charge-conserving, canonical, well-conditioned random MPS that DMRG escapes
  cleanly (TeNPy-analog). Never use raw full-`chi` block-fill for hard cases.
- Cascaded χ warm-restart (converge χ=8 → feed as init for χ=16 → χ=32) also works
  and is slightly better than a single-run ramp (fuller convergence per stage), at
  higher wall cost. Reuse `opt.state.copy()` as the next `SymDMRG2` init.

## Mixer
- `mixer="density_matrix"` (aliases `dm`) — White/Hubig **density-matrix mixer**:
  perturbs the reduced DM with the sweep-toward projector
  (`left_projector = left_env·w_left` = LHeff / `right_projector = w_right·right_env`
  = RHeff) with the **MPO virtual bond kept open**, unions sectors, eigh-splits,
  and preserves `theta`. This is the TeNPy-style one-sided expansion and the **best**
  mixer — beats `subspace` at matched `chi`.
- `mixer="subspace"` (`subspace_expansion`) — older Hubig/DMRG3S drive
  (`H_eff·theta` stacked on the MPS bond); weaker.
- `mixer="none"` — fine for easy/exact small cases; 2-site SVD still grows sectors.
- Controls: `mixer_amplitude`, `mixer_decay`, `mixer_disable_after` (turn off in late
  sweeps). Mixer alone is **not** the fix for random-init locking — initialization is.

## Other key options
- `bond_dims`: list schedule, or an auto-ramp string (`"auto"`, `"ramp"`,
  `"product_ramp"`). `bond_dims` and `cutoffs` are **independent repeat-last**
  sequences (lengths need not match). A ramp needs enough `max_sweeps` to climb it.
- `local_solver="auto"`, `local_eig_ncv` (Krylov cap; int or per-sweep list),
  `local_eig_tol`, `local_eig_energy_tol` (Ritz energy-stop; `~1e-2` is fast and
  usually enough, avg matvecs ~3–4 when healthy; ~15 signals ill-conditioning).
- `which="SA"` (smallest algebraic). `sweep_sequence="RL"`. `tol` = energy-change stop.
- `variational_sector_basis` ∈ `"adaptive"|"bond_dim"|"off"`; `sector_enrichment`.
- Read-out: `opt.energy`, `opt.state`, `opt.sweeps`, `opt.backend`, `opt.profile_summary()`.

## Validation recipes (how to check "right results")
- **Spin models (bosonic)**: densify each `ham.terms` 2-site array with
  `term.to_dense()` (shape `(2,2,2,2)=(oi,oj,ii,ij)`), embed on linear sites, sum,
  and `np.linalg.eigvalsh`. This is convention-exact (same terms the MPO encodes).
  For Z2/tfim, compare against the **even-parity** sector min (SymDMRG2 init charge 0).
- **Fermions**: the densify+kron trick does NOT apply (symmray handles fermionic
  swap phases during contraction). Build a **Jordan-Wigner** dense ED (Z-strings) and
  restrict to the fixed particle-number sector. See `_dense_jw_fermi_hubbard` /
  the retained fermionic regression in `tests/test_symmetric_tensors.py`.
- **vs TeNPy** (py312 has TeNPy 1.1.0): `SpinChain` (Heisenberg, `conserve="Sz"`),
  `FermionChain` (spinless, `H=Σ -J(c†c+h.c.)+V n n − μ n`, `conserve="N"`),
  `hubbard`, `tf_ising`. Drive:
  `dmrg.TwoSiteDMRGEngine(psi, model, {"trunc_params":{"chi_max":,"svd_min":}, "mixer":True, "combine":True}).run() -> (E, psi)`;
  `psi = MPS.from_lat_product_state(model.lat, [...])`.
- Validated to ~1e-15: Heisenberg U1 (1D/2D vs ED, vs TeNPy), tfim Z2 (vs ED),
  spinless FH t-V U1 (vs ED **and** TeNPy). Regression tests live in
  `tests/test_symmetric_tensors.py` (`test_symdmrg_fermionic_state_accepts_raw_fermion_constructor_output`) and assert genuine
  symmetric usage (`opt.backend=="symmray"`, block-sparse multi-sector MPO/MPS).

## Assert it is genuinely symmetric (not a dense fallback)
Every MPO tensor, the input MPS, and the solved MPS must be symmray arrays whose
type name starts with the symmetry (`U1…`/`Z2…`) **and carry >1 charge block**;
`opt.backend == "symmray"`. `SymMPS` is not subscriptable — get tensors via
`list(getattr(mps, "tn", mps))`. See `_assert_symmetric_pipeline` in the tests.

## Gotchas
- `fermi_hubbard` (total-U1 spinful) `to_mpo` raises NotImplementedError → use
  `fermi_hubbard_u1u1`.
- Fermionic MPO JW signs were a real bug (now fixed): symmray is native-fermion
  (Grassmann / Convention B); do not mix JW strings with symmray swap phases. If
  `<H>` via the MPO ≠ via `ham.terms`, suspect the fermionic MPO path.
- A `converged=False` flag with a flat energy across sweeps means "stuck in a basin"
  (tiny local improvements never hit `tol`), not "needs more sweeps".

## Benchmark workspace
`pepsy_examples/tenpy_vs_pepsy_dmrg/` is the pepsy-vs-TeNPy harness: `cases/`
(fh_3x2 … fh_7x6, exact-ED bridges + paper settings), `runners/`
(`run_pepsy_symdmrg2.py`, `run_tenpy_reference.py`), `tools/` (`scan_chi.py`,
`run_6x6_paper_compare.py`, `guard.py`). Fair defaults there: product-ramp init +
`density_matrix` mixer; random-rich is an explicit opt-in stress control.

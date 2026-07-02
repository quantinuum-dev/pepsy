# PLAN.md — pepsy roadmap

Status: living document
Last updated: 2026-07-02
Owners: pepsy maintainers + coding agents

This document tracks the planned workstreams for `pepsy` (boundary-MPS tools for
PEPS norm contraction and DMRG fitting). It is the "what/when" companion to the
conceptual notes in `learning/` (the "why/how"), the session journal in
`history/` (the "what happened"), and the published documentation in `docs/`.

Current package version: `0.2.0` (`pepsy.__version__` / `pyproject.toml`).

Four headline workstreams drive the roadmap:

1. **Belief propagation** contraction (+ loop / cluster expansion corrections).
2. **quimb integration** — lean on `quimb` / `cotengra` / `autoray` rather than
   reimplementing tensor-network primitives.
3. **Symmetric & fermionic tensors** via `symmray`.
4. **Variational Monte Carlo** for PEPS: current NetKet/JAX bridge plus a
   later torch-only pure-PEPS path for ITF, Fermi-Hubbard, Heisenberg, etc.

The conceptual background for each lives in `learning/bp.md`,
`learning/quimb.md`, and `learning/symmray.md`.

---

## 1. Belief-propagation contraction subpackage

### Motivation

pepsy's exact-ish contraction route today is boundary-MPS
(`build_bra_ket -> BdyMPS -> contract_boundary`). Belief propagation (BP) offers
a cheap, gauge-providing alternative that is exact on trees and a strong
approximation when loop correlations are weak (BP ≈ simple update; BP gauges a
TN). It complements the boundary-MPS path, especially at bond dimensions where
boundary-MPS / CTMRG become impractical.

### Plan

- Add a new subpackage **peer to `boundary/`** (e.g. `pepsy.contraction` /
  `pepsy.bp`) with modules like `bp.py`, `loops.py`, `expansion.py`,
  `environments.py`. BP does **not** live inside `boundary/`.
- Reuse `pepsy.build_bra_ket(ket, bra?)` to form the closed double-layer network
  (existing `KET`/`BRA` and `X*/Y*/I*` tags); BP operates on that same object.
- Return a `BPResult` dataclass mirroring `BoundaryContractResult`
  (`.cost`, `.fidel`-style fields) so downstream code treats BP and boundary-MPS
  uniformly.
- Expose fixed-point messages as a canonical/Vidal-like gauge usable as an
  initializer for the existing optimizers.

### Validation

- Compare the BP norm against `pepsy.contract_boundary` at large `chi` on a
  3×3 PEPS (numerically-exact reference).
- Document failure modes: no / non-unique / near-degenerate fixed points (e.g.
  GHZ-like PEPS), which break loop-series convergence.

### References

- Concept note: `learning/bp.md`.
- Alkabetz & Arad (2021); Tindall & Fishman (2023).

---

## 2. Loop / cluster expansion corrections

### Motivation

Raw BP is *uncontrolled* — you cannot systematically improve it by growing a
bond dimension. The loop series / loop cluster expansion turns BP into a
controlled approximation: resolve each edge as `I = P + Q` (rank-1 message
projector plus complement), keep only closed-loop excitations, and exploit
exponential suppression of high-degree loops.

### Plan

- Build on the §1 BP fixed point: enumerate low-degree connected loop clusters,
  normalize per-tensor BP-vacuum contributions to 1, and sum connected clusters.
- Deliverables mirroring the source papers: corrections to the free-energy
  density, the transfer matrix, and the 2-site density matrix.
- Prune and cap loop enumeration on square lattices (loop count grows fast with
  degree); reuse `cotengra` for the sub-network contractions.

### Validation

- Show the dominant (genus-1) loop correction improves on raw BP by orders of
  magnitude toward the large-`chi` boundary-MPS reference.

### References

- Concept note: `learning/bp.md` (loop / cluster expansion sections).
- Evenbly et al., *Loop series expansions for tensor networks*,
  Phys. Rev. Research 8, 013245 (2026), DOI `10.1103/vqks-cr6x`.
- Gray et al., *Tensor Network Loop Cluster Expansions for Quantum Many-Body
  Problems*, arXiv:2510.05647.

---

## 3. quimb integration

### Motivation

`quimb` (with `cotengra`, `cotengrust`, `autoray`) already provides mature
tensor-network, contraction-planning, and backend-dispatch machinery. pepsy
should stay a thin layer of pepsy conventions and stable public APIs over these
upstream libraries rather than reimplementing primitives.

### Plan

- **Audit `quimb.tensor` BP + gauging API** and prefer wrapping it over
  hand-rolling the §1/§2 machinery; record the "pepsy concept → quimb API" map
  in `learning/quimb.md`. (The sampling subpackage already wraps quimb's
  `sample_d2bp` in `PepsBpSampler`; extend that pattern.)
- Keep contraction path/tree optimization on `cotengra` (via
  `build_optimizer` / `build_compressed_optimizer`); forward cotengra-compatible
  options instead of inventing parallel path-search config.
- Keep backend inference and backend-agnostic array ops on `autoray`; preserve
  NumPy / Torch / JAX / CuPy compatibility.

### Open decision

- Exact pepsy-vs-quimb division of labor for BP is deferred to the BP prototype
  (see §8). Prototype first, then decide wrap-vs-own per component.

### References

- Concept note: `learning/quimb.md`.

---

## 4. Symmetric & fermionic tensors via symmray

### Motivation

`symmray` provides block-sparse abelian-symmetric and fermionic arrays that are
`autoray`-compatible and drop into `quimb.tensor` objects — the natural way to
give pepsy PEPS/MPS workflows U(1)/Z2 symmetry and fermionic statistics without
rewriting the algorithms. Groundwork already exists: `SymMPS` / `SymPEPS`
wrappers and `model="fermi_hubbard"` (U1) / `"fermi_hubbard_u1u1"` (U1U1)
constructions, plus the fermionic Fermi-Hubbard starters in
`../pepsy_examples/fermi_hubbard/`.

### Plan

- Keep `symmray` an **optional** dependency (`[project.optional-dependencies]
  vmc = [..., "symmray"]`); gate tests with `pytest.importorskip("symmray")`.
- **Backend recognition:** `pepsy.backends` must treat symmray arrays as a valid
  autoray backend and **never silently densify** block structure; route
  decompositions to symmray's linalg (`svd_truncated`, `qr_stabilized`, …).
- **Tag bridge:** map pepsy leg conventions onto symmray `dual` — `k…` (ket,
  outward) ↔ `dual=False`, `b…` (bra, inward) ↔ `dual=True`.
- **Conjugation phases (fermions):** when a network has both bra- and ket-like
  dangling legs (norms, cluster/infinite settings), explicitly phase-flip the
  conjugated network's dangling dual legs — the main correctness pitfall.

### Validation

- Abelian (`Z2` / `U1`) PEPS norm `⟨ψ|ψ⟩` matches a dense contraction on a tiny
  lattice.
- Fermionic norm and a small Fermi-Hubbard energy match ED on a 2×2 patch,
  exercising `label` / `dummy_modes` phase tracking and the conjugation
  phase-flip.

### References

- Concept note: `learning/symmray.md`.
- Gao et al., *Fermionic tensor network contraction for arbitrary geometries*,
  Phys. Rev. Research 7, 023193 (2025), DOI `10.1103/PhysRevResearch.7.023193`.

---

## 5. Variational Monte Carlo (fermionic PEPS + NetKet)

### Motivation

Sampled VMC is the scalable route to fermionic PEPS energies where exactly
summing the full contraction is infeasible. `pepsy.vmc` bridges Symmray
fermionic PEPS amplitudes (flat backend, `jax`-jittable) to NetKet's
Hilbert / operator / sampler / VMC machinery, keeping NetKet, JAX, Flax, and
Symmray optional (the `vmc` extra), so the core package stays lightweight.

### Current progress (validated)

- `pepsy.vmc` subpackage exists with a lazy, optional-dependency-friendly
  `__init__` (concrete integrations in `pepsy.vmc.netket`).
- Public API (all lazy): `configure_jax_for_vmc`, `square_lattice_edges`,
  `netket_spin_orbital_columns` / `verify_netket_spin_columns`,
  `occupation_to_phys_indices`, `pack_fermionic_peps_ansatz`,
  `make_fermionic_peps_log_amplitude_model`,
  `make_fermionic_peps_batched_amplitude_function`,
  `build_fermi_hubbard_vmc`, `make_netket_vmc_driver`,
  `make_netket_sr_preconditioner`, `make_netket_autochunk_callback`,
  `recommend_netket_vmc_settings`, `choose_netket_chunk_size`, plus the
  `SpinOrbitalColumns` / `PackedFermionicPEPS` / `NetKetVMCSettings` /
  `NetKetChunkSettings` / `NetKetFermiHubbardVMC` dataclasses.
- Validated end-to-end in
  `../pepsy_examples/fermi_hubbard/fermi_hubbard_vmc.ipynb` (2×2, `t=1`,
  `U=8`, `D=4`, `Z2`, half filling):
  - the Flax log-amplitude model matches a direct Symmray `isel(...).contract`
    (rtol 1e-7);
  - dense-sector energy is real and variational (`>= E_gs`);
  - an exact-sum `optax` optimizer reaches ED (`<H>/N - U/4 = -2.33006`);
  - NetKet sampled VMC (`MetropolisFermionHop`, `MCState`, optional SR)
    converges near ED.
- Pinned physics/layout facts: `phys_dim=4` `Z2` fold
  `phys = 2*(n_up != n_dn) + n_dn`; NetKet `SpinOrbitalFermions` orders the
  spin-down block before the spin-up block; the flat Symmray backend needs
  `einops` and (for `Z2` `subsizes="equal"`) an **even** `D`.
- Package tests: `tests/test_vmc_netket.py` (gated on optional deps).

### Plan (next Pepsy cuts)

1. `to_peps(variables)` / `update_peps_from_variables(...)` to convert an
   optimized NetKet state back into a `SymPEPS`.
2. `benchmark_log_amplitude_batch(...)` timing batch / chunk sizes and
   `exact` vs `hotrg` without a full VMC optimization.
3. `chi_sweep_energy(...)` for HOTRG `chi` convergence of sampled energies.
4. A GPU example script/notebook that measures after warmup and reports device,
   memory, batch size, and compile time separately.
5. `U1U1` and odd-parity `Z2` setup helpers so 3×3 / odd half-filled systems
   are not awkward.
6. Scalable SR: diagonal / iterative / minSR-style paths before dense SR on
   large PEPS parameter counts.
7. Keep CI tiny and CPU-only; large GPU timing stays in examples.

Development rule of thumb (from the notebook): reusable NetKet/Symmray glue
moves into `pepsy.vmc`; physics sanity checks and timing experiments stay in the
example notebook.

### Later torch-only pure-PEPS VMC

Add a torch-only VMC path focused on **plain PEPS/TNS amplitudes and standard
Hamiltonians**, not neural-network ansatz layers. Use `sjdu10/vmc_torch` and the
NN-fTNS paper as implementation references for sampling, local-energy, SR, and
boundary-reuse mechanics, but keep Pepsy's scope here to pure PEPS:

- In scope: torch PEPS amplitude wrappers, batched amplitude evaluation,
  boundary-MPS / CTMRG / HOTRG contraction choices, cached boundary reuse,
  Metropolis samplers, local-energy kernels, and optimizer/preconditioner
  utilities for tensor parameters.
- Models in scope first: transverse-field Ising (ITF/TFI), Heisenberg, and
  spinful Fermi-Hubbard on square/open lattices; add chain variants only where
  they reuse the same graph/Hamiltonian machinery.
- Sampling options: spin exchange for fixed-`S_z` Heisenberg, spin flips for
  ITF, and spinful exchange/hopping proposals for Fermi-Hubbard preserving
  `N_up`/`N_down`.
- Amplitude choices: pure quimb/Symmray PEPS packed as torch parameters, with
  optional approximate contraction (`boundary`, `ctmrg`, `hotrg`) and
  `torch.vmap`/chunking where upstream operations support it.
- Reuse strategy: cache row/column boundary environments and invalidate only
  affected rows/columns after local moves; benchmark no-reuse vs reuse at
  increasing `L`, `D`, `chi`.
- Explicitly out of scope for Pepsy until requested: Transformer/CNN/MLP
  backflow, NN-fTNS tensor corrections, neural Jastrow, Slater backflow, LoRA,
  or other neural-network wavefunction layers.

Current status:

- Dense quimb PEPS can be packed as torch parameters with exact, boundary-MPS,
  CTMRG, or HOTRG contraction choices.
- Torch samplers and connected-configuration local-energy kernels cover ITF,
  Heisenberg, and spinful Fermi-Hubbard.
- Direct SR and sample-space minSR utilities are available for real-valued
  torch amplitudes; minSR is the intended path when `n_params >> n_samples`.
- Large `L,D` runs are still not efficient until boundary-MPS environment reuse
  is implemented around local PEPS updates and connected configurations.
- True torch Symmray/block-sparse fermionic PEPS is not yet validated. The
  current torch wrapper is tested on dense quimb PEPS and rejects Symmray arrays
  until a dedicated torch `to_pytree`/`from_pytree` adapter is added.

Implementation order:

1. Keep the existing `pepsy.vmc.torch` sampler/local-energy kernels minimal and
   backend-agnostic; harden tests against tiny ED references for ITF,
   Heisenberg, and 2x2 Fermi-Hubbard.
2. Add a pure torch PEPS amplitude wrapper with `forward(config_rows)` and
   `forward_log(config_rows)` matching the sampler interface.
3. Add SR/minSR optimizer helpers for packed PEPS tensor parameters.
4. Add local-energy drivers that combine `*_connections(...)`,
   `local_energy_from_connections(...)`, and a PEPS amplitude model.
5. Add boundary-reuse PEPS amplitude evaluation after the direct/batched path is
   correct.
6. Add a small example/notebook comparing exact dense energy, plain PEPS VMC,
   and contraction-reuse timing.

### Validation

- Package: `pytest -q tests/test_vmc_netket.py tests/test_public_api.py`
  (VMC deps guarded with `pytest.importorskip`).
- Torch-only VMC: `pytest -q tests/test_vmc_torch.py` plus ED checks for tiny
  ITF, Heisenberg, and Fermi-Hubbard systems as the pure PEPS wrapper lands.
- Downstream: run
  `../pepsy_examples/fermi_hubbard/fermi_hubbard_vmc.ipynb` (the default 2×2 run
  is fast and cross-checks against ED).

### References

- Concept note: `learning/symmray.md` (fermionic amplitudes).
- Gao et al., *Fermionic tensor network contraction for arbitrary geometries*,
  Phys. Rev. Research 7, 023193 (2025), DOI `10.1103/PhysRevResearch.7.023193`.
- `sjdu10/vmc_torch`: <https://github.com/sjdu10/vmc_torch> (torch VMC
  sampling/local-energy/SR and PEPS reuse reference).
- Du, Chen, Chan, *Neuralized Fermionic Tensor Networks for Quantum Many-Body
  Systems*: <https://arxiv.org/pdf/2506.08329> (use mechanics as reference, but
  do not add NN-fTNS/neural ansatz scope to Pepsy unless explicitly requested).
- Downstream plan: `../pepsy_examples/fermi_hubbard/PLAN.md`.

---

## 6. Public API & docs hygiene (cross-cutting)

New public symbols must be threaded consistently:

- Update the owning subpackage `__all__`, the lazy top-level maps in
  `src/pepsy/__init__.py` (`_SYMBOL_MODULES`, `_MODULE_EXPORTS`, `__all__`),
  `docs/api/`, and `tests/test_public_api.py`.
- Do not reintroduce removed flat modules (`pepsy.core`, `pepsy.gates`,
  `pepsy.sampler`, `pepsy.optimize_sweep`); `tests/test_package_layout.py`
  guards this.
- Promote finalized, user-facing designs from `learning/` into `docs/`
  (tutorials / how-to).

---

## 7. Downstream examples

- `../pepsy_examples/` is the downstream validation surface; keep its notebooks
  on current public namespaces (see `../pepsy_examples/PLAN.md`).
- As BP / loop-series and expanded symmetric-TN support land here, add matching
  example notebooks there (BP / loop-series contraction demo, quimb-backed
  gauging demo, broader `symmetric_tensors/` coverage).

---

## 8. Open questions / decisions

- Exact pepsy-vs-quimb division of labor for BP (wrap vs. own per component) —
  resolve during the BP prototype (§1, §3).
- `symmray` version to pin (currently tracking v0.2.x).
- Whether loop-series corrections expose free-energy, transfer-matrix, and
  2-site-density-matrix outputs behind one `BPResult` or separate result types.

---

## 9. Backlog

- BP-gauge initializer wired into the existing sweep / global / PEPS optimizers.
- Shared reporting for BP-vs-boundary-MPS accuracy/cost traces.
- Fermionic optimizer path (gradient-based) once §4 norms/energies validate.

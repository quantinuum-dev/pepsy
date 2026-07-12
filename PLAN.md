# PLAN.md — pepsy roadmap

Status: living document
Last updated: 2026-07-11
Owners: pepsy maintainers + coding agents

This document tracks the planned workstreams for `pepsy` (boundary-MPS tools for
PEPS norm contraction and DMRG fitting). It is the "what/when" companion to the
conceptual notes in `learning/` (the "why/how"), the session journal in
`history/` (the "what happened"), and the published documentation in `docs/`.

Current package version: `0.2.0` (`pepsy.__version__` / `pyproject.toml`).

Four headline workstreams drive the roadmap:

1. **Belief propagation** contraction (+ loop / cluster expansion corrections,
   + disordered-memory / relay-BP convergence robustness).
2. **quimb integration** — lean on `quimb` / `cotengra` / `autoray` rather than
   reimplementing tensor-network primitives.
3. **Symmetric & fermionic tensors** via `symmray`.
4. **Variational Monte Carlo** for PEPS: current NetKet/JAX bridge plus a
   later torch-only pure-PEPS path for ITF, Fermi-Hubbard, Heisenberg, etc.

The conceptual background for each lives in `learning/bp.md`,
`learning/quimb.md`, `learning/symmray.md`, and
`learning/fermionic_mpo.md`.

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

### Convergence robustness — disordered-memory / relay-BP

Raw BP (§1) and the loop expansion (§2) both need a **converged, normalized BP
fixed point**. On frustrated / degenerate networks — and on QEC decoding graphs
in particular — BP often fails to converge: symmetric messages oscillate on
short-cycle trapping sets. Plan a convergence-robust BP mode in `pepsy.bp`:

- Start from quimb's **1-norm** BP (`L1BP` / `HV1BP` / `D1BP`) for the
  partition-function-style (nonnegative) contractions that decoding needs,
  wrapped per §3 rather than hand-rolled.
- Add a **disordered-memory / relay-BP** extension (Müller et al.,
  arXiv:2506.01779): per-variable random memory strengths `gamma_i` (including
  negative “anti-memory”) that break the symmetric fixed points, plus **relay**
  legs (warm-start + re-randomized disorder, keep the best result). quimb's
  `damping` is a *uniform* `(old, new)` callable, so the per-node disorder needs
  a small `BeliefPropagationCommon` subclass overriding the message update
  (~tens of lines); the relay outer loop reuses `messages=` warm-start +
  `run(info=...)` convergence flags.
- Return the standard `BPResult` gauge/messages so a converged fixed point is
  available to **feed §2 loop corrections** (which require a normalized fixed
  point) and to gauge the optimizers. Track the per-run message residual as a
  first-class convergence signal.

**Prototyped (2026-07-11):** `pepsy.bp.relay_bp` / `one_norm_bp` (new
`src/pepsy/bp/` subpackage) wrap quimb's 1-norm BP (`L1BP` / `HV1BP` / `D1BP`)
and apply per-node disordered memory around quimb's `iterate` on the public
`messages` dict (quimb's `damping` is uniform, so the per-node strength is
applied in a thin driver), with relayed warm-started legs returning the
best-converged fixed point as a `RelayBPResult`. Kept out of the lazy top-level
namespace (`import pepsy.bp`) while it is a prototype; tests in
`tests/test_bp_relay.py`. Next: recompute the post-mix residual for the
convergence check, expose the fixed point as a `BPResult` gauge, and wire it to
the §2 loop corrections.

#### Relationship to tensy

tensy already ships a **standalone classical** relay-BP on the DEM Tanner graph
(`tensy.decoders.RelayBpDecoder`, NumPy normalized min-sum) as the QEC-decoder
baseline. The pepsy work is the **tensor-network** generalization: relay-BP on
quimb TN messages plus loop corrections, exposed as a clean pepsy public API
that tensy's Phase-E TN-BP / tensor-network-message-passing decoder consumes.
Keep decoder-specific glue in tensy and the TN-BP machinery in pepsy.

#### References

- Müller et al., *Improved belief propagation is sufficient for real-time
  decoding of quantum memory*, arXiv:2506.01779 (relay-BP / disordered memory).
- Wang et al., *Tensor Network Message Passing*, Phys. Rev. Lett. 132, 117401
  (2024), arXiv:2305.01874 (exact short-loop clusters + BP messages).

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

**Prototyped (2026-07-11):** `pepsy.bp.loop_cluster_expand` / `LoopClusterResult`
(`src/pepsy/bp/cluster.py`) wrap quimb's generalized-loop **cluster** expansion
(`D2BP` / `D1BP`.`contract_gloop_expand`), with the product (Eq. 2) and sum
(Eq. 1) formulas, message reuse (`.expand(gloops)`), and a `norm` toggle
(`"2norm"` = `D2BP` wavefunction norm, `"1norm"` = `D1BP` scalar TN). Validated in
`tests/test_bp_cluster.py`: on a 3×3 `D=2` PEPS the error falls 22 % (BP) →
machine-exact as the cluster covers the lattice, and — the key point of
arXiv:2510.05647 — at a system-covering cluster the estimate is **bit-identical**
for converged vs unconverged BP messages, i.e. the cluster expansion does not
need a BP fixed point for correctness. It is more convergence-robust than the
loop *series* expansion (whose `I − m⊗m` projectors rely on the fixed point).
Next: a PEPS local-observable / energy helper (product & sum formulas) with
Wynn-ε extrapolation.

### Cluster-corrected bond compression (better than naive BP messages)

BP-gauge compression (quimb `compress_l2bp` / `compress_d2bp`, Tindall–Fishman
gauging) truncates a bond using the **rank-1 product of BP messages** as the bond
environment — i.e. the *simple-update* reduced density matrix. That environment
ignores the inter-tensor loop correlations *around* the bond, so the truncated
singular vectors are sub-optimal exactly where loops matter (frustration,
criticality, short cycles).

Plan a **cluster-corrected compression** in `pepsy.bp` that builds the bond
environment from a **local cluster** (the bond's tensor + a few neighbours,
loop-corrected via §2) instead of the naive BP-message product, then does the
SVD truncation against that better environment:

- Compute the bond's reduced density matrix / environment from a loop-cluster
  expansion (`loop_cluster_expand` restricted to the bond's neighbourhood, or a
  small exact cluster contraction closed with BP messages on its boundary),
  rather than `m_ij ⊗ m_ji`.
- Truncate with that environment (generalized/oblique SVD), giving a
  **cluster-update**-quality compression that interpolates *simple update*
  (cluster size 0 = rank-1 BP messages) → *cluster update* → *full update*
  (χ_env → ∞ ≈ boundary environment), with the **cluster size as the single
  knob** — the same dial as §2 and the tensy RG-BPLC decoder.
- Keep it thin over quimb: reuse `compress_l2bp` / `compress_d2bp` message
  machinery for the boundary closure and `contract_gloop_expand` /
  `RegionGraph` for the cluster environment; feed the result to the boundary /
  PEPS optimizers (§4) as a drop-in higher-fidelity `chi`-reduction.

This is the explicit hook Gray et al. (arXiv:2510.05647) flag — the loop cluster
expansion "can be used to approximate the environment when compressing tensors …
generalizing the so-called cluster update" — and it is cheaper than a full
boundary environment while strictly better than the rank-1 BP-message default.

### References

- Concept note: `learning/bp.md` (loop / cluster expansion sections).
- Evenbly et al., *Loop series expansions for tensor networks*,
  Phys. Rev. Research 8, 013245 (2026), DOI `10.1103/vqks-cr6x`.
- Gray et al., *Tensor Network Loop Cluster Expansions for Quantum Many-Body
  Problems*, arXiv:2510.05647.
- Lubasch, Cirac, Bañuls, *Algorithms for finite PEPS* / *Unifying PEPS
  contractions* (2014) — simple- vs cluster- vs full-update environments.
- Tindall & Fishman, *Gauging tensor networks with belief propagation*,
  SciPost Phys. 15, 222 (2023) — the BP-gauge (simple-update) compression this
  generalizes.

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

### Current progress (validated)

- The spinful Fermi-Hubbard `U1U1` direct MPO energy path is repaired for the
  tested finite-chain cases. Pepsy now evaluates the bosonic Jordan-Wigner MPO
  against a bosonized Symmray fermionic MPS with the required bond-sector phase.
- The MPO hopping convention is documented and regression tested: for `i < j`
  the left endpoint carries the JW endpoint parity, interior sites carry parity,
  and the direct MPO expectation does not fall back to source terms.
- Focused symmetric-tensor tests and the full package suite passed after the
  fix and warning cleanup (`tests/test_symmetric_tensors.py` and `pytest -q`).
- The 4 by 3 periodic `fh_mps`-shaped scratch run matched local terms and MPO
  energy through the first two imaginary-time schedule blocks. The last
  `tau=0.01` block still needs a complete notebook rerun because runtime, not a
  mismatch, interrupted the scratch check.
- `SymDMRG2` now provides the public two-site DMRG entry point. Dense/quimb MPOs
  delegate directly to `quimb.tensor.DMRG2`; Symmray MPOs select the Pepsy
  OBC block-sparse path, infer the fixed total charge from `SymMPS`, and
  optimize in that fixed sector. Periodic lattice edges are represented as
  long-range terms in an OBC MPO.
- The Pepsy-native Symmray DMRG internals now include dense left/right
  environments for `<psi|MPO|psi>`, dense norm environments for `<psi|psi>`,
  block-sparse environments for projected `H_eff`, sector-preserving `H_eff`
  and `N_eff` matvecs in the exact current `theta` block layout, an explicit
  dense generalized diagnostic solve, and a quimb-style Lanczos/LinearOperator
  local solver after MPS canonicalization. The SVD writeback records the
  actual Symmray bond sectors kept at each two-site split. The sweep also
  records per-window `N_eff ~= I` diagnostics and local-solver diagnostics so
  failures can be traced to a site, direction, theta dimension, and matvec
  backend.
- The Symmray `H_eff` matvec is now block-native by default. It avoids
  Symmray's fused multi-leg shape-mismatch path by contracting one shared leg
  at a time and tracing the remaining shared legs, then projecting back to the
  current theta block sectors. The dense-aligned matvec remains available as
  `matvec_backend="dense_reference"` for validation.
- A four-site OBC Fermi-Hubbard U1U1 regression now forces Lanczos and compares
  the final DMRG energy with a dense fixed-sector ED oracle. With enough
  initial bond-sector support, the run reaches the `(N_up,N_down)=(2,2)` ED
  ground energy to numerical precision.
- The first sector-enrichment/noise convergence layer is implemented for the
  Symmray OBC path. `sector_enrichment="template"` builds a same-charge random
  template MPS, merges its virtual-bond charge maps into the current MPS,
  copies existing blocks into the expanded layout, and fills newly valid blocks
  with zero or small `sector_noise`. The narrow `bond_dim=2` L=4 FH U1U1
  regression now reaches the same fixed-sector ED energy after enrichment.
- Symmray sweeps now build only the static-side dense, norm, and block
  environments required for each direction, grow the moving side incrementally,
  and reuse the completed norm environment for normalized energy readout. This
  removes the full post-sweep environment rebuild from the solve loop.
- `sector_enrichment="adaptive"` now repeats template enrichment before every
  sweep. Template construction preserves the current physical sector map rather
  than reducing restricted U1U1 states to a bare physical dimension.
- `SymDMRG2` now accepts quimb-style `p0`, `bond_dims`, `cutoffs`,
  `max_sweeps`, `sweep_sequence`, `verbosity`, and `suppress_warnings`
  controls, exposes a public `sweep("R"/"L", ...)`, and uses quimb's progress
  bar helper for verbose Symmray sweeps. `solve` still returns `self`; the
  quimb-style convergence boolean is stored on `converged`.
- `profile=True` now records JSON-friendly timing diagnostics for canonicalize,
  dense/norm/block environment setup, environment updates, norm checks, local
  eigensolves, projected matvecs, SVD splits, enrichment, sweeps, and solves.
  `profile_summary()` aggregates phase counts and elapsed time.
- Added `benchmarks/symdmrg2_fh_u1u1.py`, a deterministic FH U1U1 OBC
  benchmark harness that emits case metadata, final energy, and profiling data
  as JSON.
- SymDMRG2 now has TeNPy-style robust random and product-growth initialization
  paths for hard mapped-2D DMRG cases. `SymMPS.random_unitary_evolution(...)`
  and `SymMPS.random_unitary_for_model(...)` build well-conditioned random MPS
  starts by growing a product state with charge-preserving two-site random
  unitary layers. Product-like starts also take the automatic gentle
  bond-dimension ramp by default.
- Native block Lanczos now uses TeNPy-style Ritz stopping and adaptive
  truncation-aware P-error tolerance. The local solve records residual/gap
  P-error diagnostics and can update the next sweep's `local_eig_p_tol` from
  the observed maximum SVD truncation error.
- The 6 by 6 mapped-PBC adaptive-P-tolerance counter run showed
  `num_local_eig_p_tol_updates == 0`, so truncation-coupled Lanczos tolerance is
  safe but inert at `chi=32`. The speed bottleneck is per-matvec cost rather
  than local Krylov iteration count.
- The first per-matvec speed step compiles cached block-sector pair schedules
  inside each projected problem. After the first Symmray contraction establishes
  the exact output block template, non-fermionic NumPy-backed cache-hit matvecs
  reuse that sector-pair plan as output-block-level dense matmuls instead of
  asking Symmray to rediscover the same block routing.
- The next DMRG step is to run the 6 by 6 mapped-PBC control with compiled
  block plans enabled and compare `avg_lanczos_matvecs`,
  `lanczos_stop_reasons`, projected-problem cache hits, compiled block-plan
  uses, and per-matvec timing totals. If the gap remains large, continue with
  deeper hot-loop work: fused projected-problem contraction routing,
  allocation reuse, and block-native contraction kernels.

### Validation

- Abelian (`Z2` / `U1`) PEPS norm `⟨ψ|ψ⟩` matches a dense contraction on a tiny
  lattice.
- Fermionic norm and a small Fermi-Hubbard energy match ED on a 2×2 patch,
  exercising `label` / `dummy_modes` phase tracking and the conjugation
  phase-flip.

### References

- Concept note: `learning/symmray.md`.
- Fermionic MPO convention note: `learning/fermionic_mpo.md`.
- Development/debug note: `docs/development/fermi_hubbard_u1u1_mpo_notes.md`.
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
- Whether to keep the repaired finite-chain Fermi-Hubbard MPO bridge as the
  long-term route, or later add a fully native fermionic MPO path that avoids
  bosonizing the MPS.
- Whether loop-series corrections expose free-energy, transfer-matrix, and
  2-site-density-matrix outputs behind one `BPResult` or separate result types.

---

## 9. Backlog

- BP-gauge initializer wired into the existing sweep / global / PEPS optimizers.
- Shared reporting for BP-vs-boundary-MPS accuracy/cost traces.
- Fermionic optimizer path (gradient-based) once §4 norms/energies validate.

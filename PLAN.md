# PLAN.md — Belief Propagation, quimb integration, and Symmetric Tensor Networks

Status: draft / living document
Last updated: 2026-06-24
Owners: pepsy maintainers + coding agents

This document plans three related workstreams to extend `pepsy`:

1. **BP** — Belief Propagation (BP) contraction and gauging, plus the *loop
   series / loop cluster expansion* corrections that turn BP from an
   uncontrolled approximation into a controlled one.
2. **quimb** — lean on existing `quimb` implementations (BP, gauging,
   `TensorNetwork` machinery) instead of reinventing them, and wire them into
   pepsy's public API.
3. **Symmetric TN** — block-sparse abelian-symmetric and fermionic tensors via
   [`symmray`](https://github.com/jcmgray/symmray), so PEPS/MPS workflows can
   exploit U(1)/Z2 symmetries and fermionic statistics.

These three are deliberately grouped: `quimb` is the substrate, BP is the new
algorithmic layer, and `symmray` provides the symmetric array backend that both
can run on top of (all three share the same author/ecosystem, so they compose).

Companion folders:
- `history/` — session-to-session hand-off notes for agents (read on start,
  append on finish). See `history/README.md`.
- `learning/` — conceptual docs explaining the implementation and the ideas
  behind each workstream. See `learning/README.md`.

---

## 0. Background & references

### Belief propagation + loop expansions
- **Loop series expansions for tensor networks** — Evenbly, Pancotti, Milsted,
  Gray, Chan, *Phys. Rev. Research* 8, 013245 (2026).
  DOI: https://doi.org/10.1103/vqks-cr6x ·
  PDF: https://journals.aps.org/prresearch/pdf/10.1103/vqks-cr6x
  - BP fixed point → systematic loop ("excitation") corrections; the BP vacuum
    is the zeroth-order term, loop excitations are higher-order terms whose
    weights are (approximately) exponentially suppressed in their degree.
  - Provides recipes for free-energy density, transfer matrix, and 2-site
    density matrix of (i)PEPS on hexagonal / kagome / square lattices.
- **Tensor Network Loop Cluster Expansions for Quantum Many-Body Problems** —
  Gray, Park, Evenbly, Pancotti, Kjønstad, Chan, arXiv:2510.05647 (2025–2026).
  PDF: https://arxiv.org/pdf/2510.05647
  - Analyzes the loop cluster expansion as a systematic correction to BP for
    general many-body problems; ground-state observables/energies in 2D & 3D,
    open & periodic boundaries, spin & fermion problems; error converges
    ~exponentially with cluster size.
- Foundational BP↔TN links cited by the above:
  - Alkabetz & Arad, *Phys. Rev. Research* 3, 023073 (2021) — BP ≈ simple update.
  - Tindall & Fishman, *SciPost Phys.* 15, 222 (2023) — gauging TNs with BP.
  - Chertkov & Chernyak (2006) — loop calculus.

### quimb
- `quimb.tensor` already ships BP variants and BP-based gauging
  (e.g. dense/loopy 1-norm and 2-norm message passing, `tn.gauge_all`,
  contraction via `cotengra`). pepsy already depends on `quimb`, `cotengra`,
  and `autoray`.

### symmray
- `symmray` — minimal block-sparse abelian-symmetric & fermionic arrays
  (Z2, U1, Z2Z2, U1U1; fermionic via graded/Grassmann algebra), backed by
  `numpy`/`torch`/any `autoray`-compatible backend, and designed to drop into
  `quimb.tensor.TensorNetwork` objects. Constructors of note:
  `PEPS_abelian_rand`, `PEPS_fermionic_rand`, `TN_*_from_edges_rand`,
  `ham_tfim_from_edges`, `ham_heisenberg_from_edges`, `ham_fermi_hubbard_*`.
  Reference: Gao et al., *Phys. Rev. Research* 7, 023193 (2025).

---

## 1. Where this lands in pepsy

Current relevant architecture (do not break these public entry points):
- `pepsy.boundary`: `build_bra_ket`, `BdyMPS`, `CompBdy`, `contract_boundary`,
  `normalize`, `infidelity` — the existing boundary-MPS contraction path.
- `pepsy.tensors`: `OneDMap`, product/identity constructors, contraction
  optimizers, observables, `tn_norm`, `tn_fidelity`.
- `pepsy.backends`: backend inference/conversion + torch/JAX/CuPy linalg
  registration.
- `pepsy.operators`, `pepsy.solvers`, `pepsy.optimizers`, `pepsy.sampling`.

BP is conceptually a *peer* of boundary-MPS contraction: another way to
approximate `⟨ψ|ψ⟩`, environments, and reduced density matrices. The natural
home is a new subpackage that mirrors `boundary/`:

- `src/pepsy/contraction/` (or `src/pepsy/bp/`) — new subpackage:
  - `bp.py` — BP message init / update / fixed-point loop, Bethe free energy.
  - `loops.py` — loop-excitation enumeration + weights (vqks-cr6x §III–IV).
  - `expansion.py` — series accumulation / self-consistent completion
    (vqks-cr6x App. B) and cluster expansion (arXiv:2510.05647).
  - `environments.py` — BP environments, transfer matrices, 2-site RDMs.

Symmetric tensors are a *backend/array* concern, not an algorithm:
- `src/pepsy/symmetry/` (thin) — helpers to build symmray-backed PEPS/MPS,
  validate symmetry tags, and bridge pepsy lattice tags (`X{i}`, `Y{j}`, `I…`,
  `k…`/`b…` legs) to symmray index `dual`/charge conventions.

Keep the public API lazy and centralized in `src/pepsy/__init__.py`
(`_SYMBOL_MODULES`, `_MODULE_EXPORTS`, `__all__`) per AGENTS.md.

---

## 2. Workstream A — Belief Propagation

Goal: a BP contraction path that produces (a) a BP fixed point + gauge,
(b) BP estimates of norm / energy / local observables, and (c) optional loop
corrections with a tunable degree.

### A1. BP fixed point (MVP)
- Build the closed double-layer network from `build_bra_ket(...)` (reuse the
  existing tagging: `KET`/`BRA`, `X*`/`Y*`/`I*` validation, bra reindexing).
- Implement message vectors per edge, the BP update (Eq. 2 of vqks-cr6x), and a
  fixed-point loop with normalization (`m·m' = 1`) and a convergence criterion.
- Output a `BPResult` dataclass mirroring `BoundaryContractResult`
  (`res.cost`, `res.fidel`-style fields) so call sites stay uniform.
- **Prefer wrapping `quimb`'s BP** (see Workstream B) for the core loop; only
  hand-roll if quimb's API can't express what we need.

### A2. BP gauging
- Use the BP fixed point to gauge the PEPS (Tindall–Fishman). This doubles as
  "simple update" environments and as a better initializer for boundary-MPS.

### A3. BP observables & environments
- BP vacuum contributions (Eq. 3–4), Bethe free energy (Eq. 5).
- Transfer matrix and 2-site reduced density matrix from BP environments
  (vqks-cr6x Fig. 3) → feed `pepsy` observables / energy estimates.

### A4. Loop series / cluster corrections
- Edge projectors: rank-1 BP-ground `P` + complement `Q` (Eq. 6).
- Enumerate non-zero loop excitations (dangling excitations vanish, Eq. 7),
  compute weights, and accumulate via the self-consistent completion
  (App. B.2) and/or multi-excitation counting (App. B.3).
- Expose `degree` (max cluster size) as the accuracy knob; document the
  cost/accuracy trade-off and the failure modes (degenerate BP fixed points,
  GHZ-like states).
- Cross-check against boundary-MPS at large `chi` as the "numerically exact"
  reference (the papers' validation strategy, and pepsy already has this path).

### A5. Tests
- New `tests/test_bp.py`: fixed-point convergence on a small PEPS; BP norm vs
  `contract_boundary` at large `chi`; loop expansion reduces error vs plain BP
  on an AKLT or random PEPS; degenerate-fixed-point caveat documented.

---

## 3. Workstream B — quimb integration

Goal: avoid reinventing BP / gauging / contraction; wrap and adapt quimb.

- Audit `quimb.tensor` BP + gauging APIs and pin the minimal surface pepsy
  needs (message passing, fixed point, `gauge_all`-style helpers, environment
  extraction).
- Provide thin pepsy adapters that accept pepsy-tagged networks and return
  pepsy result dataclasses, so users don't context-switch between APIs.
- Keep quimb optional-but-core (it is already a hard dependency); guard any
  newer-than-pinned quimb features and document the minimum version.
- Ensure pepsy's `cotengra`-based contraction optimizers (`build_optimizer`,
  `contraction_opt="auto-hq"`) are reused for the loop-excitation sub-networks.
- Do **not** add seed kwargs to optimizer builders (tests assert their absence).

### B0. Gate-routing audit: dimension-aware SWAPs

Status: implemented as a small prerequisite for Tensy PF PEPS replay.

- Finding: quimb's adjacent `tensor_network_gate_inds(..., contract="split")`,
  `tensor_network_gate_inds(..., contract="reduce-split")`, and
  `gate_simple_` accept rectangular two-site tensors whose output physical
  dimensions differ from their input dimensions. That is enough to represent a
  mixed-dimension SWAP with shape `(d_b, d_a, d_a, d_b)`.
- Requirement: pepsy's long-range `gate` / `gate_simple` SWAP routing must infer
  the **current** physical index dimensions before each forward and reverse
  SWAP. A single cached `qu.swap(dim=2)` is only valid for binary sites.
- Scope: keep the public API stable; make dimension-aware SWAPs the internal
  default for 1D/2D/3D routed `split` / `reduce-split` paths and simple-update
  routing. For binary sites this is behavior-preserving.
- Validation target: mixed physical dimensions such as Tensy PF fused sites
  (`dim=4` frame, `dim=2` measurement, and possible larger selector sites)
  should route through spectator sites and swap back to the original layout for
  both direct split/reduce-split and simple-update replay.

Deliverable: a short `learning/quimb.md` mapping "pepsy concept → quimb API",
plus adapters under the new contraction subpackage.

---

## 4. Workstream C — Symmetric & fermionic TN (symmray)

Goal: let PEPS/MPS workflows carry abelian symmetry and fermionic statistics.

### C1. Optional dependency
- Add `symmray` as an optional extra in `pyproject.toml`
  (`[project.optional-dependencies] symmetry = ["symmray"]`); keep it optional
  and use `pytest.importorskip("symmray")` in tests (per AGENTS.md).

### C2. Backend bridge
- Teach `pepsy.backends` to recognize symmray arrays (they are `autoray`
  compatible) and to leave their block structure intact (no silent dense
  coercion). Confirm linalg (`svd_truncated`, `qr_stabilized`, `eigh`) routes
  to symmray's implementations.

### C3. Constructors & tags
- Helpers to build symmray-backed PEPS via `PEPS_abelian_rand` /
  `PEPS_fermionic_rand` and tag them with pepsy lattice tags so
  `build_bra_ket` / `BdyMPS` / BP all accept them unchanged.
- Document the index-convention bridge: symmray `dual=False` = outward/ket-like
  ↔ pepsy `k…` legs; `dual=True` = inward/bra-like ↔ `b…` legs. Handle the
  fermionic conjugation phase rules (dangling dual legs need explicit
  phase-flip when both bra- and ket-like legs are present).

### C4. Fermionic correctness
- Validate fermionic norm `⟨ψ|ψ⟩` and a Fermi–Hubbard energy on a tiny lattice
  against a dense reference, exercising symmray's `label`/`dummy_modes` phase
  tracking.

### C5. Tests
- `tests/test_symmetry.py`: skipped without symmray; Z2/U1 PEPS norm matches a
  dense contraction; fermionic Hubbard local operator energy matches ED on a
  2×2 patch.

---

## 5. Milestones (suggested order)

1. **M0 — scaffolding (this PR):** `PLAN.md`, `history/`, `learning/`.
2. **M1 — quimb BP wrapper:** BP fixed point + norm via quimb, pepsy adapter +
   `BPResult`, test vs boundary-MPS. (Workstream B + A1)
3. **M2 — BP gauging & observables:** gauge, Bethe free energy, 2-site RDM,
   energy estimate. (A2–A3)
4. **M3 — loop expansion:** degree-tunable corrections + tests vs large-`chi`
   reference. (A4–A5)
5. **M4 — symmray backend bridge:** optional dep, backend recognition, abelian
   PEPS norm test. (C1–C3, C5)
6. **M5 — fermionic support:** fermionic constructors, phase handling, Hubbard
   energy test. (C4)
7. **M6 — docs:** `docs/howto/` + `docs/tutorials/` pages; update `docs/api/`
   and `tests/test_public_api.py` for any new public symbols.

Each milestone is independently shippable and testable.

---

## 6. Cross-cutting requirements (from AGENTS.md)

- Use new package namespaces; never add old flat modules
  (`pepsy.core`, `pepsy.gates`, …).
- Any new public symbol → update owning subpackage `__all__`, top-level
  `src/pepsy/__init__.py`, `docs/api/`, and `tests/test_public_api.py`.
- Keep optional deps optional; use `pytest.importorskip` in tests.
- Don't change default bond dims, tolerances, or solver choices unless the task
  is specifically about accuracy/convergence/performance.
- Python 3.11; run focused tests first, then broader.
- Validation per area:
  - API/layout: `pytest -q tests/test_public_api.py tests/test_package_layout.py`
  - Boundary/contraction: `pytest -q tests/test_prepare_boundary_inputs.py`
  - Core/observables: `pytest -q tests/test_core_seed.py`
  - New BP: `pytest -q tests/test_bp.py`
  - New symmetry: `pytest -q tests/test_symmetry.py`
  - Docs: `sphinx-build -W -b html docs docs/_build/html`

---

## 7. Open questions / risks

- BP fixed point may not exist / be unique for ground states of local
  Hamiltonians; loop expansion diverges for (near-)degenerate fixed points
  (GHZ-like). Need a robust convergence + fallback-to-boundary-MPS story.
- Square-lattice loop enumeration grows fast with degree; need pruning /
  cost caps and reuse of `cotengra` for sub-network contractions.
- Exact division of labor between pepsy and quimb for BP (wrap vs. own) — decide
  in M1 after the quimb API audit; record the decision in `history/`.
- symmray API is young (v0.2.x); pin a version and isolate behind the bridge.

---

## 8. How agents should use this file

1. On session start: read this `PLAN.md` and the latest entry in `history/`.
2. Pick the current milestone; do the smallest shippable slice.
3. On session end: append a `history/` entry (what changed, why, validation,
   next step) and update this plan's *Status* / milestone checkboxes.

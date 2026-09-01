# Symmetric & fermionic tensors via symmray

[`symmray`](https://github.com/jcmgray/symmray) is a minimal library for
**block-sparse, abelian-symmetric, and fermionic** arrays that look as much like
ordinary ndarrays as possible. It is `autoray`-compatible (blocks can be numpy /
torch / jax / cupy) and is designed to drop into
`quimb.tensor.TensorNetwork` objects — which is exactly what pepsy already uses.
That makes it the natural way to give pepsy PEPS/MPS workflows U(1)/Z2 symmetry
and fermionic statistics without rewriting the algorithms.

Reference implementation paper: Gao, Zhai, Gray, Peng, Park, Liu, Kjønstad, Chan
— *Fermionic tensor network contraction for arbitrary geometries*,
Phys. Rev. Research 7, 023193 (2025).

## Upstream audit and current opportunities

This note is the human-readable companion to the native fermion skill. Before
changing native Symmray code, check the [latest
documentation](https://symmray.readthedocs.io/en/latest/index.html), the
[official changelog](https://symmray.readthedocs.io/en/latest/changelog.html),
and the installed package. Record what was checked here so a future agent can
tell verified behavior from a proposed design.

Audit date: **2026-08-31**. The shared Python 3.12 environment reports
`symmray 0.3.2.dev6+ga17699db6`. This is a development build, so a documented
feature still needs an installed-API probe and a focused regression before it
becomes a Pepsy dependency or changes a native execution path.

The most useful current upstream directions are:

| Symmray capability | Why it matters to Pepsy | Adoption status |
| --- | --- | --- |
| `.to(backend=..., dtype=..., device=...)` | Gives one public route for explicit block/backend conversion. | Adopt in a future adapter; never use it for silent coercion or densification. |
| Python Array API and Autoray dispatch | Reduces private NumPy/Torch/JAX type checks in generic tensor code. | Adopt for new backend-generic code. |
| `tensordot(mode="fused")` / `tensordot(mode="blockwise")` | Makes the performance/sparsity tradeoff explicit. Fused can fill missing blocks; blockwise costs more Python overhead. | Adopt as a documented policy choice and benchmark before changing defaults. |
| Pairing-capable local Hubbard builders and coordination arguments | Could simplify future pairing, BCS, and PEPO terms. | Prototype only after matching Pepsy basis order, operator order, and charges. |
| `to_pytree`/`from_pytree`, `from_blocks`, and native truncated linalg drivers | Useful for checkpointing, autodiff, and GPU work without flattening the network. | Prototype behind focused reconstruction and gradient tests. |

The practical lesson is that Symmray is becoming a better backend boundary,
not a reason to remove Pepsy's model-facing API. `Fermion` should continue to
own the simple user vocabulary while delegating graded algebra, charge
selection, phase tracking, and block operations to public Symmray APIs.

## Design and implementation guidance

When adding a native feature, keep the following boundaries visible:

1. **Metadata first.** Resolve local basis, symmetry, charge, duals, labels,
   and dummy modes before choosing a contraction or linalg route. A dense
   numerical match is not sufficient if these invariants are lost.
2. **Choose contraction mode deliberately.** Use `fused` for compatible,
   performance-critical block structures after measuring memory behavior. Use
   `blockwise` for sparse, mixed-backend, or metadata-sensitive operands. In
   both cases let Symmray prepare and synchronize fermionic phases.
3. **Keep conversion explicit.** A Pepsy `to_backend` callback or an explicit
   Symmray `.to(...)` call may move blocks. No native route should silently
   convert Torch/JAX data to NumPy or densify a block-sparse tensor.
4. **Separate native and compatibility representations.** Jordan--Wigner or
   bosonized objects remain useful for reference and selected hot paths, but
   they must not replace native graded terms or hide a phase mismatch.
5. **Adopt upstream features in small slices.** For each candidate, add one
   native-vs-dense check, one metadata check, and one backend check before
   wiring it into MPS, PEPS, or SymDMRG2.

## Maintenance loop

At the beginning of future Symmray-native work:

1. Read the docs and changelog links above and compare them with the installed
   `symmray.__version__`.
2. Inspect the installed implementation of any changed API, especially
   `fermionic_local_operators`, phase preparation, contraction modes, and
   linalg drivers.
3. Classify changes as **adopt**, **prototype**, or **defer**. Update this
   audit with the date and reason; do not treat a release headline as a code
   change request.
4. Run the closest Symmray/fermion regression and the repository lint gate.
5. Re-check MPS, PEPS/PEPO, and SymDMRG2 separately when the change touches
   contraction ordering, QR/SVD, backend conversion, or debosonization.

The 2026-08-31 MPO audit also confirmed that native Quimb contraction fuses
Symmray physical legs into packed charge sectors. Pepsy's MPO boundary now
passes explicit physical `index_maps` to Symmray when converting the fused
result to dense, restoring computational-basis order without changing the
native block representation. This adopts the installed public API and is
covered by an interior-support MPO regression.

This gives the agent a repeatable upgrade path while keeping design intent
readable for a human maintainer.

## Core objects

- **`AbelianArray`** — block-sparse symmetric array with four parts:
  1. `.indices` — a `BlockIndex` per axis (charge distribution + `dual` flag).
  2. `.charge` — overall charge selecting allowed sectors.
  3. `.blocks` — dict mapping each non-zero sector → a raw (numpy/torch/…) array.
  4. `.symmetry` — how charges combine / negate / which are valid.
- **`BlockIndex`** — `.chargemap` (charge → size) and `.dual` (bool).
  Convention: `dual=False` flows **outwards / +ve / ket-like**;
  `dual=True` flows **inwards / −ve / bra-like**.
- **`FermionicArray`** — subclass of `AbelianArray` implementing the
  graded/Grassmann approach: a parity per charge plus `dual` lets all fermionic
  swaps and sign phases be handled *locally*. Phases are tracked lazily
  (`.phases`) on transpose/fuse/conj/tensordot/trace/decompositions and applied
  via `phase_sync`. Odd-parity tensors use `label` + `dummy_modes` to keep global
  phases correct.

## Symmetries & linalg available

- Symmetries: `Z2`, `U1`, `Z2Z2`, `U1U1` (plus custom via `symmray.symmetries`).
- Linalg matches numpy where possible, with TN-specific drivers Quimb uses:
  `svd_truncated`, `eigh_truncated`, `qr_stabilized`,
  `svd_via_eig_truncated`, `svd_rand_truncated`, `qr_via_cholesky` — all accept
  an `absorb` kwarg controlling where singular/eigen values go. Configure
  native Torch PEPS autodiff through the canonical
  `register_torch_linalg(..., quimb_split_drivers=True)` call. Its QR/LQ path
  uses the zero-safe `phase(0)=1` convention, preserving exact splits for
  rank-deficient native fermionic blocks. A QR pivot at or below the shared
  scale-relative epsilon uses a regularized VJP, so near-singular blocks stay
  finite without suppressing the full local PEPS gradient; only an exactly
  zero block takes the zero-VJP convention.
- Key array ops: `conj`, `reshape`, `tensordot` (`mode="fused"` or
  `"blockwise"`), `trace`, `transpose`, plus `fuse` / `multiply_diagonal`.

## Ready-made constructors (integrate fast)

`symmray` provides quimb-`TensorNetwork` constructors and Hamiltonians:
- Networks: `PEPS_abelian_rand`, `PEPS_fermionic_rand`,
  `TN_abelian_from_edges_rand`, `TN_fermionic_from_edges_rand`.
- Hamiltonians: `ham_tfim_from_edges`, `ham_heisenberg_from_edges`,
  `ham_fermi_hubbard_from_edges`, `ham_fermi_hubbard_spinless_from_edges`.
- Local fermionic operators: `fermi_hubbard_local_array`,
  `fermi_hubbard_spinless_local_array`, number/spin operators, and builders
  `build_local_fermionic_array` / `build_local_fermionic_elements`.

## How it bridges into pepsy

- **Optional dependency:** add `symmray` as an extra in `pyproject.toml`
  (`[project.optional-dependencies] symmetry = ["symmray"]`) and gate tests with
  `pytest.importorskip("symmray")`.
- **Backend recognition:** `pepsy.backends` should treat symmray arrays as a
  valid autoray backend and **never silently densify** their block structure;
  route decompositions to symmray's linalg.
- **Tag bridge:** pepsy lattice tags (`X{i}`, `Y{j}`, `I…`) and leg conventions
  (`k…` for ket legs, `b…` for bra/operator-output legs) must map onto symmray's
  `dual`:
  - `k…` (ket, outward) ↔ `dual=False`
  - `b…` (bra, inward)  ↔ `dual=True`
- **Conjugation phases (fermions):** `.conj()` applies `phase_permutation` by
  default. If a network has *both* bra- and ket-like dangling legs (e.g. norms,
  cluster/infinite settings), the dangling dual legs of the conjugated network
  must be **explicitly phase-flipped** — this is the main correctness pitfall to
  cover with tests.

## What to validate

- Abelian (`Z2` / `U1`) PEPS norm `⟨ψ|ψ⟩` matches a dense contraction on a tiny
  lattice.
- Fermionic norm and a small Fermi–Hubbard energy match ED on a 2×2 patch,
  exercising `label`/`dummy_modes` phase tracking and the conjugation phase-flip.

For the repaired finite-chain Fermi-Hubbard MPO bridge, see
`fermionic_mpo.md`. The important convention is that a native
fermionic Symmray MPS must be bosonized into the same Jordan-Wigner gauge as the
bosonic MPO before direct MPO energy evaluation.

## Other libraries (for comparison)

`abeliantensors`, `yastn`, `pyblock3`, Google `TensorNetwork`, `grassmanntn`,
`TensorKit.jl`. symmray's distinguishing trait is the *local* graded-algebra
treatment of fermions plus tight `quimb`/`autoray` integration, which is why it
fits pepsy with minimal glue.

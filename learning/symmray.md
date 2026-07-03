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
- Linalg matches numpy where possible, with TN-specific drivers quimb uses:
  `svd_truncated`, `eigh_truncated`, `qr_stabilized` (good for gradient-based
  optimization), `svd_via_eig_truncated`, `svd_rand_truncated`, `qr_via_cholesky`
  — all accept an `absorb` kwarg controlling where singular/eigen values go.
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
`learning/fermionic_mpo.md`. The important convention is that a native
fermionic Symmray MPS must be bosonized into the same Jordan-Wigner gauge as the
bosonic MPO before direct MPO energy evaluation.

## Other libraries (for comparison)

`abeliantensors`, `yastn`, `pyblock3`, Google `TensorNetwork`, `grassmanntn`,
`TensorKit.jl`. symmray's distinguishing trait is the *local* graded-algebra
treatment of fermions plus tight `quimb`/`autoray` integration, which is why it
fits pepsy with minimal glue.

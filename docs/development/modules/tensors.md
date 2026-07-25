# pepsy.tensors

This package contains Pepsy's tensor-network construction, mapping,
contraction, validation, observable, backend, and symmetric-state helpers.
Other packages should import these helpers through `pepsy.tensors` or the
top-level `pepsy` exports rather than old flat modules.

## Modules

- `core.py`: main implementations for constructors, `OneDMap`, backend
  defaults, contraction optimizers, observables, and dense TN utilities.
- `constructors.py`: facade for product-state, identity, Haar-random, MPS,
  MPO, PEPS, and PEPO constructors.
- `contractions.py`: facade for contraction optimizers, `tn_norm`,
  `tn_fidelity`, and alignment helpers.
- `maps.py`: facade for `OneDMap`.
- `observables.py`: facade for observable and MPO expectation helpers.
- `symmetric.py`: Symmray-backed `SymMPS`, `SymPEPS`, symmetric Hamiltonian,
  gate-stream, charge-sector, and dense-operator conversion helpers.
- `validation.py`: shared PEPS tag and physical-index validation helpers.

Many leaf modules are intentionally thin facades over `core.py`; keep that
structure unless a change has a strong reason to split implementation.

## Main responsibilities

`OneDMap` maps regular 2D or 3D lattice coordinates onto a 1D path. Supported
modes include `snake`, `snake-row-major`, `row-major`, `col-major`, `hilbert`,
`hilbert-row-major`, and `diag`.

Constructors create common tensor-network states and operators:

- `ps_to_mps` (bond-one product states), `ps_to_ttn`, `ps_to_peps`, `ps_to_3dpeps`
- `ps_to_mpo`, `ps_to_pepo`
- `id_to_mpo`, `id_to_pepo`
- `haar_random_state`, `random_haar_qubit`
- `hrs_to_mps` / `hrps_to_mps` (random MPS states), `hrs_to_ttn` /
  `hrps_to_ttn` (random tree states), `hrs_to_peps` / `hrps_to_peps`
  (direct Symmray random PEPS states)

Contraction helpers include:

- `build_optimizer(...)` and `build_compressed_optimizer(...)`
- `contract_hypercompressed_tn(...)`
- `contract_hypercompressed_tn_batch(...)` — torch-only batched amplitudes
  ``<x|psi>`` for many int64 configs via `torch.vmap`, reusing one fixed
  compressed contraction tree (one-hot selection; requires `cutoff=0.0`)
- `tn_norm(...)`, `tn_fidelity(...)`, and `tns_align(...)`

Backend helpers manage package-wide defaults and optional linalg shims:

- `set_default_array_backend(...)` / `get_default_array_backend()`
- `set_default_grad_backend(...)` / `get_default_grad_backend()`
- `reset_default_backends()`
- torch and JAX linalg/stop-gradient registrations. For torch SVD,
  `reg_rel_svd_torch()` is the preferred full-SVD autodiff shim; it installs
  the relative-regularized backward rule also used by `reg_complex_svd_torch()`
  and falls back to SciPy `gesvd` on CPU forward-driver failures.

## Tag and index conventions

PEPS-like networks should carry lattice and site tags:

- `X{i}` for the x coordinate.
- `Y{j}` for the y coordinate.
- `I...` for site identity tags such as `I0,1`.

Physical outer indices conventionally use `k...` for ket legs and `b...` for
bra or operator-output legs. The boundary and optimizer packages depend on
these conventions for shape inference and layer construction.

## Symmetric tensors

`symmetric.py` provides Symmray-backed convenience wrappers and charge-sector
helpers. Symmray remains optional. Code and tests that depend on it should
import lazily or use `pytest.importorskip("symmray")`.

For spinful Fermi-Hubbard states, the named model presets are:

- `fermi_hubbard`: total particle-number `U1` sectors.
- `fermi_hubbard_u1u1`: spin-resolved `U1U1` sectors with charges
  `(N_up, N_down)`.

Use Gao et al., "Fermionic tensor network contraction for arbitrary
geometries", Phys. Rev. Research 7, 023193 (2025),
https://doi.org/10.1103/PhysRevResearch.7.023193 as the main methods
reference for Pepsy/Symmray Fermi-Hubbard examples. The relevant design cue is
to preserve Symmray fermionic parity, symmetry, and leg-order metadata through
gate application, measurement, boundary contraction, and any future arbitrary
graph lattice wrappers.

`SymMPS.fermionic_ordering()` and `SymPEPS.fermionic_ordering()` expose the
package-level record of site order, edge order, local index directions, and the
methods reference. The same record is included in `symmray_mps_summary(...)`
and `symmray_peps_summary(...)` under the `fermionic_ordering` key.

`SymHamiltonian.to_mpo(...)` builds quimb `MatrixProductOperator` objects from
Symmray block-sparse tensors. It supports generic charge-neutral rank-4
two-site terms such as `tfim`/`Z2` and `heisenberg`/`U1`, spinless
Fermi-Hubbard with `U1` or `Z2` symmetry and `delta=0`, and the specialized
spin-resolved Fermi-Hubbard path for `model="fermi_hubbard_u1u1"` with
`symmetry="U1U1"`. Spinful total-`U1` Fermi-Hubbard
(`model="fermi_hubbard"`) intentionally raises `NotImplementedError` until the
degenerate total-charge MPO convention is implemented.

For spinful `U1U1`, the builder uses the four-state local basis with charges
`(n_up, n_down)`, handles onsite `U`/`mu` terms, spin-dependent scalar-or-pair
`t` and `mu` parameters, and creates hopping channels for both spin species.
Spinless and spinful fermionic MPO paths insert fermionic parity on
intermediate MPS-chain sites for non-adjacent mapped hopping terms, so
coordinate-lattice edges preserve signs after flattening through `OneDMap`.

Coordinate edges require `mapper=OneDMap(...)` or explicit `idx2coo`/`coo2idx`
maps when calling `to_mpo(...)`; already-flat integer edges can pass `L=...`.
The method supports optional MPO compression, physical index IDs, `dtype=`, and
`to_backend=` block conversion. Focused coverage lives in
`tests/test_symmetric_tensors.py::test_*hamiltonian*mpo*`.

When changing symmetric behavior, check both dense compatibility and Symmray
routing through PEPS optimizers and boundary contraction paths.

## Editing notes

- Preserve public exports in `src/pepsy/tensors/__init__.py` and top-level
  `src/pepsy/__init__.py` when adding or removing public symbols.
- Do not reintroduce old flat modules such as `pepsy.core`.
- Avoid changing default contraction optimizers, mapper ordering, numerical
  tolerances, or backend coercion unless the task is specifically about that
  behavior.
- Focused validation usually starts with:

```sh
pytest -q tests/test_tensor_constructors.py tests/test_ham.py
```

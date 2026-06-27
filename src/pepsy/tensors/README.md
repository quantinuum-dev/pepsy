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

- `ps_to_mps`, `ps_to_peps`, `ps_to_3dpeps`
- `ps_to_mpo`, `ps_to_pepo`
- `id_to_mpo`, `id_to_pepo`
- `haar_random_state`, `random_haar_qubit`
- `hrps_to_mps`, `hrps_to_peps`

Contraction helpers include:

- `build_optimizer(...)` and `build_compressed_optimizer(...)`
- `contract_hypercompressed_tn(...)`
- `tn_norm(...)`, `tn_fidelity(...)`, and `tns_align(...)`

Backend helpers manage package-wide defaults and optional linalg shims:

- `set_default_array_backend(...)` / `get_default_array_backend()`
- `set_default_grad_backend(...)` / `get_default_grad_backend()`
- `reset_default_backends()`
- torch and JAX complex-SVD/stop-gradient registrations

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
pytest -q tests/test_core_seed.py tests/test_tensor_constructors.py tests/test_ham.py
```

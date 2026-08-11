# pepsy.boundary

This package owns PEPS-like boundary contraction, normalization, and fidelity
helpers. It is one of the central layers used by `pepsy.optimizers`, and API
changes here should be made carefully because downstream packages such as
`tc_gauge` rely on Pepsy boundary behavior.

## Modules

- `metrics.py`: public norm, overlap, normalization, fidelity, and
  infidelity helpers.
- `states.py`: `BdyMPS`, the reusable boundary-MPS store.
- `sweeps.py`: `CompBdy`, the FIT/DMRG-style boundary update engine.

## Core flow

The standard Pepsy boundary path is:

```text
build_bra_ket(ket, bra?) -> BdyMPS(...) -> contract_boundary(...)
```

`build_bra_ket(...)` prepares a tagged double-layer tensor network. The ket is
tagged in place with `KET`; the bra layer is tagged with `BRA`; shared internal
bra indices are reindexed with an `_*` suffix.

`BdyMPS` initializes reusable row and column boundary MPS environments. Its
`mps_b` dictionary uses keys like:

- `Y{i}_l` and `Y{i}_r` for column-sweep environments.
- `X{i}_l` and `X{i}_r` for row-sweep environments.

`CompBdy` updates those environments with `move_bdy(...)` or
`move_step_bdy(...)`, then contracts a final boundary network in `run(...)`.

The local boundary solver is selected with `fit_mode`:

- `"eff"` is the compatibility default and performs cached one-site sweeps.
- `"two-site"` forms neighboring boundary wavefunctions, splits them with a
  backend-native SVD, and can discover bond subspaces up to `fit_max_bond`.
- `"global"` uses the reference global FIT solve.

Two-site sweeps do not rebuild a complete environment for every pair. One
side is cached once per sweep and the moving side is updated incrementally, so
environment construction remains linear in boundary length. PEPS helpers pass
the requested `chi` as the default two-site bond cap; direct `CompBdy` users
should set `fit_max_bond` when they want a lower-rank boundary to grow. New
two-site boundaries start at bond 1 and grow through these local splits.
Reusing a boundary at a larger `chi` does not globally pad it first; lowering
`chi` still compresses existing bonds immediately.

## Public helpers

- `contract_boundary(...)`: contracts a prebuilt double-layer network with a
  supplied `BdyMPS` or `{"bdy": BdyMPS}` holder and returns
  `BoundaryContractResult`.
- `peps_normalize(...)` / `normalize(...)`: normalize a PEPS in place.
- `boundary_norm(...)` / `peps_norm(...)`: compute `<p|p>` without rescaling.
  Pass `return_info=True` for the structured result instead of only the scalar.
- `peps_infidelity(...)` / `infidelity(...)`: compute boundary-based
  infidelity, optionally reusing norm and overlap boundaries.
- `peps_fidelity(...)`: return only fidelity by default, or preserve all three
  contraction results and their FIT diagnostics with `return_info=True`.
- `contract_flat(...)`: contract an already-flat PEPS-like tensor network.

Use `result.cost`, `result.fidel`, and `result.fit_diagnostics` from
`BoundaryContractResult`; do not rely on tuple unpacking. Each typed boundary
FIT diagnostic reports actual iterations and convergence, with detailed sweep
timing populated only when `fit_timing=True`.

## Boundary methods

The default `method="dmrg"` uses Pepsy's `BdyMPS` plus `CompBdy` path. Other
methods route to Quimb-style contraction methods when the network exposes them:

- `method="mps"` uses `TensorNetwork.contract_boundary(...)`.
- `method="ctmrg"` uses `TensorNetwork.contract_ctmrg(...)`.
- `method="hotrg"` uses `TensorNetwork.contract_hotrg(...)`.
- `method="exact"` directly contracts the double-layer network.

When `strip_exponent=True`, helpers preserve Quimb's `(mantissa, exponent)`
representation and Pepsy applies exponent shifts explicitly.

## Editing notes

- Keep optional dependencies optional; tests for torch, JAX, CuPy, NLopt,
  SciPy, or Symmray should use `pytest.importorskip(...)` when needed.
- Preserve holder-dict behavior such as `{"bdy": ...}` because optimizers use
  it to reuse and update boundary state.
- Preserve lattice tag conventions: `X{i}`, `Y{j}`, `I...`, and physical outer
  indices conventionally named `k...` and `b...`.
- Focused validation usually starts with:

```sh
pytest -q tests/test_prepare_boundary_inputs.py
```

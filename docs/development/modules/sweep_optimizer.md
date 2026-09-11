# pepsy.optimizers.sweep

This package owns local PEPS slice optimization with alternating row and
column sweeps.

`SweepOptimizer` is the public entry point. It optimizes one row or column of
PEPS tensors at a time while reusing boundary environments for the local norm
and target-overlap objectives.

## Boundary engines

`boundary_engine="dmrg"` uses the existing Pepsy path:

```text
build_bra_ket(...) -> BdyMPS(...) -> CompBdy.move_bdy/move_step_bdy(...)
```

`boundary_engine="quimb-mps"` uses Quimb MPS environments for local row/column
boundaries. The first half-sweep computes the opposite-side environments once;
subsequent alternating half-sweeps reuse that static side because the moving
side has already rebuilt it from the updated planes. Each active boundary then
moves one row or column at a time and is cached. The adapter lives in
`environments.py` and exposes the legacy surface expected by the current local
objective:

- `mps_b`
- `chi`
- `expand_bnd(...)`
- `normalize()`
- `norm`

Call `store.clear("x")`, `store.clear("y")`, or `store.clear()` after an
external tensor-network mutation when the cached environments should no longer
be reused.

`boundary_engine="auto"` keeps dense inputs on the Pepsy path and routes
Symmray-looking inputs to Quimb MPS.

Torch-backed Symmray arrays remain on Torch and use autograd local solvers;
NumPy-backed Symmray arrays use the finite-difference compatibility path.

## Extraction map

- `optimizer.py`: `SweepOptimizer` and the current sweep orchestration.
- `environments.py`: boundary-engine selection and Quimb MPS boundary store.
- `local_objective.py`: target location for local loss assembly.
- `traces.py`: target location for sweep traces and progress records.

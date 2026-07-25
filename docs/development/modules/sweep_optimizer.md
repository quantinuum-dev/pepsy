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
boundaries. At the start of each half-sweep it computes the opposite-side
environments once, then moves the active boundary one row or column at a time
and caches it. The adapter lives in `environments.py` and exposes the legacy
surface expected by the current local objective:

- `mps_b`
- `chi`
- `expand_bnd(...)`
- `normalize()`
- `norm`

`boundary_engine="auto"` keeps dense inputs on the Pepsy path and routes
Symmray-looking inputs to Quimb MPS.

Torch-backed Symmray arrays remain on Torch and use autograd local solvers;
NumPy-backed Symmray arrays use the finite-difference compatibility path.

## Extraction map

- `optimizer.py`: `SweepOptimizer` and the current sweep orchestration.
- `environments.py`: boundary-engine selection and Quimb MPS boundary store.
- `local_objective.py`: target location for local loss assembly.
- `traces.py`: target location for sweep traces and progress records.

# pepsy.optimizers.peps

This package owns the PEPS/PEPO gate-stream optimizer.

`PepsOptimizer` is the public entry point. It applies one-site gates directly,
builds exact two-site targets, compresses warm starts to the requested PEPS
bond dimension, and optionally refines those warm starts with either
`SweepOptimizer` or `GlobalOptimizer`.

## Boundary controls

Keep the chi controls separated by job:

- `chi`: virtual bond cap for the optimized PEPS/PEPO.
- `boundary_chi`: sweep/global optimizer environment bond dimension.
- `normalize_chi`: standalone normalization bond dimension.
- `evaluation_chi`: diagnostic infidelity bond dimension.

`boundary_engine` selects the sweep cleanup boundary implementation:

- `"auto"`: dense inputs use Pepsy `BdyMPS`/`CompBdy`; Symmray-looking inputs
  use Quimb MPS boundaries.
- `"dmrg"`: force the Pepsy boundary path.
- `"quimb-mps"`: force Quimb MPS boundaries and default scalar metric
  contractions to `method="mps"`.

`boundary_options` is forwarded to the Quimb MPS boundary store when sweep
cleanup uses that engine.

## Extraction map

- `optimizer.py`: `PepsOptimizer` and the current orchestration logic.
- `gates.py`: target location for gate routing helpers.
- `warmstart.py`: target location for warm-start construction.
- `routing.py`: target location for sweep/global backend selection.
- `diagnostics.py`: target location for fidelity and progress records.

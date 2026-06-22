# Plan: PEPS optimizer split

## Status

Implemented:

- `PepsOptimizer` exposes `boundary_engine` and `boundary_options`.
- `boundary_engine="auto"` keeps dense PEPS on Pepsy boundaries and routes
  Symmray-looking sweep cleanup to Quimb MPS boundaries.
- Explicit `boundary_engine="dmrg"` and `"quimb-mps"` choices are preserved.
- Sweep cleanup receives the resolved boundary engine and Quimb options.

## Next steps

- Move gate routing helpers from `optimizer.py` into `gates.py`.
- Move warm-start compression helpers into `warmstart.py`.
- Move sweep/global dispatch into `routing.py`.
- Move step-record and fidelity-trace helpers into `diagnostics.py`.
- Add downstream smoke coverage in `tc_gauge` once the Pepsy API settles.

## Boundary contract

`PepsOptimizer` should not know how to build individual local sweep
environments. It should choose the boundary policy, pass it to
`SweepOptimizer`, and keep scalar normalization/infidelity defaults consistent
with that policy.

For Quimb MPS sweep cleanup, `PepsOptimizer` should:

- pass `boundary_engine="quimb-mps"` to `SweepOptimizer`;
- pass `boundary_options` unchanged;
- default sweep-time normalization to `method="mps"`, `mode_="mps"`, and
  `balance_bonds=False`;
- preserve explicit user overrides through `sweep_kwargs` and per-run
  `sweep_kwargs`.

# Plan: Sweep boundary providers

## Status

Implemented:

- `boundary_engine="dmrg"` for the historical Pepsy boundary path.
- `boundary_engine="quimb-mps"` for Quimb MPS environment stores.
- `boundary_engine="auto"` for Symmray-aware routing.
- Conservative full-axis Quimb environment refreshes after local updates.

## Next steps

- Replace direct `bdy.mps_b[...]` reads with provider methods that request the
  environment around an `axis/index` pair.
- Move local objective construction into `local_objective.py`.
- Move trace and progress summarization into `traces.py`.
- Tighten Quimb reupdates from full-axis refreshes to narrower ranges once the
  provider API is stable.
- Add numerical comparisons between dense Pepsy and Quimb MPS environment
  paths.

## Contract

Boundary providers should support the local sweep optimizer without leaking
their construction details into the objective code. The current compatibility
surface is the legacy `mps_b` dictionary plus `chi`, `expand_bnd(...)`,
`normalize()`, and `norm`; direct provider methods should replace that surface
as the split continues.

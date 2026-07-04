# 2026-07-04 -- SymDMRG2 profiling diagnostics and benchmark harness

- Milestone: First careful scalability measurement layer for the Symmray
  `SymDMRG2` path.
- Branch: `develop`, after `3c0cf3c`.

## What changed

- Added opt-in `profile=True` support to `SymDMRG2`.
  - Profiling is disabled by default.
  - When enabled, the optimizer records JSON-friendly event diagnostics in
    `profile_diagnostics`.
  - `last_profile_diagnostic` returns the most recent event.
  - `profile_summary()` aggregates elapsed time and event counts by phase.
- Timed phases now include:
  - canonicalization
  - dense/norm/block sweep-environment setup
  - per-site environment updates
  - effective-norm checks
  - local dense/Lanczos/generalized eigensolves
  - projected `H_eff` matvecs
  - SVD split/writeback
  - sector enrichment
  - sweep totals
  - solve totals
- `summary()` now includes profile status, event counts, the last profile event,
  and the aggregate profile summary.
- Added `benchmarks/symdmrg2_fh_u1u1.py`.
  - Builds deterministic OBC Fermi-Hubbard U1U1 MPS/MPO cases.
  - Runs `SymDMRG2(..., profile=True)`.
  - Emits JSON with case metadata, final result metadata, profile summary, and
    optional raw profile events.

## Validation focus

- Added a regression that checks profiling records the expected phase names,
  elapsed times, matvec counts, and `summary()` metadata.
- Added a regression that imports the benchmark harness by path and verifies its
  return value is JSON-shaped with profiling data.

Focused result during implementation:

```text
pytest -q tests/test_sym_dmrg.py
  20 passed
```

Final validation before commit:

```text
pytest -q tests/test_sym_dmrg.py tests/test_public_api.py tests/test_package_layout.py
  486 passed

python -m pyflakes src/pepsy/optimizers/sym_dmrg.py tests/test_sym_dmrg.py benchmarks/symdmrg2_fh_u1u1.py
  passed

sphinx-build -W -b html docs docs/_build/html
  passed

python benchmarks/symdmrg2_fh_u1u1.py --length 2 --chi 3 --initial-bond-dim 2 --sweeps 1 --local-solver dense --dense-threshold 100
  emitted JSON with 13 profile events
```

## Remaining work

- Run the benchmark harness across larger `L` and `chi` values on target
  hardware.
- Use the phase timings to prioritize matvec caching/projection improvements.
- Compare one-shot and adaptive enrichment schedules using the benchmark JSON.

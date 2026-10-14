# 2026-10-14 — NetKet PEPS TDVP basics before gauge removal

- Scope: verify basic TDVP elements and native solver wiring before gauge
  removal; discuss spatial/spin-flip symmetry separately. No gauge-removal,
  symmetry-constrained ansatz, sample-space solver or compression-gradient work.
- Branch / baseline: `develop` / `8f8695b`.
- Publication scope: the user authorized committing and pushing these NetKet
  TDVP changes to `develop`. This entry accompanies that scoped commit;
  unrelated compression, fitting, backend and Torch edits remain excluded.

## Findings and fixes

- Exact-contraction `qgt="auto"` passed the `QGTAuto` class as a metric
  constructor, producing a factory without `solve()`. Instantiate the selector
  first, with the selected solver. Compressed automatic selection continues
  to choose a reverse Jacobian.
- At t=0.03, `advance(1e-8)` with dt=0.01 previously ended at t=0.04.
  NetKet's proximity check consumed the first nearby stop before stepping.
  A consumed start sentinel keeps real stops available to cap each step.
  Stop values must also be finite and inside the requested interval.
- A deliberately failing native solver returned NaN accepted parameters
  without an adapter error. Endpoint synchronization now reduces finiteness
  on device and raises clearly. This does not implement rollback or recovery.
- Added a concise equation/ownership/validation table to the
  [TDVP API guide](../docs/api/vmc.md#netket-peps-time-evolution), and updated
  the changelog. Native NetKet continues to own forces, QGT and integration.

## Validation

- Independent two-spin covariance equations cover real/holomorphic charts,
  sampled/full-sum states, automatic/dense/PyTree/on-the-fly QGTs and named
  pseudoinverse/Cholesky/CG solvers. Together with tiny/adjacent/invalid stops
  and failing-solver guards: **55 passed**, 67 deselected, 83.10 s.
- PEPS full-sum Euler/midpoint/RK4 timestep-order checks against exact
  Schrödinger and normalized imaginary-time evolution, both charts:
  **12 passed**, 86 deselected, 15.88 s.
- Forced CUDA with clean library resolution: both PEPS coordinate charts
  with CG, exact full-sum midpoint evolution and dense-metric allocation
  forbidden: **2 passed**, 120 deselected, 16.62 s.
- Added a genuinely truncated 3x3/D=2/chi=1 sampled boundary TDVP regression,
  checking finite accepted updates, fixed bond dimension, endpoint residuals
  and matrix-free CG. Final full TDVP/VMC/public-API/package-layout selection:
  **224 passed**, 373.45 s, four expected upstream/deprecation warnings.
  Counts overlap with the focused checks and must not be added.
- Task-owned and final whole-source/test/example Ruff, whitespace checks and
  local API Markdown links pass. An intermediate whole-tree Ruff attempt saw
  an unused variable in a concurrently edited projector test; it was absent
  on the subsequent whole-tree check. This task did not edit that test.
- Existing Python 3.12 environment; NetKet 3.22.4, JAX/JAXlib 0.11.2,
  Flax 0.12.10, Quimb 1.15.1.dev91+gb3bf45a50,
  Symmray 0.4.1.dev18+g44fc5ae53. No dependency changes.
  Installed signatures/source and public native TDVP documentation inspected.

## Publication validation

- Exported the Git index to an isolated snapshot and verified imports use
  that snapshot, excluding concurrent uncommitted implementation changes.
- Snapshot solver/stop/failure, integrator-order, endpoint-diagnostic,
  truncated-boundary and example-snapshot selection: **77 passed**,
  46 deselected, 101.16 s on CPU.
- Snapshot public-API/package-layout selection: **55 passed**, 6.07 s,
  two expected compatibility-alias warnings. Snapshot whole-source/test/example
  Ruff and staged whitespace checks passed. These are scoped checks, not a
  full repository numerical suite.

## Scope of readiness

The verified baseline is the dense-spin PEPS/TFIM bridge, not universal
fermionic or symmetry-constrained trajectory readiness. Basic numerical checks
do not establish long-time, D, sampling or contraction convergence. No full
repository numerical suite or large physical trajectory is claimed.

For uniform square open boundaries, rotations/reflections remain while
translations are broken. TFIM H=-J sum ZZ-h sum X conserves product X;
native computational-basis parity tensors represent product Z instead.
Enforcing the same physical Z2 requires a basis rotation or an explicit
spin-flip projection, plus compatible sampling. This remains a separate
implementation choice, as do tensor-orbit parameter sharing and virtual-leg
permutations. Physical symmetry constraints do not remove virtual gauge freedom.

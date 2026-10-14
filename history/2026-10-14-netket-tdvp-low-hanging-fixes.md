# 2026-10-14 — Practical NetKet TDVP reliability and restart

Implemented configurable stage-solve checks and TFIM physical-state restart;
the user requested low-hanging improvements without matching the full paper.

- Branch / baseline: `develop` / `a820f3c`.
- Publication scope: the user authorized committing and pushing these changes
  to `develop`. This entry accompanies the scoped publication commit;
  concurrent compression, fitting, backend and Torch edits are excluded.

## Changes

- `solver_max_relative_residual` optionally checks each actual native solver
  call, including later RK stages and fresh endpoint diagnostics. It compares
  the regularized/scaled equation passed to the solver using one extra matvec;
  CG remains free of dense metric allocation. A zero RHS requires zero residual.
- Failed/nonfinite velocities become NaN sentinels caught by the existing
  accepted-state guard. There is no rollback, retry or trajectory error bound.
  Default API tolerance is `None` to retain existing solver truncation policy;
  the TFIM example enables the guard with `1e-5` by default.
- Example checkpoints include complete PEPS tensors/layout/scale, simulation
  settings, time/step, and optional reference trajectory data. Numeric archives
  are written to a temporary file and atomically replaced after a complete write.
- `--resume` restores physical state/settings and skips PEPO startup. Additional
  steps and absolute final-time targets preserve time and numbering, including
  a fractional last step. Conflicting saved settings are rejected.
- Restart uses fresh sampling with saved burn-in and fresh solver warm starts;
  sampler/RNG/integrator internals are not restored. Older analysis snapshots
  without tensor layouts cannot resume through this option.
- Updated [VMC API documentation](../docs/api/vmc.md) and
  [changelog](../CHANGELOG.md). Owning changes are in the
  [NetKet bridge](../src/pepsy/vmc/netket.py),
  [TFIM example](../examples/tfim_netket_tdvp.py) and
  [TDVP regressions](../tests/test_netket_tdvp.py).

## Validation

- CPU guarded solver/coordinate/QGT/MC-full-sum matrix, invalid controls,
  truncated boundary CG, and reference-mode snapshot/restart selection:
  **66 passed**, 66 deselected, 144.38 s.
- Public API/package layout and atomic-write/old-snapshot checks:
  **57 passed**, 132 deselected, 9.46 s, two expected compatibility warnings.
- Final guard checks, including an inaccurate later RK stage, and checkpoint
  failure checks: **8 passed**, 127 deselected, 12.07 s. Counts overlap.
  The initial later-stage test used a real-coordinate fixture with a constant
  force along its flow and therefore never triggered its intended bad branch.
  Switching to the holomorphic varying-force fixture exercises the failure;
  no implementation change or assertion weakening was needed.
- Whole-source/test/example Ruff, whitespace and local API links passed.
  Existing Python 3.12 environment with NetKet 3.22.4 and CPU JAX; no dependency
  or installed-library changes. No full numerical suite or new GPU checks.

## Publication validation

- Exported the scoped Git index to an isolated snapshot and verified its source
  imports. Concurrent uncommitted implementation changes were absent.
- Snapshot stage-guard/invalid-control/truncated-CG/checkpoint/restart selection:
  **21 passed**, 114 deselected, 69.72 s on CPU.
- Snapshot public API/package layout: **55 passed**, 8.65 s, two expected
  compatibility warnings. Whole-source/test/example Ruff, staged whitespace
  and local documentation links passed. These are scoped checks.

## Deferred scope

The [paper-gap review](2026-10-14-netket-tdvp-paper-gap-review.md) remains the
algorithm comparison. Sample-space/minSR dynamics, small-o storage, adaptive
integration and sequential PEPS sampling were not implemented. Compression
gradient validation, gauge removal, optional symmetry constraints and large
trajectory convergence also remain outside these practical fixes.

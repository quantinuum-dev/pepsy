# 2026-10-12 — Default PEPS boundary FIT to direct guesses

- Scope: user requested direct rather than SRC initialization and confirmation
  of the boundary iteration budget and reuse behavior.
- Branch / baseline: `develop` / `fc72c42`.
- Publication: accompanies the session's authorized commit/push.

`PepsOptimizer` now defaults to `guess-direct` outside sweep mode. Sweep
mode retains `reuse-direct`: direct compression seeds a cut, then FIT refines
the compatible stored boundary MPS. Explicit initialization settings retain
precedence. FIT mode and the ten-sweep boundary budget are unchanged; this
is independent of inner/outer ALS round trips. Changed dependencies require
fresh environment contractions or boundary refitting. Standalone metric
calls construct fresh boundaries.

Updated the [API guide](../docs/api/optimizers/peps.md) and changelog.
The ongoing D=3/4 comparison uses its isolated checkout and remains unchanged.
This default change does not resolve the insufficient norm-cap errors in the
[frozen-state diagnosis](2026-10-12-peps-boundary-components.md).

Validation in the unchanged isolated validation environment, with the active
task's existing upstream audit reused:

- 64 tests passed across `test_peps_sweep_defaults.py`,
  `test_peps_strip_batches.py`, and `test_peps_environment_reuse.py`.
  Numerical normalization checks cover NumPy/Torch complex128 and global/
  full-update modes. Existing tests check refitting and cache invalidation.
- Seven boundary-option routing cases passed in `test_optimize_peps.py`,
  including explicit SRC overrides. The other 153 tests were deselected.
- Ruff over `src tests` and `git diff --check` passed.
- Two warnings: existing Quimb SVD defaults and the tested Torch SRC
  compatibility fallback. No full-suite or new trajectory accuracy claim.

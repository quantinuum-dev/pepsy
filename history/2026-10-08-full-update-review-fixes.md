# 2026-10-08 — Full-update review fixes and strip environment reuse

- Scope: fix the five issues in the
  [independent review](2026-10-08-full-update-independent-review.md), then check
  Hermitian/positive environments, QR/LQ gauges, and column environment reuse
  against [Lubasch et al., Sec. III B](https://arxiv.org/pdf/1405.3259).
- Branch / baseline: `develop`, `fe11e24`.
- Status: implementation, regression tests, API docs and changelog are
  uncommitted. Nothing staged or published.
- Unrelated backend, MPS, cutoff/conversion, JAX test and MPS documentation
  edits were present or appeared concurrently; they were not modified by this
  task. The changelog also includes an independent MPS stabilization entry.

## Fixes

- Adaptive calibration rejects explicit `fit_max_bond` in shared or effective
  sweep options. Run validation happens before initial normalization or gate
  application; direct calibration calls are guarded too. Users can select
  adaptive `start_chi`/`max_chi`, or disable convergence for fixed FIT caps.
  This prevents the reproduced GHZ false plateau without silently replacing
  an explicitly requested cap.
- Cached full-update normalization converts paired caps to the norm component
  and preserves explicit `None` semantics.
- Full-update acceptance uses raw pre/post metric validity. Clipped artifacts
  cannot reject a positive-environment ALS result. Valid worse candidates still
  roll back to the guess.
- A successful precheck retry supplies its effective chi to the postcheck.
  Step records expose that cap, raw values, and detached evaluation records.
- Target normalization now propagates to the reduced problem's reconstruction
  scale. Magnitude is kept in the network exponent and phase on the reduced
  target; the least-squares objective changes only by a common factor.
  Exterior tensors remain unchanged by the fit. Explicit opt-out and the
  `non_unitary` default are tested, including an input exponent of 80.

## Paper audit and cache changes

The existing reduction uses exact Quimb QR on the left and LQ on the right.
The smaller reduced norm is Hermitianized as `(N + N.H)/2`, diagonalized,
and negative eigenvalues are clipped before constructing its square root.
The eigensystem is reused for independent environment gauges. Because the
implementation stores `N = root.H @ root`, both leg unfoldings use QR;
this is the conjugate/transposed convention of the paper's QR/LQ construction
for `N = X @ X.H`. Singular gauges are skipped. Tests inspect the actual
metric passed to ALS for Hermiticity/PSD and verify that independent gauges
whiten a separable positive metric to a scaled identity.

The weighted-QR solver regauges after each local solve. At the initial review
stage, Quimb controlled its own inner iterations and Pepsy balanced its
returned factors. The follow-up below replaces that full-update path with
monitored public Quimb sweeps and QR/LQ regauging between sweeps. No upstream
library was modified; the shared reduced ALS solver now offers opt-in
monitoring while preserving its unmonitored default.

Outer boundary MPS cuts were already reused. This task adds exact
prefix/suffix contractions inside the current row or column, with no extra
truncation. Cache validation includes original PEPS source arrays (rather than
fresh bra allocations), Torch mutation counters, index/tag layouts, boundary
arrays, preceding partial contractions, and the contraction optimizer.
Input references prevent recycled array identities from matching stale entries.
Moving to another strip clears its entries, bounding storage to one strip.
Network scale is kept separate because the reduced metric is required only
up to an overall positive scalar.

Partial and final contractions receive the same configured Cotengra optimizer.
The default `build_optimizer` returns `ReusableHyperOptimizer`, reusing tree
planning as well. `strip_environment_reuse` exposes cumulative hit/rebuild
counts separately from outer boundary reuse. Adaptive
`reuse_environments=False` bypasses strip reuse; fresh adaptive confirmation
can replace boundary arrays and legitimately invalidate cached partials.

## Earlier validation, before monitored ALS

Used `~/envs/py312`, local source, CPU-only execution, one BLAS/OpenMP/Numba
thread per process, and `LOKY_MAX_CPU_COUNT=2`. The installed upstream audit
from the immediately preceding review was reused; the paper and installed
Quimb ALS internals were additionally inspected for this follow-up.

- Combined PEPS full-update, convergence, environment reuse, gate order,
  safeguard, batching, optimizer, boundary numerics, timing, performance and
  shared BP reduced-update suites: **403 passed**, 66 warnings, 113.43 seconds.
- Added complex64 coverage after that run; row/column strip-cache tests in
  complex64 and complex128: **4 passed**, 33 deselected, 3.90 seconds. Two
  complex128 cases overlap the combined run; counts should not be added as
  distinct tests.
- Default smoke: **94 passed**, two warnings, 34.03 seconds.
- `python -m ruff check src tests`: passed.
- `git diff --check`: passed.
- Cached and fresh reduced norms agree after moving along a strip, in-place
  spectator changes, and changes outside the strip that rebuild boundary MPS.
  Consecutive real column updates register cache hits and return the same
  dense state with strip reuse enabled or bypassed.
- No full repository numerical-suite or CUDA run was performed; no production
  simulation or shared environment was changed.

Detailed temporary logs: `/tmp/pepsy_fix_regression.log`,
`/tmp/pepsy_fix_smoke.log`, `/tmp/pepsy_fix_cache_dtypes.log`.
Owned implementation files: `boundary/_reuse.py`,
`optimizers/peps/_full_update.py`, and `optimizers/peps/optimizer.py` under
`src/pepsy`; regressions are in `test_peps_full_update.py` and
`test_peps_boundary_convergence.py`.

## Native ALS, automatic stopping and 4×4 follow-up

The user's follow-up requested another numerical review, 4×4 TFIM evolution,
native Quimb/Pepsy reuse without moving Torch tensors to NumPy, and clear
automatic stopping. Implemented `monitor_convergence=True` for dense shared
ALS and enabled it in full-update. The default `rtol="auto"` reuses the shared
FIT policy (complex64 1e-5, complex128 1e-9); zero/None requests a fixed budget.
Reports expose complete-sweep counts, normalized residual/change stopping,
solver status and best-candidate status. Quimb's public native solves retain
the same open overlap networks, with QR/LQ regauging between sweeps. Invalid
nonfinite tolerances and unsupported monitoring routes fail explicitly.

Detailed upstream classification, parameters, numerical results and remaining
limits are in the [native ALS / 4×4 evidence note](../docs/development/notes/2026-10-08-full-update-native-als.md).
Five independent state-vector experiments cover real D=2/D=4, cached/fresh
environments, adaptive boundaries and imaginary time. At real t=0.4 the
infidelity against exact Trotter evolution is 1.65e-3 at D=2 and 1.47e-8 at
D=4. Cached/fresh D=2 vectors agree to 4.35e-9 in phase-aligned norm.
Imaginary-time energy decreases and all 48 adaptive boundary checks pass.
The experiments are short-time CPU evidence, not GPU or long-time validation.

Final checks with the same activated environment and thread limits:

- Combined affected PEPS and shared reduced-update suites: **416 passed**,
  66 warnings, 109.73 seconds. Covers bulk NumPy/CPU conversion guards for both
  solvers/dtypes, automatic stopping, fixed-budget opt-out, lazy environments,
  scale-invariant stopping and earlier fixes.
- Default smoke: **94 passed**, two warnings, 29.57 seconds.
- Ruff and `git diff --check`: passed.
- Full repository run attempted, then interrupted after **24 failures,
  184 passes, three skips**. All 24 failed identifiers reproduce on an
  isolated unchanged `HEAD` archive in `test_bp_compression.py` and
  `test_bp_open_series.py` (**24 failures, 42 passes**). They fail BP fixed-point
  preconditions or post-compression convergence; no clean full-suite claim.

Additional owned files: `src/pepsy/bp/reduced_update.py`,
`tests/test_bp_reduced_update.py`, `docs/api/bp.md`, and the evidence note.
Existing unrelated backend/MPS changes remain untouched. Still uncommitted;
nothing staged or pushed. Temporary logs are linked in the evidence note.

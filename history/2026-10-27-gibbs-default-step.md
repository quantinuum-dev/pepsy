# 2026-10-27 — Gibbs MPS beta steps and trajectory

- Scope: set a clear second-order Gibbs preparation step default and assess
  local infidelity reporting; no production simulations requested.
- Branch / baseline commit: `develop` / `aa750d7`.
- Publication: reviewed for the user's requested commit and push on `develop`.
  Earlier validation below predates publication; Git records the final commit.

## What changed

- `GibbsMps.prepare` defaults to thermal increments `beta_step=0.05`, with
  second-order Trotterization. `dt` retains its earlier purification-time
  meaning; `n_steps` takes precedence. Actual `beta_step`, `trotter_step`,
  and `n_steps` expose both units and the chosen count.
- Added `evolve()` to reuse one optimizer and purification, yielding β=0 and
  every five completed steps by default, plus the final beta, for measurements.
  Explicit `measure_every=1` retains per-step output. It stores the latest gate increment
  and completed step count without retaining MPS snapshots automatically.
- Added `local_infidelity`, now automatically measured for non-unitary
  direct/DMRG compression. Dense paths cancel the canonical exterior and
  DMRG reuses its terminal FIT norm; native arrays use full graded overlaps.
  Readout performs no contraction. The unitary ledger is unchanged.
- Reuse the installed gate stream and bound energy operators during a
  trajectory. Added every-N-step and explicit-beta measurement schedules,
  with fusion between yields. Document canonical-center synchronization
  after external live MPS readout and native diagnostic fallback cost.
- Review follow-up: energy readout now passes an isolated copy of the known
  canonical-center metadata to Quimb, avoiding center rediscovery while
  preserving the live arrays. The same `build_optimizer()` and operator
  dictionary are reused across measurement intervals.
- Updated the [API guide](../docs/api/optimizers/gibbs_mps.md),
  [implementation note](../docs/development/notes/gibbs_mps.md), and changelog.
- Added shared local readout in the existing MPS/tree energy classes:
  canonical/environment route selection for MPS, named energy/observable
  batches, one RDM per support, and optional Torch einsum/vmap final
  contractions. Gibbs delegates to the MPS measurement layer and retains
  canonical readout by default. Native arrays use graded fallbacks;
  state-dependent environments are never cached across evolution steps.
- Added a capability-gated, warned exact fallback for older Quimb builds.
  Reconcile global exponent handling across canonical/environment routes.
- Pre-commit review fixed the explicit final-beta measurement case: a float
  endpoint in `measure_at` now retains the supplied beta tensor and gradient.

## Validation

- Earlier, before the stream/operator reuse follow-up:
  `python -m pytest -q -o addopts='' tests/test_gibbs_mps.py
  tests/test_public_api.py tests/test_package_layout.py
  tests/test_quimb_compat.py` → 117 passed,
  2 deprecation warnings.
- Earlier focused rerun after that follow-up:
  `python -m pytest -q -o addopts='' tests/test_gibbs_mps.py` → 45 passed.
- Before the final review: Gibbs, MPS replay/normalization, public API, package layout,
  Quimb compatibility, and sampler suites → 427 passed (six warnings).
- Native non-unitary diagnostic/hopping checks → nine passed, including
  truncated and untruncated graded targets.
- Follow-up Torch schedule gradient checks → four passed; direct sub-MPO
  diagnostic check → one passed.
- Latest review: Gibbs, MPS replay/normalization, public API and package
  layout → 303 passed (five existing warnings). New checks cover the default
  five-step cadence, reuse of measurement metadata/cache without live-state
  mutation, and dense non-unitary fidelity across moving centers, reversed
  supports, one-site filters and repeated normalization.
- Shared-readout implementation: 350 focused checks passed, followed by
  161 final energy/Gibbs/public-API/layout checks. These include CPU/CUDA
  Torch values/gradients, JAX fallback, and native MPS/tree cases.
- Pre-commit rerun: 354 tests passed across energy readout/tree, Gibbs, MPS
  replay/normalization, public API, package layout and factorized-pair MPO.
  The endpoint-gradient regression passes; Ruff and diff whitespace checks pass.
- Pre-commit native non-unitary/hopping selection: nine tests passed.
- The broader repository run was deliberately interrupted during unrelated
  JAX compilation after 572 passes and two skips in 310.55 seconds. This is
  partial coverage, not a full-suite pass.
- `python -m ruff check src tests` and `git diff --check` passed.
- No production simulation was run; the full numerical suite did not complete.

## Finding

- The incoming-norm ratio is invalid for imaginary-time compression. New
  diagnostics use the post-gate target norm; automatic reporting adds
  contraction work. It measures the latest gate/window, not total Gibbs error.
- An earlier no-truncation filter probe had no non-unitary diagnostic and
  gave 0.975 with the incorrect unitary flag. New dense-reference checks
  separate physical norm change from actual compression loss.
- Dense intermediate energy/log-Z and unfused product-formula checks pass.
  Torch preserves β/coefficient gradients across repeated optimizer runs.
  Finite-χ equivalence to fully fused `prepare` is not claimed.
- Existing CPU/GPU supervisors still launch independent per-β preparations;
  using the new trajectory in those runners is a separate change. No runtime
  speedup for the class setup reuse has been measured.
- An exploratory Torch DMRG diagonal-gate probe had NaN gradients both with
  and without the new diagnostics. A nondegenerate Hermitian gate passes;
  the existing degenerate-SVD gradient issue remains outside scope.

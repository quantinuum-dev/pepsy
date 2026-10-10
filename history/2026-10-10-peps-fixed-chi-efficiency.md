# 2026-10-10 — Implement fixed-chi sweep efficiency controls

- Scope: implement performance improvements after the user rejected adaptive
  boundary accuracy. Keep boundary dimensions fixed.
- Baseline: Pepsy `develop` at `4adb5f9`; examples `main` at `84fb44c`.
- Commit scope: user-requested package and downstream commits include this
  implementation and the API review below. No push in this follow-up. Existing
  background runs and notebook outputs remain unchanged.

## Changes

Implemented bounded expression reuse, optional converged-endpoint skipping,
absolute/minimum/patience inner stopping, optional lazy boundary guesses,
and a public single-strip fit. Driver summaries include actual solves,
skipped visits, and expression-cache hits. The examples worker and launcher
support fixed-chi strip or pair-only cleanup between periodic global sweeps;
new controls are snapshotted and recorded in metadata. Legacy manifests send
no new flags to old frozen workers. The extension launcher snapshots its new
configuration helper too.

See the [API guide](../docs/api/optimizers/sweep.md#fixed-chi-performance-controls)
and [measurements/upstream audit](../docs/development/notes/2026-10-10-fixed-chi-sweep-controls.md).
The example command is documented in sibling `FullUpdate/README.md`.

## Validation

- Initial implementation: 154 existing ALS checks and 32 new performance
  checks passed. A broader regression then passed **646 tests**, 88 warnings,
  in 125.01 seconds, including the original native GPU ALS paths.
- Lazy-boundary follow-up: 36 performance tests passed. After the final lazy
  changes, **296 tests passed**, 9 warnings, in 78.74 seconds across performance,
  sweep safeguards, PEPS driver, boundary engines, and public API.
- Two additional joint-gradient strip tests passed, covering both axes.
  Total distinct package coverage: **652 passing tests** (614 existing + 38 new).
- Final downstream run: **19 tests passed**, 73.24 seconds. This includes
  independent trajectories, exact local acceptance, periodic strip/pair cleanup
  across timestep boundaries, deterministic resume, and manifest flags.
- Ruff passed for package `src tests` and examples `FullUpdate`; diff checks
  passed. Full repository suite was not run.
- Mature D=2 gate replay: equivalent fidelity with about 15% less refinement
  time for reuse + skipping + lazy guesses. One full timestep with conservative
  stopping was about 20% faster, differing in true fidelity by 2.67e-7.
  Periodic global cleanup was about 58% faster with a larger accuracy tradeoff.

## Limits

Default solver and work budgets remain unchanged apart from expression reuse.
The new schedule and stopping examples are optional, not evidence of long-time
accuracy at D=3,4,5. Boundary chi never grows through these controls. No stale
state-dependent boundary environment is reused across gates.

## Final API review

The downstream worker and launcher share control defaults and option groups.
Programmatic booleans, tolerance types, and cleanup choices are validated as
strictly as CLI values. Default settings now agree for calls with or without
an argparse namespace. Invalid scopes, gate adjacency, and controls fail before
copying networks or constructing boundaries, including for column L-BFGS.
Added worker-parser round-trip and early-failure regressions. No additional
numerical algorithm or default accuracy change was needed in this review.

Final review checks: **177 package tests passed**, 23 warnings, in 41.66 seconds
(speed controls, ALS, public API, package layout); **32 downstream tests passed**
in 76.11 seconds. Ruff, staged diff checks, and both launcher/worker CLI help
checks passed. Earlier broader results above remain separate; the full suite
was not rerun. Unrelated simulation archives were excluded from staging.

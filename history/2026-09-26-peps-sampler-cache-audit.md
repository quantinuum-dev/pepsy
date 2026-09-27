# 2026-09-26 — PEPS sampler cache audit

- Scope: user's request to check future boundary MPS and within-row/column
  caching for accuracy and efficiency.
- Branch / baseline commit: `develop` / `a13031b`, preserving the preceding
  uncommitted boundary-scaling and rho-repair work.
- Commit status: working-tree changes only; no staging, commit, or push.

## Implemented and checked

- Confirmed future MPS preparation is reused across samples/queries until
  `refresh()`, for both Quimb and DMRG engines.
- Fixed optional dense row-cache prefix underflow using positive rescaling of
  private factors and partial contractions. Rare-prefix log likelihoods now
  remain finite in complex64/complex128.
- Retained the initial-row cache across calls. Later rows remain conditioned
  on each incoming sampled prefix; branching descendants share immutable
  suffixes and maintain separate left prefixes.
- Removed unused retained center networks and terminal prefix contractions.
  Counted persistent first-row storage in the preallocation memory estimate;
  exposed actual builds/reuse and correct early-zero statistics.
- Added lifecycle, exact Born/amplitude, precision/backend, JAX device, Torch
  gradient, rare-branch, zero-branch, and memory-accounting regressions. Updated
  the API docs and changelog. Details and measurements are in the
  [cache audit note](../docs/development/notes/peps_sampler_cache_audit.md).

## Measurements and limitations

- Warm 8×1 four-shot dense-cache calls improved relative to the previous cache:
  grouped 7.79→6.57 ms NumPy and 12.84→11.58 ms Torch CPU; serial 17.42→11.55 ms
  and 30.48→23.79 ms, respectively.
- Stability operations add overhead on tested 2×3/3×3 cases. Default dense-cache
  budget remains zero; the reference route was faster there. No universal
  speedup or GPU throughput claim is made.
- A factored-cache prototype has much smaller held arrays but was 9–10% slower
  than the reference on bounded D=3 probes. It remains a temporary prototype,
  not integrated code. See the note for distinct memory measurement definitions.
- No production job, shared environment, installed dependency, or sibling repo
  was modified. No full repository numerical suite was run. Native Symmray
  sampling remains unsupported.

## Validation

- Focused cache/device checks before removing redundant local-rho rescaling:
  21 passed, 215 deselected, 3 existing Quimb `mode`/`method` warnings.
- Final before/after timing probes assert matching configurations, log proposals,
  and log amplitude magnitudes against the reference within 3e-12.
- Public API/layout: 58 passed, 1 existing failure,
  `test_package_version_matches_installed_distribution` (installed/runtime
  0.4.0 versus checkout metadata 0.5.0).
- Smoke: 162 passed, 1 skipped, 1 failed at the same existing metadata check.
- Final complete sampler suite: **236 passed**, 37 existing Quimb
  `mode`/`method` warnings, 300.73 seconds. NumPy/Torch/JAX CPU, both complex
  precisions; two JAX CPU devices were exposed for placement checks.
- Ruff (`src tests`), new documentation links/catalog entries, and
  `git diff --check`: passed.

The same-session official upstream audit was reused with unchanged versions;
installed Quimb/Cotengra signatures and exponent stripping were inspected for
the cache failure. `apply_patch` again failed at sandbox startup with the
bubblewrap `mountinfo path is not absolute` error. Checked Python replacements
through approved execution were used for tracked-file edits.

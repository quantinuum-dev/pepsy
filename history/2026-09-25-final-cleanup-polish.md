# 2026-09-25 — Internal import cleanup and documentation polish

- Scope: finish the remaining compatibility-import cleanup and polish active
  documentation and installation guidance.
- Branch / baseline: `develop` / `46eb601`. At startup, local and remote
  `main` and `develop` matched this commit and the checkout was clean.
- Commit status: included in the internal-import and documentation cleanup
  commit. The user authorized committing and pushing this work to `develop`.

## Changes

- Migrated the eight remaining internal `tensors.core` imports to
  `tensors.observables` or `tensors.contractions`. This covers fitting,
  boundary sweeps, MPS/MPO/sweep/global optimizers, stabilizer overlap
  diagnostics, and the BP sampler's lazy optimizer construction.
- Kept the public compatibility module and its patch hooks unchanged.
  Internal consumers now bypass its temporary global overrides. Documented
  the instrumentation boundary and added success/failure restoration tests.
- Corrected the tensor ownership guide, which still described implementation
  modules as facades over `core.py`.
- Filled four empty API pages (global optimization, gradient solvers, finite
  differences, and tensor maps) with short guides and runnable examples.
  Removed repeated placeholder footers from the active API pages.
- Replaced remaining active source installation messages that suggested
  registry installation with local-checkout extra commands, preserving
  GitHub-only distribution. Historical release records were left intact.

## Validation

- Package API, import boundaries, contraction dependencies, and legacy hooks:
  **83 passed**. After adding lazy-call coverage, the changed import test and
  six hook tests passed again.
- Focused fidelity, overlap, and global-optimizer checks: **69 passed**,
  871 deselected.
- Full local collection for the import migration: **4,607 passed, 121
  skipped**, 734 warnings, in 328.43 seconds. Skips remain unvalidated paths;
  this is not a claim of complete optional-backend or MPI integration coverage.
- Installation-message follow-up checks (backends, MPI unit contracts,
  contraction dependencies, and compatibility): **136 passed, 1 skipped**.
  Message-only edits were made while the full run was in progress; these
  follow-up checks used the final message text.
- All four new guide examples executed; both solver examples reduced the
  quadratic loss below `0.01`.
- Strict Sphinx build passed with an empty diagnostic log. **8,429** rendered
  local links across **26** changed pages resolved, including fragments.
- AST comparison across all 16 changed runtime files found only import-target
  or text-literal differences. Numerical kernels, defaults, and public
  exports were not edited. No numerical upstream audit was needed.
- Ruff and `git diff --check` passed.

## Boundaries

This removes internal compatibility routing; it is not a measured numerical
speedup or dependency-size reduction. The larger symmetry model/state
extraction remains the separate, deferred refactor identified in the
[simplification review](2026-09-25-simplification-review.md).

The prior tree-guide integration completed at `46eb601`; this supersedes the
pending integration status in the [earlier handoff](2026-09-25-tree-guide-organization.md).
The `v0.5.0` tag and release configuration are unchanged by this cleanup.

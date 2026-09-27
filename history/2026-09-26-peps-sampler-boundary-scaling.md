# 2026-09-26 — PEPS sampler boundary scaling

- Scope: fix the relative-cutoff overflow identified in the sampler audit and
  verify one cached future sweep reused by opposite-direction sampling.
- Branch / baseline commit: `develop` / `a13031b`.
- Commit status: uncommitted working-tree edits. Preserve the earlier audit
  notes and handoff; no staging, commit, push, or production-job changes.

## What changed and why

- `src/pepsy/sampling/peps.py` rescales private Quimb future inputs and
  conditioned ket boundaries before relative-cutoff compression. The future
  sweep uses the existing Quimb norm-equalization option. Absolute cutoff
  semantics, source tensors, physical amplitudes, and row ordering are preserved.
- `tests/test_peps_sampler.py` adds scale, absolute-cutoff, cache lifecycle,
  backend/precision, and analytic-gradient regressions.
- API docs and `CHANGELOG.md` describe the behavior. The dated
  [implementation evidence](../docs/development/notes/peps_sampler_boundary_scaling.md)
  links the pre-fix audit and records the original 5×6 reproducer after the fix.
- The Torch complex64 reproducer improves from ESS/N 0.854099102 to
  0.9999992715; future ranks at rows 0 and 1 are 14 and 15 instead of 1 and 1.

## Validation

- Focused scaling and existing truncation-policy tests: 24 passed.
- Absolute semantics, cache/refresh, and gradient tests: 9 passed.
- Public API and package layout: 58 passed, 1 known environment failure:
  `test_package_version_matches_installed_distribution` sees installed/runtime
  0.4.0 while checkout metadata says 0.5.0. The same failure is recorded in
  [the earlier sync handoff](2026-09-26-pepsy-sync-push.md).
- Ruff (`src tests`): passed.
- Smoke: 162 passed, 1 skipped, 1 failed at the same existing version-metadata
  check. No additional smoke failure was found.
- Complete PEPS sampler suite: 159 passed in 241.90 seconds, with 32 existing
  Quimb `mode`/`method` deprecation warnings. JAX exposed two CPU devices to
  include source-device placement checks.
- New documentation links/catalog entry and `git diff --check`: passed.

## Limits and environment

- CPU validation uses NumPy, Torch, and JAX; complex64 and complex128. Production
  GPU work was left untouched. The entire repository suite was not rerun.
- Raw arbitrary-scale contractions and finite-χ/χ′ support loss remain limits.
  No native batch-axis or new large-D GPU performance claim is made.
- Installed upstream/source evidence from the preceding audit was reused;
  this implementation adopts public APIs without editing dependencies.
- `apply_patch` was attempted but failed before file access because bubblewrap
  reported `mountinfo path is not absolute`. Checked Python replacements under
  approved command execution were used for tracked-file edits.

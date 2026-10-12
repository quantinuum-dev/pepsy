# 2026-09-26 — Correct gauge scaling and add adaptive QR differentiation

- Scope: implement the user's requested gradient and renormalization repairs,
  with complete public-loss regressions in the sibling Gaugy checkout.
- Branch / baseline: `develop` / `74c533e`.
- Commit status: included with the implementation commit; not published.

## Changes

- Shared overflow-safe, exactly compensated gauge scale extraction.
- Opt-in adaptive Torch QR policy; preserve finite native VJPs and warn on
  singular/nonfinite fallback. Preserve the forward policy through backward.
- Added reconstruction, derivative, context restoration, and batch regressions.
- API documentation and changelog updated; existing global defaults preserved.

## Validation

- New scale/QR regressions: 39 passed, including the smallest positive float
  whose RMS rounds to zero with a disabled scale floor.
- Backend plus new regressions: 105 passed, 1 skipped, 11 warnings.
- Earlier gate/backend check after initial implementation: 178 passed, 2 skipped.
- Ruff across `src tests`, CI mypy selection, and `git diff --check` pass.
- Strict documentation build (`sphinx -E -W --keep-going`): passed.
- Final full suite: 4683 passed, 121 skipped, 732 warnings in 322.57 seconds.
  The first run aborted in the native macOS Matplotlib GUI backend; the
  successful run uses `MPLBACKEND=Agg` for headless plotting.
- The first headless run found seven suite-order failures in tests forbidding
  host scalar reads, caused by an active stabilized QR registration. The new
  QR tests now establish and clean up their native policy explicitly. The
  affected QR/stabilizer/tree test group passes: 66 passed, 48 skipped.
- Strict documentation and Ruff checks passed again after the final edits.
- Paired Gaugy full suite: 214 passed, 7 warnings. Its 25 new public-loss
  regressions include actual SU truncation and reference-stream gradients.
- No push: earlier unpublished branch work and remote advancement require a
  separate integration step; no unrelated commits were rewritten or published.

See [detailed evidence and limits](../docs/development/notes/pepo_gradients_scale_2026_09.md).

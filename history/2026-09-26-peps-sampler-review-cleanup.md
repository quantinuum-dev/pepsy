# 2026-09-26 — PEPS sampler API and implementation follow-up

- Scope: finish reviewing the five sampler improvements and fix correctness,
  API, and code-comment gaps requested by the user.
- Branch / baseline: `develop`, `a13031b`.
- Commit status: uncommitted; preserved prior working-tree edits, no staging,
  commits, publication, environment changes, or production-job changes.

## Implemented

- Publish the reusable amplitude cache only after all scaled leaves are ready.
  A failed preparation can now be retried without `refresh()`; regression
  injects a preparation failure and checks the retry against a raw contraction.
- Reject scaled-result field lengths that differ from the configuration count.
  This prevents NumPy broadcasting from silently producing incorrect weights.
  Six regressions cover both probability/amplitude fields and pair components.
- Weight diagnostics convert each scaled field to host only once; retain stable
  normalization and the existing invalid-mass checks.
- Reuse sample-count validation, remove the unused raw scaling helper, and
  clarify cached physical slices, diagnostics lifetimes, memory estimates, and
  raw amplitude reconstruction in docstrings and the handwritten API guide.

## Validation

- Existing sampler validation/result/serial/batch selection: **40 passed**,
  196 deselected.
- Shared sampler consumers and public API/layout: **160 passed, 6 skipped,
  1 failed**. The failure is the existing installed-distribution mismatch:
  `test_package_version_matches_installed_distribution`. This run observed
  installed/runtime `0.4.1` versus checkout `0.5.0` (the previous handoff recorded
  `0.4.0`). No metadata or test assertion was modified to conceal the mismatch.
  CUDA was hidden for these CPU checks; six CUDA/CuPy checks skipped.
- Complete new-feature suite on final code: **37 passed**, six existing Quimb
  `mode`/`method` deprecation warnings, 113.31 seconds. Covers NumPy, Torch, and
  JAX proposals, chunk replay, extreme scales, cache invalidation/retry,
  diagnostic graph retention, truncation, and malformed result lengths.
- Combined follow-up checks: **237 passed, 6 skipped, 1 known metadata failure**.
- Ruff (`src tests` and the sampler benchmark) and `git diff --check`: passed.
- Earlier full sampling-domain evidence and GPU measurements remain in the
  [efficiency handoff](2026-09-26-peps-sampler-efficiency.md); they were not
  repeated or represented as validation of this follow-up's final code.

## Limits

No full repository numerical suite or new GPU benchmark was run. Native
batch-axis GPU sampling, Symmray sampling, and roughening-runner integration
remain separate work; the implemented five improvements do not imply those
features. Finite-cap proposal support and exact-contraction cost limitations
remain documented in the [API guide](../docs/api/sampling/samplers.md).

`apply_patch` again failed during sandbox initialization with the bubblewrap
mountinfo error. Tracked edits used exact-match checked replacements instead.

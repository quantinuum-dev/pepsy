# 2026-10-23 — Align branches and version metadata

- Scope: synchronize `main` and `develop`, the changelog, and package version.
- Baseline: clean `develop` at `f025438`; `origin/develop` at `b7e7dd2`.
  Remote `main` was 33 commits behind local `develop`. Local `main` at
  `7ae29f6` also contained three older merge commits absent from `develop`.
- Commit status: ancestry merge `7e257d1` is committed. This entry accompanies
  the metadata commit; fast-forwarding `main` and pushing both branches are
  the final publication steps, pending when this record was written.

## Changes

Preserved local `main` ancestry by merging it into `develop`. The merge tree
exactly matches baseline `f025438`: no implementation or test changes.
The publication can fast-forward both remote branches without rewriting history.

Set the source version and README to **0.6.0**. Place all existing Unreleased
entries under the dated 0.6.0 heading, preserving their text and leaving an
empty Unreleased section. The minor increment reflects accumulated features
and documented changes to defaults under the pre-1.0 stability policy.

Add a migration overview and point installation to `main`. Distinguish current
source metadata from the existing `v0.5.0` release tag; no new tag or registry
publication is part of this task. Refresh the selected environment's editable
Pepsy installation without installing or upgrading dependencies.

## Validation

- Public API/package-layout suites: **55 passed**, two alias deprecation warnings.
- Built 0.6.0 sdist and wheel; strict Twine metadata checks passed for both.
- All **266 Python modules** in the wheel match checkout source bytes.
- Installed the wheel without dependencies in a fresh temporary environment;
  package import, experimental discovery, and runtime/distribution version
  agreement passed. The editable installation also reports 0.6.0.
- Project metadata, README, and first versioned changelog heading agree.
  Existing changelog entries are byte-for-byte preserved.
- Relative links/anchors in edited Markdown and `git diff --check` passed.

No numerical behavior changed. The full numerical suite was not rerun;
[earlier numerical validation](2026-10-21-qr-periodic-boundary-fixes.md) and
[later gauge-QR checks](2026-10-22-peps-gauge-qr.md) retain their own scope.

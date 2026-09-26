# 2026-09-25 — Tree guide organization and cleanup integration

- Scope: commit the tested cleanup, reorganize the tree guide, and synchronize
  `main` from `develop` after CI passes, without a pull request.
- Branch / baseline: `develop` / `cab817e` (documentation and MPS reporting
  cleanup, pushed before this documentation change).
- Commit status: this record accompanies the tree guide organization commit.

## Changes

- Replaced the 2,386-line tree API page with a short overview and topic links.
  Six guides cover layout, states, operators, replay, variational fitting,
  and readout. The longest guide is about 550 lines.
- Kept every original section heading on the overview as a link to its new
  location. Moved all 35 code blocks unchanged, with the technical guidance
  retained and cross-page navigation added.
- Preserved GitHub-only distribution and the existing `v0.5.0` tag. This
  documentation change does not change runtime behavior.

## Validation

- Strict Sphinx HTML build passed with no diagnostics.
- Rendered local links and all 17 original section anchors passed checks.
- A code-block comparison confirms all 35 original examples are unchanged.
- `git diff --check` passed. No numerical tests were added or rerun for the
  documentation split; the earlier focused cleanup checks are recorded in
  [the reporting handoff](2026-09-25-reporting-extraction.md).

## Integration

The cleanup was pushed to `develop` as `cab817e`; hosted CI was still running
when this entry was prepared. Synchronizing `main` remains gated on successful
CI for the final documentation commit. Check Git and Actions for the final
integration result; this entry does not claim a completed remote merge.

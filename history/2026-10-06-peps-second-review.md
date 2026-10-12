# 2026-10-06 — Second PEPS review

- Scope: user requested another review after the delegated norm fixes.
- Branch / baseline: `develop`, `8f7c896`.
- Commit status: new review notes only this turn; existing source/test edits
  remain uncommitted. Nothing committed or pushed.

## Findings

The [report](../docs/development/notes/2026-10-06-peps-second-review.md)
confirms the previous three fixes and records additional findings: optional
global mode duplicates final normalization and does not honor the final
normalization opt-out; a substantially negative coarse sweep diagnostic can
abort before accurate outer acceptance; stored retry options can leak to the
boundary metric. The report separates a robustness limitation from flag and
configuration defects and includes concrete reproductions/workarounds.

## Validation

Fresh tests: **166 passed**, 36 warnings, 23.02 seconds in batching/PEPS suites.
Additional real global and coarse-boundary probes confirmed the findings.
Review links and `git diff --check` passed. No production or test edits in
this turn, no full suite/GPU run. Concurrent solver changes were preserved.
Proposed fixes are not yet implemented; see the report before claiming a
complete all-path audit.

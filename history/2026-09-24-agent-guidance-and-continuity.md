# 2026-09-24 — Agent guidance and session continuity

- Scope: clarify Pepsy's agent guidance, assess its health, and preserve the
  history of recent work without treating historical notes as instructions.
- Branch / baseline commit: `develop`, `64f7e1f`.
- Commit status: this handoff and the recent guidance and implementation
  changes are uncommitted. Nothing was staged, committed, or published.

## What changed

- The preceding reorganization reduced the root guide from 308 to 162 lines,
  moved detailed invariants to their owning skills, and added
  [shared numerical contracts](../.github/skills/pepsy-maintainer/references/numerical-contracts.md).
  The current follow-up adds a short journal startup/handoff rule to that guide.
- Reconnected [AGENTS.md](../AGENTS.md) to the existing [journal](README.md).
  Its guide distinguishes Git history, uncommitted work, historical evidence,
  and current policy. Resumed tasks can finish the same draft handoff.
- Corrected the journal's stale `learning/` reference and assumption that root
  `PLAN.md` is the package roadmap. That file now describes a deferred native
  PEPO design. No deferred implementation was started.
- Removed the old machine-specific environment example from the shared README.
  The ignored device-local override remains unchanged.

## History and health findings

- Before this entry, `history/` contained 23 dated handoffs, most recently
  September 16. They remain intact. Git also contains 31 commits in the
  followed history of `AGENTS.md`, starting May 30, 2026.
- Skill governance was established in `dcf7739` on July 27. The latest
  committed root-guide change was `1ccf3ed` on September 16. The September 24
  reorganization is still a working-tree change, not a new Git checkpoint.
- Guidance ownership is clearer, but history discovery had been missing from
  the root guide. The older journal checklist could also misdirect agents to
  an unrelated deferred plan; this follow-up corrects both problems.
- The catalog validator exists and is runnable locally. The inspected CI
  workflows do not currently run it. Automated guidance validation is a
  proposed improvement, not an implemented CI feature.

## Recent implementation handoff

The preceding user-authorized task implemented measurement compatibility and
independent intermediate/final compression settings (priorities 1 and 2).
See the [adoption assessment](../docs/development/notes/quimb_symmray_opportunities_2026_09.md#implementation-follow-up-priorities-1-and-2)
for exact scope, dependency versions, tests, and limitations. Priorities 3–6
remain deferred. Native numerical safeguards remain in place.

Earlier checks on September 24: 136 focused tests passed, including 49 new
adoption regressions; the older-stack adoption check passed 48 with 1 skipped.
The full run recorded 4,566 passed, 121 skipped, and 13 failed. These failures
remain documented in the assessment; the package does not have a clean full
suite. These numerical results were not rerun for this documentation review.

## Suggested next step (subject to current task scope)

Review the existing guidance diff separately from the numerical implementation
diff. Add the catalog validator to CI if that work is requested. Investigate
the recorded full-suite failures as a separate numerical task. Any commit
should include the intended new reference files and exclude the local override.

## Validation of this documentation follow-up

- `python .github/skills/pepsy-maintainer/scripts/validate_catalog.py` passed:
  all 12 skill packages are present in the catalog and manifest.
- Checked 36 local Markdown links in the changed guides and this handoff:
  all targets exist.
- Confirmed the 23 older dated journal entries are unchanged and
  `AGENTS.override.md` remains untracked and locally excluded.
- `git diff --check` passed. No package implementation changed in this
  follow-up, so numerical tests were not rerun.

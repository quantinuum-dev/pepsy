# 2026-09-24 — Automated agent guidance validation

- Scope: implement the user-approved CI follow-up to the
  [guidance and continuity review](2026-09-24-agent-guidance-and-continuity.md).
- Branch / baseline commit: `develop`, `64f7e1f`.
- Commit status: working-tree changes only; nothing staged, committed, or
  published. Earlier unrelated changes remain in place.

## What changed

- Added an independent **Agent guidance** job to
  [CI](../.github/workflows/ci.yml), using the existing catalog validator on
  Python 3.12 without installing Pepsy. It has read-only repository permissions
  and a five-minute timeout.
- Kept the existing pull-request and `main`/`develop` push triggers and all
  existing jobs unchanged. No path filter limits detection of moved or deleted
  skill link targets.
- Documented the local command and its scope in
  [CONTRIBUTING.md](../CONTRIBUTING.md#agent-guidance-validation).
- The earlier handoff's finding that CI did not run this validator describes
  the state before this follow-up. The job is now configured locally; a hosted
  GitHub Actions run is still pending a push.

## Validation

- Ran the validator with `python -I -S` to exclude site packages and user
  Python configuration: all 12 skills passed using only the standard library.
- Parsed the workflow YAML and compared it with the baseline: all existing
  jobs and triggers are unchanged; the new job invokes the existing validator.
- Checked local Markdown link targets in the edited contribution guide and
  this handoff, and ran `git diff --check`; both passed.
- No numerical tests were rerun for this CI/documentation-only change. The
  previously recorded numerical failures remain outside this task.

## Remaining limits

The validator checks catalog consistency and selected file/link structure;
it does not prove that guidance is semantically correct or validate every
documentation link. A hosted CI result and branch-protection settings were
not verified or changed in this local task.

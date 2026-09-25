# history/ — agent session hand-off log

This folder carries context **between** coding sessions. After substantive
work, append a dated handoff so the next agent or person can resume without
repeating the investigation. Existing entries are historical evidence, not
current instructions or permission to start their suggested next steps.

Git records committed file changes; this journal explains decisions, checks,
and unfinished work. A journal entry does not commit the work it describes.
Uncommitted entries and code are absent from Git's committed history.

To inspect the instruction history from the repository root:

```bash
git log --follow --oneline -- AGENTS.md
git log --oneline -- .github/skills history
git diff -- AGENTS.md .github/skills history
git diff --cached -- AGENTS.md .github/skills history
```

## Conventions

- One file per session, named `YYYY-MM-DD-<short-slug>.md`
  (e.g. `2026-06-22-bp-scaffolding.md`). Multiple sessions in a day get
  `-a`, `-b`, … suffixes.
- Keep entries short and factual. Link to the files/tests you touched and
  task-relevant plans or detailed evidence. Do not copy entire test logs.
- Preserve completed entries; append corrections in a new entry and link the
  older one. When resuming the same unfinished task, update its draft handoff
  instead of creating a new entry for every interruption.
- State the branch and baseline commit separately from commit status. Record
  what is implemented, proposed, unverified, or blocked.
- Label earlier validation with its date/scope. Passing focused tests does not
  establish a clean full suite. Temporary logs can disappear, so retain the
  important result and failure names in tracked documentation.
- Put durable policy in [AGENTS.md](../AGENTS.md) or the owning
  [domain skill](../.github/skills/README.md), and public behavior in
  [API documentation](../docs/api/). This journal records what happened.

## Start-of-session checklist

1. Follow [AGENTS.md](../AGENTS.md), including the local environment override
   and working-tree check.
2. Read the newest one to three entries relevant to the current task. Search
   older entries when a specific decision needs investigation.
3. Consult a plan only when relevant. The broader
   [project roadmap](../docs/development/plans/project.md) contains historical
   and proposed work; root [PLAN.md](../PLAN.md) currently describes a deferred
   native fermionic PEPO design, not the status of the whole package.

## End-of-session checklist

1. For substantive work, create `history/YYYY-MM-DD-<slug>.md` from the template
   below, or finish the draft for the same resumed task. A read-only answer
   with no durable finding does not require a journal entry.
2. Record the authorized scope, changes, validation, unresolved issues, and
   whether work is committed, staged, or still in the working tree.
3. Link the most useful evidence and proposed next step. Update a task's plan
   only when its actual status changed; do not activate deferred work.

## Entry template

```markdown
# <date> — <title>

- Scope: <user-authorized task>
- Branch / baseline commit: <branch and ref>
- Commit status: <uncommitted, staged, or committed with ref>

## What changed
- ...

## Why
- ...

## How it was validated
- `pytest -q tests/...` → <result>

## Decisions / findings
- ...

## Suggested next step (subject to current task scope)
- ...

## Open questions / blockers
- ...
```

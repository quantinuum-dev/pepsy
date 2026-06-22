# history/ — agent session hand-off log

This folder carries context **between** coding sessions. When an agent finishes
a session, it appends a dated entry here; when the next agent (or person) starts,
they read the most recent entries to resume without re-discovering everything.

## Conventions

- One file per session, named `YYYY-MM-DD-<short-slug>.md`
  (e.g. `2026-06-22-bp-scaffolding.md`). Multiple sessions in a day get
  `-a`, `-b`, … suffixes.
- Keep entries short and factual. Link to `PLAN.md` milestones and to the
  files/tests you touched.
- Never delete old entries; append new ones. History is read-only context.
- Put durable, repo-wide facts (conventions, build commands) in `learning/` or
  `AGENTS.md` instead — `history/` is for *what happened and what's next*.

## Start-of-session checklist

1. Read `PLAN.md` (status + current milestone).
2. Read the newest 1–3 files in `history/`.
3. Confirm working tree state with `git status --short`.

## End-of-session checklist

1. Create a new `history/YYYY-MM-DD-<slug>.md` from the template below.
2. Update `PLAN.md` *Status* / milestone notes if anything moved.
3. Record validation (commands run + result) and the single most useful
   "next step".

## Entry template

```markdown
# <date> — <title>

- Milestone: <e.g. M1 — quimb BP wrapper>
- Branch / commit: <ref>

## What changed
- ...

## Why
- ...

## How it was validated
- `pytest -q tests/...` → <result>

## Decisions / findings
- ...

## Next step (do this first next time)
- ...

## Open questions / blockers
- ...
```

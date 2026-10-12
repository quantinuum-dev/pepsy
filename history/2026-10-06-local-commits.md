# 2026-10-06 — Commit current Pepsy and examples work

- Scope: user requested committing the current work in `pepsy_examples` and
  `pepsy`. No push requested or performed.
- Baselines: Pepsy `develop` at `7cb1044`; examples `main` at `fbdacf1`.
- Examples committed as `7055290` (Align roughening PEPS modes and refresh
  transferred-result plots). Pepsy's solver fixes and session handoffs are
  included in the commit containing this entry; its hash is available in Git.

## Included and excluded

Examples includes the current PEPS full-update defaults and SU/sweep CLI
clarification, plotting helpers, retained notebook outputs, transferred-run
selection, optional wall-module discovery, tests, and documentation.
Seven transferred simulation directories remain untracked and unchanged.

Pepsy includes native solver/backend guards and JAX final/best-state
bookkeeping fixes, solver docs/tests/changelog, the status ledger, and the
five existing review/implementation handoffs. Earlier statements that work
was uncommitted describe those sessions; this entry records the later commit.
No sibling Gaugy changes are part of these commits.

## Validation

New pre-commit checks in the shared Python 3.12 environment:

- Pepsy gradient solver suites: **78 passed**.
- Examples plot-helper and PEPS norm plotting suites: **100 passed**, one
  empty-legend warning.
- Pepsy-wide Ruff and changed examples Python-file Ruff: passed.
- Both modified notebook schemas and whitespace checks: passed.
- Explicit staging lists excluded generated archives and device-local files.

Earlier checks in this active task remain separate evidence: **25** full-update
adapter tests, **179** Pepsy PEPS tests, and **5** mode-separation checks passed.
The broader examples runner selection had **402 passed, 4 baseline failures**
in bubble/MPS mode compatibility; benchmark-wide Ruff retained six existing
effective-model issues. See the
[PEPS defaults handoff](2026-10-06-roughening-peps-defaults.md).
No full-suite or GPU validation is claimed.

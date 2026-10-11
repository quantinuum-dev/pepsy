# 2026-10-11 — Concise mathematical Markdown and planning routes

- Scope: user-requested writing policy and plan/skill organization across packages/examples.
- Baseline: Pepsy `develop` / `4adb5f90`; examples `main` / `fbdacf1`.
- Commit status: working-tree documentation/metadata only; no commit or push.

## Changes

- Added the [Markdown policy](../docs/development/markdown_policy.md): short
  explanations, precise mathematical definitions and colored important terms.
  Root instructions, development docs and the maintainer skill/catalog route to it.
- Organized the [plan index](../docs/development/plans/README.md) by current
  behavior, validation, domain skills and historical/deferred designs.
  Root [plan.md](../plan.md) keeps its deferred native fermionic PEPO scope.
- Updated [skill policy](../.github/skills/SKILL_POLICY.md) and the
  [agent upload manifest](../.github/skills/agent-bundle.yaml) with writing and
  planning references. Existing domain skills keep their numerical ownership.
- Extended the examples' [documentation conventions](../../pepsy_examples/docs/COMMENTING.md#concise-mathematical-markdown)
  and linked their [plan](../../pepsy_examples/PLAN.md), agent guides and index
  to the policy and existing package skills. Validation copies preserve source outputs.
- Gaugy's [handoff](../../gaugy/history/2026-10-11-markdown-plan-policy.md)
  records its compact roadmap, archived earlier design and examples policy.

## Validation

- Installed skill-creator metadata check passes for `pepsy-maintainer`;
  `validate_catalog.py` passes for all 12 skills and upload references.
- Cross-repository changed-page local link/anchor and filename-case checks pass.
  Display-math delimiters/braces and all four repositories' whitespace checks pass.
- Changed-file audit confirms only Markdown, documentation tooling and agent
  metadata changed. No numerical source/tests or notebooks changed.

The device-local override is untouched. No numerical tests, notebook execution,
benchmarks, remote-link checks, commit or publication were needed or performed.

## Publication follow-up

The user subsequently authorized commit and push. The
[cross-repository publication record](../../gaugy/history/2026-10-11-documentation-publication.md)
records the fresh checks and confirmed publication results; the initial
uncommitted baseline above remains historical evidence.

# Pepsy skill policy

This file is the source of truth for designing and maintaining the local
Pepsy skills under `.github/skills/`. It governs the skill catalog itself; it
does not replace domain-specific invariants in an individual `SKILL.md`.

## Design principles

- Keep one skill focused on one reusable subsystem or workflow. Do not create
  a skill for a single class, one-off investigation, or information already
  covered by an existing skill.
- Keep repository-wide rules in `AGENTS.md` and cross-cutting routing in
  `pepsy-maintainer/SKILL.md`. Keep numerical and API invariants in the owning
  domain skill.
- Prefer composition over duplication. A cross-domain skill may point to the
  owning skills; it must not copy their invariants and let the copies drift.
- Optimize for progressive disclosure: concise `SKILL.md` first, one-level
  `references/` only for details needed by a subset of tasks.

## Package contract

Every skill is a flat directory with this shape:

```text
.github/skills/<name>/
├── SKILL.md                 # required instructions and trigger metadata
├── agents/openai.yaml       # required local UI metadata
├── references/              # optional, direct supporting material
└── scripts/                 # optional, deterministic local helpers
```

Required rules:

- Use lowercase hyphen-case for `<name>`, matching the `name` frontmatter.
- `SKILL.md` frontmatter contains only `name` and `description`. Put all
  trigger conditions in `description`; do not hide them in a body section.
- Keep `SKILL.md` under 500 lines. Move detailed method notes, API maps, and
  variant-specific guidance to direct `references/` files.
- Keep references one level deep and link them from `SKILL.md`. Do not add
  README, CHANGELOG, installation, or quick-reference files inside a skill.
- Keep `agents/openai.yaml` synchronized with the skill name and purpose.
- Add every skill to `.github/skills/README.md` and
  `.github/skills/agent-bundle.yaml` with the correct role and upload files.

## How to use skills

1. Read the repository `AGENTS.md` and inspect Git status before editing.
2. Classify the request as cross-cutting or domain-specific.
3. For cross-cutting work, read `pepsy-maintainer/SKILL.md`, then load the
   smallest set of domain skills that own the affected code or invariants.
4. For a focused task, read only the matching domain skill and its direct
   references. Do not load the full catalog by default.
5. If two skills overlap, identify the primary owner and use the other only
   for its explicit interface or invariant. Report the selected skills.
6. Read the closest source, tests, and API docs after the skill guidance.
7. Validate the skill/catalog changes separately from package behavior; run
   focused package tests when implementation behavior also changes.

## Adding a skill

Add a skill only when all of these are true:

- The subsystem has a stable public or implementation boundary.
- Users will ask for it repeatedly and its correct workflow is not obvious
  from the source alone.
- The skill has clear trigger language and a clear owner namespace.
- Its rules cannot be expressed more cleanly by extending an existing skill.

Use this sequence:

1. Search existing skills, `AGENTS.md`, docs, and source for overlap.
2. Choose a short hyphen-case name and define concrete trigger phrases.
3. Create the package with `SKILL.md` and `agents/openai.yaml`; add
   `references/` or `scripts/` only when needed.
4. Write the smallest reliable workflow, including boundaries, invariants,
   source/test paths, and validation commands.
5. Add the skill to the catalog and upload manifest. Update the maintainer
   router only if the new routing is not already covered.
6. Run `quick_validate.py` for the new skill and
   `python .github/skills/pepsy-maintainer/scripts/validate_catalog.py`.
7. Forward-test a representative request. If implementation behavior changed,
   run the owning focused tests and Ruff as required by `AGENTS.md`.

## Updating, deprecating, and removing

- Update the skill in the same change as the code/API workflow it documents.
  Remove stale claims; do not preserve historical behavior as active guidance.
- When a subsystem moves, update source links, tests, namespace examples,
  catalog entries, and the upload manifest together.
- Before deprecating a skill, search the repository and agent bundle for its
  name, links, and trigger phrases. Add a replacement route to the maintainer
  skill and catalog before removing it.
- Remove a skill only after its replacement is documented, references are
  migrated, and validation passes. Never delete a skill merely because it is
  currently unused in one task.
- Do not silently merge two skills. Keep the clearer owner, move unique
  guidance deliberately, then remove duplicate references and update routing.

## Quality gate

For every skill/catalog change, require:

```bash
source ~/envs/py312/bin/activate
python /home/reza.haghshenas@quantinuum.com/.codex/skills/.system/skill-creator/scripts/quick_validate.py .github/skills/<name>
python .github/skills/pepsy-maintainer/scripts/validate_catalog.py
git diff --check
```

For a skill that governs implementation behavior, also run the closest
focused test and `python -m ruff check src tests`. For cross-cutting changes,
run the full suite according to `AGENTS.md`, or explicitly report why it was
not run.

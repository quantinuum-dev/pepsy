# 2026-10-12 — Clearer Markdown ownership and planning

- Scope: user-requested professional, concise documentation and removal of
  competing `plan.md`/`PLAN.md` routes.
- Branch / baseline: `develop` / `bf83def`.
- Commit status: local documentation/agent-metadata edits; not committed or pushed.

## Changes

- Made the [planning index](../docs/development/plans/README.md) the single
  planning entry point. Moved root `plan.md` to the
  [native fermionic PEPO proposal](../docs/development/plans/native_fermionic_pepo.md),
  preserving the deferred design. Removed the uppercase/lowercase root ambiguity.
- Moved root `policy.md` to the [code-change policy](../docs/development/code_policy.md).
  Its instructions are preserved; agent/contributor guidance links the owner.
- Shortened README from 90 to 72 lines and CONTRIBUTING from 168 to 63.
  [Test profiles and CI](../docs/development/testing.md) preserve the complete
  prior testing/CI sections; setup remains in the installation guide.
- Replaced the 724-line status ledger with a 103-line
  [current summary](../docs/development/cluster_optimization_status.md), retaining
  old section anchors and the [complete dated archive](../docs/development/notes/2026-10-12-cluster-status-archive.md).
- Added a [document ownership map](../docs/development/README.md#document-ownership)
  and synchronized the maintainer skill/catalog/upload references.
  Existing journal prose is unchanged; one relocated link destination was updated.

## Validation and limits

Full Markdown/local-link audit: no new missing files, anchors or case errors.
Changed pages have zero local-link errors. Two pre-existing errors are repaired:
the root `PLAN.md` casing and the historical contributor-validation anchor.
All original entry-point anchors, the full status body, testing/CI instructions,
deferred proposal and code-policy instructions pass preservation comparisons.
Skill metadata, the 12-skill catalog/upload validator and whitespace checks pass.

The broad scan still finds 29 pre-existing references to retired examples or
benchmarks in dated notes/journals. They are outside the reorganized pages;
their historical claims were preserved. No remote-link or visual-rendering
check, numerical suite or notebook execution was needed or performed.
Numerical source, tests, dependencies, saved outputs and the local override
are unchanged. This work adds no implemented numerical capability.

# Contributing to Pepsy

Use Python 3.12 or newer and the environment selected by the session or
device-local override. Read [AGENTS.md](AGENTS.md) for repository scope and
[installation](docs/installation.md) for editable installs and optional extras.

## Set up and run the smoke tests

Install the development profile, activate its environment, and run from the
repository root:

```bash
python -m pytest -q
```

This runs the small `smoke` profile: public API, imports and package layout.
It does not establish full numerical or optional-backend coverage.

## Choose a test scope

[Test profiles and CI](docs/development/testing.md#choose-a-test-scope) owns
the full, core, optional, domain, MPI and device-specific commands. Start with
the closest deterministic check and broaden when the change requires it.
Keep backend, dtype, seed and tolerance assumptions explicit.

## What CI checks

Push/PR CI checks the smoke profile, Ruff, selected types, agent guidance and
package builds. Nightly checks the full suite, coverage and MPI. See the
[CI scope](docs/development/testing.md#what-ci-checks) for exact commands,
profiles and runtime limits.

## Keep changes focused

- Put behavior in its [owning namespace](docs/development/package_layout.md).
  Preserve public imports and the [compatibility window](docs/stability.md).
- Follow the [code-change policy](docs/development/code_policy.md) for rationale
  comments, regressions and evidence of behavior changes.
- Update the owning API guide and closest tests together. Record measured
  results and unfinished work in the [session journal](history/README.md).
- Share MPS/tree reference builders in `_mps_test_helpers.py` and
  `_tree_test_helpers.py`. Mark optional-backend, differential and stress cases
  as `integration` or `slow`.
- Preserve notebook outputs, runtime data and unrelated edits. Keep generated
  artifacts and local environment files out of commits.

Use the [documentation ownership map](docs/development/README.md#document-ownership)
and [Markdown policy](docs/development/markdown_policy.md) when writing guides.
The [plan index](docs/development/plans/README.md) is the single planning entry point.

<a id="agent-guidance-validation"></a>

## Validate agent guidance

After skill, catalog or linked-reference changes, run:

```bash
python .github/skills/pepsy-maintainer/scripts/validate_catalog.py
```

Follow [SKILL_POLICY.md](.github/skills/SKILL_POLICY.md) for individual metadata
validation and catalog/upload synchronization. Finish with local link checks
and `git diff --check`; numerical suites are required when behavior changes.

# 2026-09-25 — Prepare the develop-to-main release review

- Scope: review accumulated compatibility changes, consolidate the unreleased
  changelog, update migration guidance, and prepare a draft `develop` → `main`
  pull request. No merge, version selection, tag, or release was requested.
- Branch / baseline: `develop` / `0a60252`; target `main` / `e652197`.
- Commit status: documentation changes accompany the preparation commit.

## Changes and findings

- Consolidated the unreleased changelog by final behavior, preserving all
  released sections verbatim. The original detailed record remains accessible
  through its commit link in the
  [release review](../docs/development/notes/release_readiness_2026_09.md).
- Expanded the [migration guide](../docs/development/api-migration.md) for
  Python/dependency requirements, direct MPS defaults, removed SU/permutation
  spellings, mixed FIT, Hamiltonian construction, tree traversal/rank changes,
  diagnostic opt-ins, and native backend restrictions.
- Corrected superseded release descriptions of the Python 3.11 symmetry-extra
  baseline and three-virtual-bond TreePEPS limit.
- `develop` is 105 commits ahead and 3 behind `main` at these baselines.
  A synthetic merge is conflict-free and preserves main's direct boundary
  aliases. No branch merge was performed. Version metadata remains `0.4.1`;
  the PR is a draft for review and an explicit later version decision.

## Validation

- Exported synthetic merge tree `c2cc3f9a9198348fc4ac746751b1511e3278141b`
  into a temporary directory and ran the selected environment against its
  source and tests: **231 passed, 15 deselected**, with nine warnings.
  Selection: public API and package layout suites; existing MPS default,
  removed-mode, permutation, and mixed-policy contracts; Hamiltonian default
  and compatibility contracts; boundary-input tests except the two direct
  compressor parameter families. This is focused merge validation, not a new
  full-suite run.
- Confirmed main's boundary alias object identity and both package-facade
  deprecation warnings in the synthetic merge. The initial standalone probe
  needed path normalization for macOS's `/tmp` → `/private/tmp` mapping;
  this was a probe assertion issue, not a package failure.
- Documentation HTML build completed with exit 0 using temporary documentation
  dependencies; the shared environment was unchanged. The first attempt
  lacked AutoAPI/Furo, so the declared docs dependencies were installed under
  `/tmp`. Fixed the new review's handoff link for the generated site.
- Compared completed baseline and final Sphinx logs after normalizing checkout
  paths/line numbers: no new diagnostics. Existing warnings and a generated
  `mpo_product` API `Unexpected indentation` error remain; Sphinx exits 0
  without strict warning enforcement. This is not a warning-free docs build.
- All 44 initially checked local documentation links and Markdown fences
  passed; the handoff link was subsequently changed to its rendered GitHub
  URL. Released changelog sections are byte-identical, unreleased categories
  are unique, and `git diff --check` passed.
- Hosted full-suite evidence remains
  [run 36149515229](https://github.com/quantinuum-dev/pepsy/actions/runs/36149515229)
  at `d55e10c`, recorded by `0a60252`. All eight jobs passed there. New PR
  merge CI is separate evidence and must be checked before merging.

## Remaining release decisions

- Review the draft PR and its merge CI, select the next version, then align
  package metadata, README, and dated release notes before building/tagging.
- No numerical code, dependency requirement, test, or workflow was changed
  during this preparation. Hardware and dependency-combination limitations
  remain as recorded in the dependency audit.

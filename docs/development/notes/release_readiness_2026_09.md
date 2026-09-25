# Release-readiness review — September 25, 2026

Scope: prepare the accumulated `develop` changes for a draft pull request to
`main`. This is a compatibility and release-documentation review, not a new
line-by-line numerical audit of every changed subsystem.

## Branch scope and merge behavior

- Reviewed `develop` at `0a60252` against `main` at `e652197`.
  The branches have 105 and 3 unique commits respectively. The development
  changes span 357 files relative to their common ancestor, `d94fb8d`.
- Two main-only commits are merge commits. The remaining change,
  `ef0bc0b`, makes the boundary leaf-module `normalize` and `infidelity`
  compatibility names direct aliases of their canonical functions.
- A synthetic merge has no conflicts and preserves those aliases. Its only
  source difference from `develop` is that existing main-side cleanup in
  `src/pepsy/boundary/metrics.py`. Neither branch was merged during review.
- The release changelog also includes unreleased features already on `main`;
  its scope is broader than the pull request's branch delta. Historical
  released sections beginning at 0.4.1 are preserved unchanged.

## Compatibility findings

| Area | Verified implementation and migration consequence |
| --- | --- |
| Environment | `pyproject.toml` requires Python 3.12+, raises core/optional floors, and bounds NetKet's JAX dependency. Regenerate environment locks for the selected extras. |
| MPS replay | The constructor defaults to `direct`. Explicit `dmrg` retains variational replay, with the current documented FIT schedule. |
| Removed MPS modes | `su`, the `routing` keyword, and composed permutation solver names are rejected by existing regression tests. Dedicated simple-update APIs and standalone permutation SVD remain available. |
| Mixed replay | One-site FIT with a direct guess is fixed policy; other block/initialization choices belong to ordinary DMRG. |
| Hamiltonian construction | All four `ham_tn.to_*` builders default to term-by-term compression. Explicit auto/automaton policies and disabled numerical compression remain available. |
| Tree algorithms | Automatic FIT path/branch traversal, successive environments, and live-rank compression can change finite-rank/seeded results. TreePEPS permits four virtual bonds per site. |
| Diagnostics | Runtime scans, timing, and MPI summaries have explicit opt-in controls. A successful run does not imply these diagnostics were collected. |
| Imports and aliases | Canonical namespace and stabilizer naming updates retain the documented deprecated import aliases. This does not preserve removed mode spellings. |

The [migration guide](../api-migration.md) now covers these actions, logical
ordering after permutation replay, and native symmetry restrictions. The
summary changelog removes repeated categories and superseded intermediate
descriptions, including the old Python 3.11 symmetry-extra wording and
three-virtual-bond TreePEPS limit. Detailed development entries remain in
[the pre-consolidation changelog](https://github.com/quantinuum-dev/pepsy/blob/0a60252/CHANGELOG.md),
the domain API pages, and dated session records.

## Validation evidence and limits

- [Hosted run 36149515229](https://github.com/quantinuum-dev/pepsy/actions/runs/36149515229)
  passed all eight jobs at repair commit `d55e10c`: core minimum versions,
  full extended tests with the 60% coverage gate, docs, package, type checks,
  agent guidance, and MPI with two and three ranks. `0a60252` only records
  that result. This run tested `develop`, not the eventual PR merge commit.
- Dependency-floor probes and their untested combinations are recorded in
  the [dependency audit](dependency_minimums_2026_09.md). Hosted success does
  not establish every optional-version combination or accelerator path.
- New review checks and the synthetic-merge results are recorded in the
  [session handoff](https://github.com/quantinuum-dev/pepsy/blob/develop/history/2026-09-25-release-readiness.md).
  The review changes documentation only; numerical code, requirements, tests,
  workflows, and package version are unchanged.
- The initial local HTML builds retained baseline warnings and a generated
  `mpo_product` API formatting error under the non-strict build. The subsequent
  [documentation cleanup](https://github.com/quantinuum-dev/pepsy/blob/develop/history/2026-09-25-documentation-cleanup.md)
  resolved all 48 diagnostics and verified a fresh strict HTML build with no
  warnings or errors. CI and Read the Docs now fail on documentation warnings.

## Decisions before merging or publishing

1. Review and accept the environment, removed-mode, and default changes in
   the draft PR. Check CI for its merge commit, including the retained
   main-side boundary aliases.
2. Select the release version and reconcile it with the documented stability
   policy. The metadata still says `0.4.1`; the environment/mode changes
   should not be described as a backwards-compatible patch. This review
   neither chooses a version nor changes the 0.x alias-removal policy.
3. Before a release tag, update package metadata, the README version, and the
   dated changelog section together; build and check the resulting artifacts.
   The existing release workflow builds on `v*` tags, but package publication
   requires its explicit workflow-dispatch input and protected environment.

No merge, tag, or package publication is part of this preparation task.

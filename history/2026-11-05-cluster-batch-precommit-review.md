# 2026-11-05 — Cluster batch review and commit

The additional review found no new numerical defect in the prepared batch or
automatic dependency scheduler, and adds mixed-gate regression coverage.

- Scope: user requested another review and a commit of the batch/auto work.
- Branch / baseline: `develop`, `859fda0`, including the accumulated batch and
  auto working-tree changes from the preceding sessions.
- Commit status: this note and the reviewed implementation, tests, API,
  changelog, module map, status ledger and earlier draft records are included
  in this commit. Earlier draft statements describe their original
  uncommitted state. Commit identity is available from Git. No push was requested.

## Review and fresh checks

- Rechecked both read/write dependency directions, skipped shape/one-site
  actions, nested pending groups, source boundaries and original commit order.
- Rechecked current-gauge merging, RMS/exponent preservation, shape/program
  compatibility, bounded cache eviction, per-member stopping and spectral
  fallback. The numerical implementation required no further changes.
- Added deterministic CPU/CUDA regressions with nonsymmetric complex
  nonunitary upper/lower gates, reversed site order, a nonzero exponent and
  contraction-cache capacity two. This tests physical output and local errors
  through cache eviction and lookahead beyond the original diagonal-ZZ cases.
- Eight additional randomized CPU/CUDA stream probes cover four seeds, each
  with 24 actions, mixed one-/two-site gates, reversals and cache evictions.
  Sampled physical relative differences from sequential execution stay below
  3.4e-14; normalized local error differences stay below 9.1e-17. Pair
  iterations and callback order match.
- Fresh existing batch/prepared/Torch-ALS selection: **110 passed in 30.89 s**.
  Expanded batch suite including the two new regressions: **37 passed in
  10.64 s**. These selections overlap; their counts are not additive.
- The earlier broader **555-test** selection remains valid for the unchanged
  numerical code; it was not rerun solely for additional tests/prose. The full
  repository suite and its separately recorded baseline failures were not
  reclassified as passing.
- `python -m ruff check src tests`, documentation targets and staged diff
  whitespace checks pass. No generated simulation data, local override or
  external runtime output is staged.

Evidence: `/tmp/pepsy-cluster-precommit-tests-20261105.log`,
`/tmp/pepsy-cluster-precommit-expanded-20261105.log` and
`/tmp/pepsy-cluster-precommit-random-20261105.json`.

The [auto scheduling record](../docs/development/notes/2026-11-05-auto-cluster-batches.md)
and [initial batch measurement](../docs/development/notes/2026-11-05-independent-cluster-batches.md)
retain performance methodology and limitations. AOT batching, downstream CLI
exposure and the three pre-existing optional-path defects remain separate.
No production process, checkpoint, frozen source snapshot or sibling repository
was changed by this review.

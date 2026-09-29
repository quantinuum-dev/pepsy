# 2026-09-28 — Publish integrated Pepsy cluster and PEPS work

- Scope: user requested committing, pushing, and pulling Pepsy.
- Branch / original baseline: `develop`, `4e398e4`.
- Publication: implementation commit `8673ef7` pushed to
  `origin/develop`; this handoff and the status-ledger addendum accompany
  a documentation-only follow-up commit. No release tag was created.

## Integration

- Reviewed and committed the earlier authorized cluster MPO/PEPO work,
  trace-only evaluation, PEPS optimizer/sampler fixes, tests, notes, and
  experiment launch handoffs. No device-local override, runtime outputs,
  environment files, or generated binaries were included.
- Pulled remote `develop` through `89e6d29` with rebase. The two conflicts
  were documentation-only: `CHANGELOG.md` and
  `docs/development/cluster_optimization_status.md`. Both sets of changes
  were retained. The numerical source merged automatically.
- Re-fetched before the push; remote had not advanced again.

## Validation on the integrated tree

- Affected cluster/PEPS/public-API/layout selection: **547 passed,
  2 skipped** (4 warnings).
- Full CPU suite with `CUDA_VISIBLE_DEVICES=''` and
  `JAX_PLATFORMS=cpu`: **5,313 passed, 105 skipped**, 684 warnings
  in 1701.07 seconds. No failures.
- `python -m ruff check src tests` and `git diff --check` passed.
- The full suite ran before the documentation-only publication addendum.
  No numerical source or test files changed afterward.

## Remaining limits

- The full run did not validate GPU or multi-rank MPI paths. Optional
  backend skips and warnings retain their normal interpretation.
- Trace-only results are complete at the selected cluster order, but may
  differ from the trace of a subsequently truncated MPO or PEPO.

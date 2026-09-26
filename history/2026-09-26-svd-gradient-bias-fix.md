# 2026-09-26 — Repair smooth Torch SVD derivative bias

- Scope: user's request to fix the two remaining Gaugy SVD-gradient failures.
- Branch / baseline: `develop` / `26a4141`.
- Commit status: included with the implementation commit; local only.

Both failures reproduced. The shared real/complex Torch SVD now uses compact
gap stabilization and a separate numerical-rank tolerance for inverse
singular values. No global constants were tuned, forward decomposition changed,
or Gaugy cost formula changed. Native Symmray uses the same registered driver.

See the [numerical policy, audit, and evidence](../docs/development/notes/2026-09-26-compact-svd-backward.md).
Focused backend validation: 112 passed, 1 skipped; Ruff and CI mypy pass.
Strict Sphinx build passes after retrying with network access for official
intersphinx inventories. Added both the new note and the previously omitted
native-scale note to the documentation toctree. Full Pepsy suite:
**4752 passed, 121 skipped**, 735 warnings in 336.53 seconds. Diff checks pass.
The paired Gaugy full suite passes: 414 passed, no xfails. Its change promotes
the strict xfails into ordinary regressions and expands full-loss coverage.

Inside stabilization regions the derivative remains a finite surrogate;
rank crossings are not promised to be differentiable. CUDA was not tested.
No push: the branches include earlier unpublished work outside this fix.

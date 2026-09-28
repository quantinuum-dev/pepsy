# 2026-09-28 — Explicit cluster MPO bond compression check

- Scope: make numerical post-compression of a fixed-cluster MPO measurable,
  including bond size, time and error at several chi values.
- Branch / baseline: `develop`, `4e398e4`.
- Status: uncommitted working-tree edits only; no staging or publication.
  Earlier cluster and unrelated optimizer/sampler changes are preserved.

## Implementation and evidence

The existing `compress_numerical` path was used explicitly after fixed MPO
construction. Its opt-in dense Frobenius error estimate now QR-canonicalizes
the difference MPO before computing the norm; this fixes strong cancellation
observed for small errors. Added a runnable 2x3 p=2 chi sweep and NumPy/Torch
regressions against dense reference errors. The [dated numerical record](../docs/development/notes/2026-09-28-cluster-bond-compression.md)
contains policy, measurements and limits. The owning
[MPO API guide](../docs/api/operators/mpo_cluster.md#explicit-bond-compression-after-fixed-construction)
shows the call sequence.

## Validation

- New error regression plus MPO suite: 138 passed (26.69 s) before the final
  Torch case was added.
- Affected MPO, fixed-cluster, public API and layout gate: **274 passed**,
  two existing deprecation warnings (60.40 s). It preceded a final narrow
  change limiting QR error stabilization to dense MPO data.
- Compression and native-sector regression gate after that guard:
  **22 passed, 117 deselected** (11.42 s).
- Runnable benchmark: completed on CPU with three repeats; measured timings
  and errors are in the dated record.
- Full Ruff, whitespace and 21 affected relative documentation links passed.

No new full-suite or GPU result is claimed. The earlier 5,241-pass full CPU
suite predates this change. Peak assembly memory, 5x6 p=4 numerical
construction, native-sector error stabilization and gradient-through-SVD
compression remain unverified.

# 2026-11-05 — Automatic independent-cluster opportunities

`batch_size="auto"` now selects compatible ready clusters from the stream,
lattice footprints and current workspace rather than requiring a fixed width.

- Scope: user requested automatic discovery and sizing of parallel cluster
  updates based on the gate stream and lattice.
- Branch / baseline: `develop`, `859fda0`, with the preceding
  [batch prototype](2026-11-05-independent-cluster-batches.md) already in the
  working tree. Existing prototype and review records were preserved.
- Commit status: all prototype/auto implementation, tests and documentation
  remain uncommitted. No sibling implementation, production job, checkpoint or
  frozen source snapshot changed.

## Implementation

Automatic readiness checks both write-after-read and read-after-write
dependencies against every preceding skipped/pending action, using complete
patch read sets and active-site write sets. Selected patches are disjoint and
share current program layouts/shapes. Fits may run ahead of independent
actions, while all mutations, exponent changes and callbacks retain original
order and every apply boundary remains a barrier. New ready groups can run
while an earlier lookahead result is waiting to commit.

Width has a lattice/stream ceiling, capped at 16, then shrinks for current
shapes, estimated memory and explicit contraction limits. The estimate
includes full-network gauge-inserted preparation snapshots and extra gradient
storage. Available host/device memory is refreshed with pending groups. The
policy falls back to serial when no batch fits; geometry-only serial windows
are rejected before making extra numerical pair copies. Integer caps retain
consecutive grouping, and the default remains one. Eager is required.

Diagnostics expose the configured policy, geometric ceiling, current memory
allowance/member estimate, actual widths/histogram and looked-ahead updates.
The owning [API guide](../docs/api/bp.md#prepared-ordered-pepo-cluster-replay),
changelog and status ledger are updated. Detailed method and evidence are in
the [auto scheduling note](../docs/development/notes/2026-11-05-auto-cluster-batches.md).

## Validation

- Batch/auto suite: **35 passed**, including CPU/CUDA, both complex
  precisions, Hermitian/PSD environments, physical reconstruction, rank growth,
  independent stopping, source/one-site barriers, transitive and shape-skip
  dependencies, nested pending batches and lattice-sized groups of six.
  Workspace/FLOP limits and larger full-network snapshots correctly shrink
  the width. Gate/gauge derivatives match sequential replay and finite
  differences with native/stabilized linalg.
- Final affected selection: **555 passed, 46 warnings in 83.36 s**. This
  covers the earlier BP/API/layout/PEPS/PEPO/paired-FU selection and the native
  graded reduced-compression regression, with the expanded auto suite.
- Full Ruff, documentation link targets and changed-file whitespace checks
  pass. The complete repository suite was not rerun; previously recorded
  unrelated native fermionic readout failures retain their separate scope.

## Measurements

- Geometry-only full Eli depth20 scans project batching 1,360/5,040 pair
  actions on 5x5 and 6,200/24,640 on 10x10, versus 940 and 4,720 with
  consecutive cap4. Actual shapes/ranks/memory can change these groups; they
  are not throughput or convergence claims.
- Six synthetic disjoint patches interleaved with independent one-site gates
  select width six. CUDA D4/D6/D8 ratios are **1.407x / 1.235x / 1.050x**;
  CPU D2 is **1.424x**, with matching pair diagnostics and physical matrix
  element relative errors below 1e-14. The linked note records controls.
- A bounded 96-action saved-state real chi6 replay finds no compatible
  independent groups and correctly stays serial. Pair diagnostics and the
  separate MPS-cap18 diagnostic agree; measured auto overhead is about 0.9%.
  This is additional replay on a saved final operator, not a new T=6 result.

Classification remains **prototype**. Installed dependencies and the upstream
audit from the active batch task are unchanged. No dependency, installed
source or linalg registration changed. The example runner does not yet expose
the batch-size flag. Existing optional-path defects and AOT batch support
remain outside this work.

Evidence is under `/tmp/pepsy-cluster-auto-*20261105*`, especially the final
synthetic/real-stream JSON files, schedule scan and
`/tmp/pepsy-cluster-auto-final-tests.log`. The note links their reproduction
scripts and distinguishes geometry scans, synthetic timing and saved-state
diagnostics.

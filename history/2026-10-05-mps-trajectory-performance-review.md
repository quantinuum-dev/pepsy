# 2026-10-05 — MPS trajectory performance and API review

- Scope: review performance opportunities and API improvements; no new
  implementation authorization was inferred.
- Branch / baseline: `develop` / `bec773a`, including prior uncommitted fixes.
- Commit status: two new documentation files only for this review; existing
  working-tree changes preserved. Nothing staged, committed or published.

Measured small NumPy trajectory workloads, a nonadjacent four-outcome Kraus
kernel, shot-setup profiling, and retain=none allocation growth. A temporary
shared-amplitude-block prototype suggests a roughly fourfold probability-kernel
opportunity; this is not an end-to-end production speedup. Existing coalescing
substantially helps the tested small-branch stream, while automatic threading
slows this small workload. Proposed bounded local reduction and clearer
trajectory/result APIs alongside the kernel and setup optimizations.

See the [review and measurements](../docs/development/notes/2026-10-05-mps-trajectory-performance-review.md)
for conditions, source ownership, priorities and validation. Worker determinism,
dense-reference probability and prototype state-preservation checks passed.
Relative documentation links and diff whitespace checks passed. No new test-suite
run or backend-wide performance claim; prior limitations remain recorded in the
[fix handoff](2026-10-05-mps-trajectory-fixes.md).

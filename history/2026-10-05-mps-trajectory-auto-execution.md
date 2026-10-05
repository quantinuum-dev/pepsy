# 2026-10-05 — Automatic trajectory scheduling

- Scope: implement metadata-based MPS trajectory choices without user trial
  runs, preserve explicit overrides, and validate selected paths.
- Branch / baseline: `develop` / `bec773a`, with prior working-tree changes.
- Commit status: uncommitted; nothing staged, committed or published.

Added an MPS-owned worker policy, rare-fixed-mixture coalescing selection,
importance-aware structural planning, branch-cap diagnostics and API guidance.
Fixed the conditional reset/measure-reset tuple parser discovered by new tests.
Earlier changes in the same files and unrelated working-tree changes remain.
See the [implementation and limits](../docs/development/notes/2026-10-05-mps-trajectory-auto-execution.md).

Validation: combined trajectory/Kraus/dense-reference/dynamic-control/fermion,
MPI and public API/layout selection: **563 passed, 1 skipped, 30 deselected**.
The skip requires a second JAX device; deselections are slow tests. Additional
conditional-control/layout/STN selection: **11 passed, 166 deselected**.
Final automatic-execution tests, including real CUDA/CuPy and a bond-64 CPU
parallel replay: **43 passed**. This rerun overlaps the combined selection and
includes one additional CPU test. Ruff, relative-link checks and
`git diff --check` passed. No full suite or multi-rank GPU run.

The rules are deterministic scheduling heuristics, not universally optimal
performance claims. They preserve the selected precision/truncation settings;
automatic branch overflow replays independently without pruning. Prior GPU
ledger tolerance failures and unrelated full-suite failures remain unresolved
as documented in earlier handoffs. No additional numerical tolerances changed.

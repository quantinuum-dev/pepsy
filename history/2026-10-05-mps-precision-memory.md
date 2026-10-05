# 2026-10-05 — MPS precision and memory budgeting

- Scope: user requested items 1 and 2: resolve the three complex64 ledger
  comparisons and implement automatic GPU retained-state memory budgeting.
- Branch / baseline: `develop` / `2039fbc`.
- Status: working-tree implementation and documentation; nothing committed or
  pushed in this follow-up. Unrelated edits preserved.

Implemented bounded double-precision small-operator factorization with factors
cast back before MPS compression, preserving caller SVD policy and state dtype.
The original ten ledger checks pass. Added allocator-aware local-shot memory
planning, explicit byte budgets, branch/worker limits, retention preflight,
and guarded automatic fallback that releases the failed frontier first.

See the [audit, numerical findings and limitations](../docs/development/notes/2026-10-05-mps-precision-memory.md)
and the [public API](../docs/api/optimizers/mps.md). Memory estimates are not
allocator reservations or universal OOM guarantees. MPI budgeting and
continuation from a capped frontier remain deferred. Traced JAX operators with
x64 disabled retain their previous factorization precision.

Validation completed:

- Original ledger-only selection: **10 passed**, with unchanged 3e-6 bounds.
- Final combined precision/memory, MPS compression/normalization/backend,
  Kraus/trajectory/importance/dense-reference/dynamic-control/fermion, MPI
  unit and public API/layout selection: **838 passed, 2 skipped, 30 deselected**.
  The skips are unavailable Metal and a second JAX device; slow cases were
  excluded. Includes actual Torch CUDA, CuPy and JAX GPU execution.
- Earlier combined run: 831 passed and one newly added JAX gradient test
  failed because it omitted the optimizer's highest-matmul accumulation scope.
  The final test includes that existing contract and passes; no tolerance
  was weakened. Torch CPU/CUDA and JAX with x64 enabled/disabled gradients
  match dense references.
- Full-suite attempt, `--maxfail=3`: **83 passed, 1 skipped, 3 failed**, stopped
  on BP convergence failures. All three reproduce on an exported clean
  `2039fbc` baseline: `test_sequential_loop_series_compression_reuses_projected_messages`,
  `test_sequential_loop_series_refreshes_topology_cache_after_reduction`, and
  `test_simultaneous_loop_series_compression_uses_one_boundary_snapshot` in
  `tests/test_bp_compression.py`. This is not a clean full-suite result.
- All-source/test Ruff, changed-document local links and `git diff --check`
  passed. No multi-rank GPU memory test.

A synchronized alternating timing check shows a precision cost on a small
complex64 workload: Torch CUDA 49.83 to 64.39 ms (~29% slower), CuPy 62.15 to
63.46 ms (~2% slower). See the audit for workload and limits. Correctness was
prioritized; no overall speedup is claimed for the precision correction.

# 2026-09-28 — Compiled MPO/PEPO cluster API review

User requested a further API review and focused improvements. Branch `develop`
at baseline `4e398e4`, with the earlier uncommitted geometry and compile-review
work preserved. This review is also uncommitted and unpublished.

## Additions

- `MPOClusterProductExpansion.compile_exp()` reuses its callable, including
  through the existing cached `MPOBasis` compile helpers. Numerical results
  remain fresh per evaluation.
- `CompiledMPOClusterProduct.last_report` exposes the latest construction
  diagnostics (None before evaluation). `exp`, `evaluate` and `__call__` accept
  `return_report=True`, returning `(semantic_mpo, report)`. The default result
  remains `FirstDegreeMPO`.
- `CompiledPEPOExp` exposes `cache_info` and `cluster_inventory` directly.
  Ordered PEPO products and their compiled wrappers expose the shared spatial
  inventory; product cache diagnostics include independent per-factor snapshots.
- The API guides distinguish requested assembly settings from the resolved
  report, square-lattice connectivity from snake ordering, history order from
  spatial cluster size, and joint residual PEPOs from composed layer PEPOs.
  A complete reusable 2D MPO compile example is included and executed.

This review does not change numerical defaults, graph assembly fallback,
rank/truncation policy, boundary handling or the local residual algorithm.
Shape inspection is distinct from a numerical residual or error report.

## New validation

- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -o addopts='' tests/test_mpo_cluster.py tests/test_cluster_expansion.py tests/test_public_api.py tests/test_package_layout.py`:
  **160 passed**, two existing compatibility deprecation warnings, 54.67 s.
- Full Ruff, whitespace checks, and relevant local documentation links passed.
- Executed the new square-graph/OneDMap compiled example successfully.
- New/extended checks cover cached callable identity with fresh results,
  last-report updates and optional paired returns, diagnostic snapshot isolation,
  and a square bond spanning separated snake sites against a dense exponential.
  The latter also checks that requested `auto` remains visible while the report
  records the bounded collection fallback.
- Full repository suite not run. No new performance claim from these API edits.

All pre-existing unrelated PEPS/sampler edits and running jobs were preserved.
Gaugy was not edited. Tracked changes used unified patches through `git apply`
because apply_patch's sandbox failure was already diagnosed in this session.

# 2026-09-26 — Native Symmray SU gauge renormalization

- Scope: shared Pepsy fix required for the user's native spin-Z2 Gaugy workflow.
- Branch / baseline: `develop`, `0a43488`.
- Commit status: included with this entry in a local commit; not published.

`renorm_gauge` now reduces backend blocks without densifying Symmray vectors
or calling its unsupported mean. It preserves the same detached scale for
division and exponent bookkeeping, including zeros and extreme weights.

Validation: 165 passed, 1 skipped in gate/scale suites; Ruff and diff checks
passed. See the [dated audit](../docs/development/notes/2026-09-26-native-su-gauge-scales.md).
No installed libraries changed. The full Pepsy suite was not rerun for this
single-helper change. Device-local policy and unrelated files are excluded.

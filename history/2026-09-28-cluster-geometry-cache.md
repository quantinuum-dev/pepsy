# 2026-09-28 — Pepsy cluster inventory and geometry reuse

- Scope: user requested tree/loop separation and improved caching, Pepsy first.
- Branch / baseline: `develop` at `4e398e4`.
- Commit status: working-tree edits only; nothing staged, committed or published.
- Added per-size oriented/C4 tree/loop inventories to dense plans and Pauli
  bases. Cached immutable shape levels across cutoffs and finite embeddings
  across builds; numerical residuals and parameter graphs stay dynamic.
- New validation: 120 cluster/API/layout tests passed; full Ruff and whitespace
  checks passed. Independent 5x6 OBC enumeration confirms 295 four-site clusters,
  including 20 plaquettes. Full repository suite not run.
- Geometry-only measurements: sequential cutoffs 2–7 approximately 1.12x faster;
  30 repeated four-site placement passes approximately 23.45x faster. End-to-end
  acceleration is unmeasured. See the
  [evidence and limits](../docs/development/notes/2026-09-28-cluster-geometry-cache.md).
- Preserved unrelated PEPS/sampler working-tree changes and existing run
  handoffs. Gaugy was not edited and existing simulations were not restarted.

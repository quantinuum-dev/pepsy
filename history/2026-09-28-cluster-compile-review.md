# 2026-09-28 — Complete static preparation for compiled cluster PEPOs

- Scope: user requested another review of fixed-Hamiltonian-structure,
  fixed-geometry compile/cache reuse for PEPO construction; Pepsy first.
- Branch / baseline: `develop`, `4e398e4`, plus prior working-tree cluster edits.
- Commit status: uncommitted; nothing staged or published. Preserved unrelated
  PEPS/sampler edits and existing jobs; no Gaugy changes.
- Compile located maps and tree metadata eagerly, reuse maps by local graph,
  prepare higher-order uniform source maps, and select the located route for
  all factors of mixed uniform/located products. Static-only caches remain
  separate from fresh parameter-dependent numerical work.
- New checks: 122 cluster/API/layout tests passed; full Ruff and whitespace
  checks passed. Includes repeated coefficient/time values and Torch gradients
  against dense references. Full repository suite not run.
- For 5x6 OBC order 4, static-map entries 492 -> 15, bytes 84,268,544 ->
  2,828,544; median static preparation 1.234 s -> 0.042 s. End-to-end PEPO
  acceleration unmeasured. See the
  [review evidence](../docs/development/notes/2026-09-28-cluster-compile-review.md).

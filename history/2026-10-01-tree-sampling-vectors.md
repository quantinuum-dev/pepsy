# 2026-10-01 — Improve exact TreeSampler efficiency

- Scope: user requested `left_inds`/canonicalization inspection and additional
  TreeSampler performance improvements.
- Branch / baseline: `develop` / `5eadd9f`.
- Commit status: working-tree edits only; nothing staged, committed or published.
  Earlier solver/notebook-review edits and their handoff were preserved.

## Implemented and confirmed

- [TreeSampler](../src/pepsy/sampling/tree.py) now carries structurally pure
  conditional environments as vectors, retaining densities for mixed sibling
  environments. Exact Born weights, draw order, precision and chi are preserved.
- Cleared dense scoring's recursive closure on success/failure, releasing
  discarded snapshots without cyclic GC.
- Confirmed canonical preparation already reuses the live canonical region
  and `left_inds`, with only the center-to-root path needed. Sampling itself
  does not move a center or mutate its source.
- Updated [sampling API](../docs/api/sampling/tree.md), changelog and
  [sampler regressions](../tests/test_tree_sampler.py).

## Fresh validation

- Sampler, canonical-region, public-API and package-layout selection:
  **186 passed**, one two-GPU skip, two compatibility deprecation warnings.
- Covered NumPy, Torch CPU/CUDA, CuPy, native Symmray, independent dense Born
  references, normalized Torch gradients, source ownership, canonical-path
  reuse and GC-disabled snapshot release on success/failure.
- The vector-route and snapshot-lifetime regressions fail on the frozen
  before-change implementation. Ruff and whitespace checks passed.
- Repeated synthetic 30-site complex128 benchmarks matched all configurations
  exactly and passed independent scoring. CuPy chi=256, 8,192 shots/chunk 2,048:
  median **28.93 s → 17.73 s**, **1.63× faster** over two comparisons. NumPy
  chi=32, 2,048 shots/chunk 256: median **15.68 s → 8.68 s**, **1.81× faster**
  over three comparisons. These use explicit `threads=1`; defaults remain
  unchanged. CPU baseline variability is recorded in the detailed note.
- Full package tests and production-state benchmarks were not run.

See the [detailed implementation, upstream audit, numerical evidence and
measurement setup](../docs/development/notes/2026-10-01-tree-sampling-vectors.md).

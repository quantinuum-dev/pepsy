# 2026-09-30 — Shared-density TreeSampler optimization

The user authorized integrating the exact shared-density optimization from
the [performance audit](2026-09-30-tree-sampling-performance-audit.md).
Baseline: `develop` at `414798e`. These are working-tree changes; no commit
or publication was made. Unrelated PEPS edits were preserved.

## Implemented

[TreeSampler](../../../src/pepsy/sampling/tree.py) starts each dense batch
with one root density. First-child traversal retains that singleton until
physical conditioning makes the remaining sibling tensors depend on each
shot. Physical probabilities are normalized before broadcasting to all
shots, preserving the original uniform-draw ordering.

For a singleton density, the first-child transfer contracts the incoming
density with the node tensor first, then its conjugate. The intermediate has
the node's parent/child/sibling dimensions, rather than the quartic
parent-density-to-child-density map. This path is used for both chunked and
unchunked sampling, including one-shot requests. Shot-dependent paths retain
the previous contractions and optional environment tiling.

No truncation or persistent numerical cache was introduced. Shared densities
are local to a batch, so refresh invalidation and numerical lifetimes are
unchanged. Canonicalization, canonical-region ownership, native Symmray
sampling, source ownership, backend/device selection, and public signatures
are unchanged. Tree densities may remain mixed due to unmeasured sibling
branches; the optimization does not substitute MPS prefix vectors for them.

The [API guide](../../api/sampling/tree.md) explains sharing and existing
canonical reuse. The unreleased changelog records the optimization.

## Validation

Activated the existing py312 environment with local source imports and one
BLAS/OpenMP thread. Reused the unchanged environment/dependency audit linked
above: **adopt** existing native einsum/broadcast operations; no upstream
shim or dependency changes.

- Sampler, tree entropy, and tree unitary-stability selection:
  **151 passed, 1 skipped**, with two expected legacy `dmrg1` warnings.
- After adding the work-sharing assertions and complex-gradient regression,
  the updated/new regression selection passed **19 tests**. Eighteen of these
  overlap the preceding selection; there are **152 unique passing tests**
  across the two runs, not 170. No implementation changed between runs.
- The skipped case requires two GPUs; real Torch CUDA and CuPy paths passed
  on the available RTX A5000. Full-package tests were not run.
- Added independent dense conditional sampling references for NumPy, Torch
  CPU/CUDA, and CuPy, including chunked/unchunked sampling, ternary branches,
  physical-root conditioning, source preservation, and device placement.
- Singleton-transfer checks compare independent dense contractions and
  verify no quartic environment is constructed. Torch complex128 gradients
  match the direct three-operand reference.
- Both the singleton-transfer and first-conditional work-sharing regressions
  fail against the isolated baseline: the old helper still builds tiles and
  the old first conditional carries 31 density copies instead of one.
- `python -m ruff check src tests`, local documentation links, and
  `git diff --check` passed.

## Measured comparison

One warmed measurement per implementation on the same synthetic 30-site
balanced tree, with three root children, actual chi=256, CuPy complex128,
8,192 samples, chunk_size=2048, sample seed 2, and state seed 19. This reuses
the audit's construction with edge dimensions capped by subtree Hilbert
space. GPU synchronization surrounds each complete call; this comparison
does not add per-contraction instrumentation. State construction and sampler
capture are outside the timed region. No competing GPU benchmark ran.

| Implementation | Sampling time |
| --- | ---: |
| Baseline `414798e` | 39.87 s |
| Integrated shared densities | 28.90 s |

This is approximately **27.5% less elapsed time** in this single comparison,
not a production throughput guarantee. All **8,192 configurations matched
exactly**. Maximum relative probability difference was 4.62e-14; separate
bottom-up scoring of 16 outputs agreed to 4.30e-14 relative error.

Temporary reproducer/results: `/tmp/pepsy_shared_density_benchmark.py` and
`/tmp/pepsy_shared_density_benchmark.json`; baseline source snapshot:
`/tmp/pepsy_tree_sampler_before_sharing.py`. Earlier prototype measurements
remain historical evidence and are not substituted for this integrated run.

Remaining limits: the user's actual evolved 5×6 tree was not supplied;
geometry, dtype, and hardware affect throughput. General shot-dependent
density transfers remain expensive. Shared-prefix densities are recomputed
once per chunk, not retained across calls, and arbitrary equal measurement
histories are not grouped. Full multi-GPU and full-package validation remain
unperformed.

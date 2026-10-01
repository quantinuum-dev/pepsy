# 2026-10-01 — Native factor sampling and dense cache improvements

- Scope: the user approved all three proposed TreeSampler improvements,
  continuing the existing commit/push authorization.
- Branch / baseline: `develop` at published `ab9a445`.
- Commit status: included in the scoped commit titled `Add native Symmray
  tree factors and reduce factor cache overhead`. Publication targets
  `origin/develop` and is checked against the remote ref at final handoff.
  Unrelated solver, MPI, PEPS and trajectory changes are excluded, including
  the separate JAX changelog entries.

## Changes and decisions

Implemented native Symmray canonical-factor sampling with shared measured
prefixes, graded centre norms, source-sector code remapping and normalized
prefix states. Sample probabilities accumulate in backend float64; fixed
configuration Torch gradients survive capture, sampling and scoring. Native
scalar unwrapping now preserves Torch graphs and CuPy scalars. Captured native
normalization uses its canonical graded root, clearing only the private
global exponent. Dense factor sampling reuses the unconditioned virtual-root
density and merges density-cache entries by direct scatter.

Native factor retains cross-chunk payloads only when a subsequent chunk can
reuse them. The memory comparison exposed unnecessary one-chunk retention,
which is removed. The source remains unchanged; exact standard is available
explicitly. `backend="auto"` still densifies Symmray to NumPy compatibility;
use `backend="native"` or `"symmray"` for the new native factors. The dense
workspace target does not tile native QR or bound total memory.

The [API guide](../docs/api/sampling/tree.md) describes public behavior; the
[audit and measured evidence](../docs/development/notes/2026-10-01-tree-native-factor-improvements.md)
records installed versions, dispatch checks, allocation scope and raw timing
tables. Earlier factor-default handoffs accurately describe the native
algorithm before this change.

## Validation and limits

Final focused selection after the cache admission refinement: **585 passed,
two skipped**, four existing compatibility warnings in 184.55 seconds.
Earlier current-task validation passed 582 tests before that refinement;
the linked evidence distinguishes both results. The final command covers
validity, both dense strategies, native factors,
sampler integration, canonical regions, entropy, unitary stability, public
API and package layout. Torch CPU/CUDA, CuPy and native Symmray ran; skips are
the known CuPy subnormal-source limitation and a check requiring two GPUs.
Ruff, relative documentation links and staged/working whitespace checks pass.
The linked note records the exact command, environment and result scope.

Dense bond-256 CuPy before/after measurements improved small-chunk timings,
with little overall peak-memory change. The small native NumPy comparison
produced identical samples to standard, with much faster sampling and a lower
tracked peak after removing unused cache retention. Large-bond native,
production-checkpoint and Torch/autograd peak memory remain unverified;
full-package tests were not run. There is no universal memory/speed guarantee.

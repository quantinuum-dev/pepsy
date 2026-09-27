# 2026-09-26 — PEPS sampler performance implementation

- Scope: user approved implementation of the assessed immediate sampling
  improvements after inferred/explicit backend support.
- Branch / baseline commit: `develop` / `80f451a` (ahead 8 at start).
- Commit status: uncommitted working-tree edits; nothing staged or published.
  Preserved prior backend work and unrelated qMERA/tree changes.

## What changed

- Added a 64 MiB configurable dense-row cache estimate/budget and fallback
  before allocation, covering dtype, live groups, future/conditioned bonds,
  and uncompressed represented-bond growth.
- Grouped native validation/RNG/log updates by site. Reduced local-rho network
  copies, reused row metadata/identity caps/column traces, and preserved network
  exponents in direct tensor contractions.
- Made complex64 Hermiticity diagnostics stable under very large/small scales.
- Extended API docs, changelog, regression tests, and
  [detailed evidence](../docs/development/notes/peps_sampler_performance_implementation.md).

## Measured findings

- 4x4 D4, 32 shots: NumPy median 0.761→0.610 s; shared CUDA complex128
  7.286→5.810 s; shared CUDA complex64 1.532→1.176 s. Limited repetitions and
  ongoing production load constrain these results.
- CUDA complex128 sampler validation/choice reads each fell 349→16 per batch.
- 3x3 D4 now automatically avoids the prior 1 GiB column cache; new batch
  median 0.0629 s versus the earlier recorded 15.934 s dense-cache measurement.
- JAX correctness/placement passed, but first-call cost increased 10.28→17.64 s
  in the small probe. No general JAX throughput improvement is claimed.

## Validation

- Focused suite 50 passed, followed by 2 new represented-bond checks passed.
- Additional sampler/public API/layout: 161 passed, 1 known failure.
- Smoke: 152 passed, 1 known failure.
- Both failures are installed 0.4.0 vs pyproject 0.5.0 version metadata;
  shared environment and package metadata were not changed.
- Torch CPU/CUDA four-path matrix: 12/12 passed, source arrays preserved.
- Ruff over src/tests and diff whitespace checks passed.

## Limits / handoff

JAX shape-compilation startup, eager per-prefix contractions/compression,
Symmray, JAX GPU/sharding, and production high-D throughput remain unverified
or deferred as detailed in the note. No production jobs were altered. The
sandbox helper failed before file access, including apply_patch; authorized
shell actions used automatic escalation and checked file replacements instead.

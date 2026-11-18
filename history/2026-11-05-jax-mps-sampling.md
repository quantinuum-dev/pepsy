# 2026-11-05 — Native JAX MPS sampling

- Scope: implement and test native JAX `MpsSampler` sampling.
- Branch / baseline: `develop`, `ad523b6`.
- Commit status: local working-tree changes; no commit or publication.

## What changed

- Added lazy JAX preparation, device-native draws, normalized probabilities
  and complex amplitudes, PRNG keys, inference/gradient controls, and refresh
  support. Advertise dense capabilities through `MpsSampler.native_backends`.
- Added exact/empirical, dtype/device, JIT/autodiff, scale and cache regressions;
  classified the new module as core/optional/sampling, outside default smoke.
- Updated API and implementation documentation and changelog.

## Validation and limits

The [dated evidence](../docs/development/notes/2026-11-05-jax-mps-sampling.md)
records the upstream audit, CPU timings, 26 passing new tests, 184 passing
existing tests, ten skips and the environment's sole package-version failure.
Full source/test Ruff passed. Full numerical-suite and GPU execution were not
tested. No shared packages or device-local policy files were changed.

Gaugy's word policy was separately adapted to prefer native JAX only when
Pepsy advertises support, retaining its older-version host fallback. That
sibling change has its own validation and publication record.

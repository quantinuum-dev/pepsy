# 2026-11-04 — Reduced full-update factor reuse

- Scope: user-approved FU acceleration and CPU Torch16 χ6 bottleneck study.
- Branch / baseline: `develop` / `ee22198`.
- Commit status: working-tree changes; no commit or push requested this turn.

## Implementation

Cache the rank-truncated initial SVD factors and use QR/LQ plus a χ×χ SVD
to balance ungauged reduced solver results. Keep the explicit environment-gauge
final split. Preserve source amplitude, target normalization phase, SU starts,
both PEPO legs and paired updates. Native reduced Quimb ALS uses dense local
solves, promoting the downstream Torch/CuPy dispatch fix.

The sibling Pepsy benchmark retains one optimizer across separate source-chunk
`run()` calls and throttles progress-file writes while retaining pair records.
Numerical kernels stay in Pepsy. Historical frozen runs remain unchanged.
See the [implementation/audit note](../docs/development/notes/2026-11-04-full-update-factor-reuse.md)
and [public guide](../docs/api/optimizers/peps.md).

The user then requested boundary reuse and four DMRG iterations, plus an SRC
comparison. Native constructor/runtime and benchmark defaults now agree on
`reuse-direct`, four boundary iterations; explicit overrides retain precedence.
One-site FIT cannot grow an early small boundary. Non-differentiable dense
warm starts are expanded with public zero-padding/QR operations, preserving
their represented state. Differentiable/unsupported guesses and reduced caps
retain cold seeding. Native Symmray behavior is unchanged. This supersedes an
intermediate conservative cold-refresh guard retained in the benchmark archive.

## Validation

- Final boundary/FU/gauge/cache/scheduling/fermionic/optimizer selection:
  **848 passed**, including actual CPU/CUDA/CuPy checks. Earlier 436/437
  selections and intermediate 843-pass default-assertion recheck overlap it.
- New comparisons check both operator legs and paired action, complex64/128,
  complex scale and normalization phase, no repeated pair split, the symmetric
  singular-value gauge, and native dense χ6 solves.
- `ruff check src tests` and `git diff --check` pass.
- Default smoke: **112 passed**. Benchmark integration: **92 passed**.
- Full extended suite attempted with `--maxfail=1`: **212 passed, 2 skipped,
  1 failed** at `test_bp_symmray.py::test_native_reduced_loop_compression_uses_graded_svd_adapter`.
  The same test fails on frozen pre-change Pepsy with the same reduced shape
  mismatch `(1,1,2,2)` versus `(1,2,2,2)`. Do not claim full-suite success;
  this BP/Symmray failure is pre-existing and outside the requested FU work.
- Full T=6, 23-spin χ2/b12 exact-scalar controls: factor/cache changes agree
  with the frozen baseline within 8e-14; final warm expansion/four sweeps
  agrees with fresh direct/four sweeps within 1.3e-9. This does not establish
  χ6 or norm-boundary convergence. CPU16 timing is a bounded prefix study,
  not a full χ6 evolution or GPU speedup measurement.

## Measured scope

The benchmark's matched 23-spin inhomogeneous D20/χ6/b36 two-source-slice
CPU16 study took 657.15 s with the frozen two-boundary-iteration baseline,
604.93 s after factor/PSD/orchestration changes with the same controls,
513.51 s with conservative guarded reuse/four iterations, and **288.96 s**
with final warm expansion/four iterations. Final fresh direct seed calls
fell from 245 to 51. This is a bounded pipeline measurement, not a completed
full-T χ6 result or isolated equal-iteration comparison. Fresh SRC was slower
on the separate initial-slice control despite cheaper seed construction.
The example owns the detailed dated record, configs, profiles and source pins.

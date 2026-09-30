# 2026-09-26 — PEPS sampler efficiency, streaming, diagnostics, and benchmarks

- Scope: user authorized all five follow-ups from the sampler review:
  conditional reuse, bounded batches, amplitude evaluation, diagnostics,
  and CPU/GPU profiling. Runner integration was not requested yet.
- Branch / baseline: Pepsy `develop`, `a13031b`.
- Commit status: uncommitted working-tree changes; no staging or publication.
- Preserved earlier uncommitted sampler scaling/rho/cache work and unrelated
  files. The starting sampler copy is `/tmp/peps_sampler_before_efficiency.py`.

## Implemented

- Opt-in factored row suffix caching, with reference fallback and memory estimates.
- `iter_samples` and optional collected-batch chunking; fixed seed/chunk replay.
- Reusable exact amplitude plan and scaled physical-slice cache with exponent
  stripping, phase preservation, zero handling, and refresh invalidation.
- Stable result weights/ESS and compact sampler diagnostics. Aggregated Torch
  diagnostic scalars do not retain prior chunks' autograd graphs.
- Evolved-state benchmark, focused regressions, API docs, and changelog.
- No changes to cutoffs, proposal repair defaults, evolution code, installed
  libraries, or the running production jobs. Small time-limited GPU probes
  shared the occupied GPU; isolated production throughput remains unverified.

## Evidence

See the [efficiency study](../docs/development/notes/peps_sampler_efficiency.md)
for parameters, full measurements, upstream audit, and limitations. Wider 8×4
D=2 CPU sampling improved about 1.52×; tested 4×4 cases had extra cache overhead,
so defaults remain unchanged. CUDA amplitudes agreed with a dense oracle to
1.92e-8; GPU timing was variable under contention and establishes no speedup.

## Validation

- Initial focused implementation checks: 8 passed.
- Initial new tests: 22 passed, one test used an unrepresentable complex64
  single-site probability. Corrected the fixture to test representable local
  probabilities with an underflowing full-prefix product; that check passed.
- Initial existing cache failures were strict diagnostic-schema expectations
  missing the added `cache_representation` key. Updated the explicit expected
  schema; numerical comparisons already passed.
- After physical-slice caching, amplitude/cache/weight/memory selection: 33 passed.
- After the diagnostic graph-retention guard: chunk/graph/truncation selection, 9 passed.
- New-feature suite after that guard: 28 passed (NumPy/Torch/JAX CPU).
- Additional host-exponent precision guard: all 10 amplitude checks passed.
- Complete sampling-domain validation (started before the final diagnostic
  graph and host-exponent guards; covered separately above): **374 passed,
  6 skipped**, 50 existing Quimb deprecation warnings, 900.02 seconds.
  NumPy/Torch/JAX CPU included two JAX CPU devices. Five CuPy checks and one
  MPS CUDA check skipped because CUDA was intentionally hidden in this run.
  Separate bounded Torch CUDA PEPS benchmarks passed their
  independent amplitude checks.
- Public API/layout: 58 passed, one existing installed-version metadata failure.
- Smoke: 162 passed, one skipped, same existing metadata failure. Installed
  distribution/runtime is 0.4.0 while checkout metadata declares 0.5.0.
- Ruff (`src tests` plus benchmark) and `git diff --check`: passed.
- No full repository numerical suite was run.

`apply_patch` was attempted but failed at sandbox startup with the existing
bubblewrap mountinfo error. Exact-match checked replacements were used for
tracked edits. No approval rejection or environment modification occurred.

# 2026-10-01 — Improve and integrate the factor sampling prototype

- Scope: user requested further work on factor, grouping and caching.
- Branch / baseline commit: `develop` / `5eadd9f`.
- Commit status: working-tree edits only; no staging, commit or publication.
  Earlier solver and pure-vector sampler work was preserved.

## Implemented

Added experimental exact `TreeSampler(strategy="factor")`, with call-local
bounded density caches, compact factors/densities/remainders, tiled gathers
and projections, and same-draw retries for oversized grouped intermediates.
Fixed duplicate cache-full contractions and numeric-key overflow/order
handling. The standard default and native Symmray algorithm remain in place.
No literal batched-center QR implementation was promoted.

Updated the sampling API, module map and changelog; added focused sampler
regressions. Details and numerical evidence are in the
[integration note](../docs/development/notes/2026-10-01-tree-factor-sampling-integration.md).

## New checks and limitations

- Focused sampling/canonical/public API/package layout selection: **323 passed,
  one skipped** (requires two GPUs), two compatibility warnings.
- Factor suite after explicit native output/device assertions: **134 passed**.
- Includes CPU/CUDA Torch gradients, exact dense references, overflow known
  Bell correlations, zero branches, source ownership, cache exhaustion,
  reentrant calls, refresh and GC-disabled success/failure cleanup.
- Ruff and whitespace checks passed. Full package tests and the user's
  production checkpoint were not run; no checkpoint was supplied.
- Cache/workspace controls are local payload/tiling policies, not a global
  memory or autograd-graph bound. Default promotion remains deferred.

## Final measured result

Serial synthetic CuPy 30-site chi=256 complex128 comparisons, 8,192 samples
in 2,048-shot chunks, preserve exact configurations/reference probabilities.
Three-root-child median: standard 17.768 s, old prototype 2.928 s, new factor
2.951 s (6.02x versus standard). Binary-root median: standard 14.155 s, old
prototype 3.556 s, new factor 3.551 s (3.99x versus standard).

The improvement over the prior prototype is reliability/backend coverage and
live allocation cost, rather than a further measured speedup. Additional live
CuPy-pool peaks fall from 6.494 to 2.150 GiB (three children) and 6.029 to
1.670 GiB (two children), reductions of 66.9% and 72.3%. These exclude source
and captured state payloads and unused reserved pool blocks. See the linked
note for raw CPU/GPU times, allocator method, cache sizes and reproduction.

## Resumed review — 2026-10-01

The user requested checking the unfinished TreeSampler work and resuming it.
Confirmed the implementation and earlier solver edits remain uncommitted on
`develop` at `5eadd9f`; no implementation edits were made during this review.
The existing Python 3.12 environment has the same dependency versions as the
earlier audit. Reviewed the shared traversal, factor transfers, grouping and
retry/cache lifetime against the API guide and focused regressions.

Fresh checks:

- `python -m pytest -q -ra -o addopts='' tests/test_tree_factor_sampler.py tests/test_tree_sampler.py tests/test_tree_canonical_regions.py tests/test_public_api.py tests/test_package_layout.py`:
  **323 passed, one skipped**, two compatibility deprecation warnings.
  The skip still requires two CUDA devices.
- Sixteen additional default-budget checks, without forcing the grouping
  thresholds: NumPy, Torch CPU/CUDA and CuPy, physical/virtual roots, chunks
  unset/seven, seven-site chi=5 complex128 states, 41 shots, state seed 19
  and sampling seed 17. Configurations matched the standard sampler exactly;
  probabilities matched independently normalized dense states at
  `rtol=2e-11`, `atol=1e-14`.
- Ruff (`src tests`) and `git diff --check` passed.

Logs: `/tmp/pepsy_tree_resume_checks.log` and
`/tmp/pepsy_tree_resume_default_checks.log`. No new performance measurements,
full-package run, production checkpoint validation, staging, commit or
publication. The experimental factor path remains opt-in; production-state
profiling and full-package validation remain outstanding before considering
default promotion. Further implementation direction was requested from the
user; these findings do not authorize default promotion.

## Finalization — 2026-10-01

The user subsequently requested finalizing the current implementation.
Kept `strategy="factor"` experimental and opt-in. Added 16 default-setting
regressions covering single-site leaf/root geometries and branching trees on
NumPy, Torch CPU/CUDA and CuPy. Each case compares explicit and persistent RNG
calls, with and without chunking, to standard configurations and independent
dense Born weights (96 factor calls, each paired with a standard call).
The factor suite is now marked
`tree`, `integration` and `optional`, matching its domain/backend test scope.
Clarified the public guide and constructor help's workspace-driven batching,
and the guide's within-call cache lifetime. No further production algorithm
change was needed by this review.

- Focused factor/sampler/canonical/entropy/unitary/public API/package selection:
  **408 passed, one skipped**, four compatibility warnings. The skip needs
  two CUDA devices; the available single-GPU paths were exercised.
- Domain/profile collection selects all **150** factor cases.
- Ruff and whitespace checks pass; relevant local documentation links resolve.
- Full CPU suite, with CUDA hidden, JAX on CPU and one numerical thread:
  **6,493 passed, 287 skipped, seven failed**, 716 warnings, in 34:01.
  All seven failures reproduce separately: three MPS tests request the
  removed `dmrg1` mode, and four PEPS 4x4 checks disagree with amplitude or
  backend-parity references. Their implementation/test files are unchanged by
  this task. See the detailed
  [finalization record](../docs/development/notes/2026-10-01-tree-factor-sampling-integration.md#finalization-validation--2026-10-01)
  for names and limitations. This is not a full-suite pass.

The implementation review, focused validation and documentation/profile
integration are complete in the working tree; default promotion remains
deferred. No test runners remain active. Logs:
`/tmp/pepsy_tree_finalize_focused_final.log`,
`/tmp/pepsy_tree_finalize_domain_collection.log`,
`/tmp/pepsy_tree_finalize_full.log`, and
`/tmp/pepsy_tree_finalize_unrelated_mpi.log`.
Additional isolated failures are recorded in
`/tmp/pepsy_tree_finalize_unrelated_peps.log` and
`/tmp/pepsy_tree_finalize_unrelated_remaining.log`.
No new performance measurement or production-checkpoint validation; existing
timings remain earlier synthetic evidence. No staging, commit or publication.

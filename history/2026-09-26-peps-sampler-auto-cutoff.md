# 2026-09-26 — Automatic PEPS sampler cutoffs and precision validation

- Scope: user requested small-case checks using cutoff="auto" and
  cutoff_mode="auto", consistent with MpsOptimizer for complex64/complex128.
- Branch / baseline: `develop` / `80f451a`, ahead 8 at start.
- Commit status: working-tree edits only; nothing staged, committed, or
  published. Existing and unrelated edits preserved.

## Implemented

- Sampler defaults to the shared dtype cutoff policy: complex64=1e-6,
  complex128=1e-12, with automatic rsum2 cutoff mode.
- Resolves after conversion and on refresh; accepts explicit numeric/mode
  overrides and forwards policy to Quimb future and ket compression, FIT ket
  guesses, and CompBdy. Documents fixed-rank one-site FIT's lack of SVD
  thresholding during its updates.
- Adds numerical policy/precision regressions and updates API/changelog/example.

## Validation

- 32/32 NumPy/CUDA cases passed (1×3, 2×3, 3×2, D=2; both precisions;
  Quimb and DMRG/FIT, with ample caps and active chi_prime=1 truncation).
- Max ample-cap probability error versus exact: complex64 8.07e-8;
  complex128 2.23e-16. Full proposal normalization, sampled/query logs, and
  original amplitudes pass their precision tolerances under active truncation.
- New initial precision selection: 29 passed; includes NumPy/Torch/JAX CPU.
- Full focused sampler suite: 132 passed in 201.95 s, with 25 existing
  Quimb mode/method compatibility warnings.
- Sampler/API/layout: 162 passed, 1 pre-existing metadata failure.
- Smoke: 153 passed, same installed/runtime 0.4.0 versus project 0.5.0 failure
  in test_package_version_matches_installed_distribution.
- Ruff src/tests/example, updated example, whitespace, and local links passed.
  No full-suite success claimed.

See [detailed numerical evidence](../docs/development/notes/peps_sampler_auto_cutoff.md)
and [API policy](../docs/api/sampling/samplers.md#automatic-truncation-policy).
Installed versions/signatures were rechecked and the prior same-task official
upstream audit reused. The obsolete JAX context name in the initial new test
was fixed before the successful rerun. No production backend shim was added.

## Limits and next-session context

Controlled ket and future-boundary examples confirm small branches can lose
proposal support under automatic rsum2; cutoff is a local truncation criterion,
not a bound on global sampling error. No FIT block-size change, native Symmray
sampler, JAX GPU claim, or large-D benchmark is included. Existing Quimb
mode/method compatibility warnings remain. Shared environments and running
jobs were unchanged. The sandbox patch helper failed before file access;
checked replacements through reviewed execution were used for edits.

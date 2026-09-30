# 2026-09-30 — Publish pending PEPS and gate work

- Scope: user requested committing and pushing Pepsy and Pepsy Examples.
- Baseline: Pepsy `develop` at `5316cb0`; examples `main` at `a48c6f6`.
- This entry accompanies the Pepsy publication commit. Git records the final
  commit and remote status; prior handoffs retain their historical status.

## Included changes

- Public automatic gate cutoffs and scale-preserving SU gauge exponent
  tracking, with dense, backend, native symmetry, and gradient regressions.
- Boundary-MPS amplitudes as the sampler default; explicit exact and
  proposal-only modes; scaled amplitude and weight diagnostics.
- Related API/changelog documentation and numerical evidence, plus the
  previously requested removal of six package examples outside
  `MpsMagnetization`.
- The subsequent [proposal-only handoff](2026-09-30-peps-proposal-only.md)
  supersedes the earlier exact-amplitude-default compatibility decision.

## Fresh publication validation

Shared Python 3.12 environment, local package source first, CUDA hidden,
single-thread OMP/OpenBLAS, temporary outputs under `/tmp`:

- Gate, cutoff, SU scale, sampler, efficiency, boundary-amplitude, BP norm,
  public API, and package-layout selection: **541 passed, 3 skipped,
  2 failed** in 460.68 seconds (`/tmp/pepsy-publish-checks.log`).
- One failure was a test module collected before the concurrent update
  explicitly selected exact amplitudes in
  `test_peps_sampler_absorbs_only_after_complete_row_and_respects_ket_chi`.
  Its final standalone rerun passed. The final boundary-amplitude and
  exact-reference selection also passed **15 tests** after that update
  (`/tmp/pepsy-publish-final-amplitudes.log`).
- The remaining failure is the known installed-distribution metadata
  mismatch: 0.4.0 installed versus project 0.5.0. The smoke profile likewise
  reported **92 passed, 1 failed**, solely this mismatch. No shared
  installation was changed to hide the failure.
- Full `ruff check src tests` and staged whitespace checks passed.
- This is focused numerical plus smoke validation, not the entire package
  suite or new GPU validation. Prior GPU limits remain in the domain notes.

Examples publication commit `aeb3367` includes package DMRG defaults,
PEPS diagnostics/reference tooling, and notebooks. Its complete benchmark
suite reported **518 passed, 6 skipped, 4 known bubble compatibility
failures**; changed-file lint and notebook validation passed. See its
`2026-09-30-publication-dmrg-peps.md` for details.

No production simulations were launched, restarted, or stopped. Runtime
archives/caches and device-local instructions are excluded from publication.

# 2026-09-26 — Real 4×4 D=4 OBC sampler validation

- Scope: user's request for an end-to-end real-size PEPS sampling test after
  the cache audit.
- Branch / baseline commit: `develop` / `a13031b`; all preceding uncommitted
  sampler/cache/rho work preserved.
- Commit status: working-tree edits only, no staging, commit, or push.

## Work and findings

- Added `tests/test_peps_sampler_4x4.py`, marked integration/slow/sampling.
  It uses a full random entangled 4×4 qubit PEPS, D=4 on all 24 OBC bonds,
  and a separate dense 65,536-amplitude single-layer oracle.
- Drew 8,192 main samples: NumPy exact-limit 4096, Torch complex64 2048,
  DMRG future 1024, and small-cap absolute-repair 1024. Checked every returned
  complex amplitude, proposal replay, weights, 41 diagonal observables,
  first-row joint distribution, norm, ESS, reproducibility, boundary caps,
  source immutability, and unchanged shared future caches.
- Exact-limit χ=1024/χ′=16/cutoff=0 agrees in log probability to 3.55e-14 over
  all 4096 draws. Rare/extreme configurations down to p=1.39e-12 agree to
  1.40e-11. Effective sample fractions: 100%, 99.834%, 99.857%, and 69.582%.
- Full-state measurements reveal that χ=256 can still truncate temporary
  ket/bra-layer boundaries despite fitting the final rank. Installed Quimb's
  layer loop confirms why; cutoff=0 alone does not avoid this cap truncation.
  Updated the API explanation and corrected the stale row-scaling caveat.
- Actual row-cache reuse/refreshed-source comparison tested separately on the
  same D=4 state with χ=0/χ′=2 and a safe 64 MiB budget. Positive-χ production
  paths correctly reject oversized dense transfers before allocation.
- Positive-χ future-cache replacement after a source filter and refresh also
  matches a fresh sampler; filtered amplitudes match the dense oracle.
- No runtime implementation, dependency, GPU/production job, or sibling-repo
  change. Details, measured uncertainties, and limitations are in the
  [validation note](../docs/development/notes/peps_sampler_4x4_validation.md).

## Validation

- Main integration invocation (four statistical cases, actual row cache,
  and three native backend parity cases): **8 passed**, 6 existing Quimb
  `mode`/`method` warnings, 347.64 s. Together with the two separate additions,
  all **10 new integration tests passed**.
- Rare/extreme configuration selection: 1 passed, 8 deselected, 1 existing
  Quimb warning, 9.99 s.
- Positive-χ future refresh: 1 passed, 9 deselected, 1 existing Quimb warning,
  4.50 s. These two checks were added after the main process collected tests
  and therefore ran separately; the module now contains ten tests.
- Ruff (`src tests`), new documentation links/catalog entries, and
  `git diff --check`: passed.
- The previous task's 236-test sampler suite is earlier evidence from the same
  runtime implementation, not a newly rerun full suite. Its known API/smoke
  version-metadata mismatch remains recorded in that earlier handoff.

## Limits and provenance

- One random state and filtered variants; CPU NumPy/Torch/JAX. Not a proof for
  arbitrary PEPS, all finite-cap proposal support, GPU throughput, Symmray,
  or off-diagonal observable estimators. The absolute-repair run required
  only roundoff-scale spectral correction; deterministic negative-spectrum
  coverage remains in the previous rho suite.
- Main measurements use unbiased target-normalized importance means with an
  independently computed exact norm and six-standard-error acceptance limits.
  All observed diagonal/marginal errors were below three standard errors.
- Installed dependency versions and relevant public contraction signatures
  were rechecked; no upstream patch or environment update was needed.
- `apply_patch` was attempted but failed at sandbox startup with bubblewrap
  `mountinfo path is not absolute`; approved shell writes and checked Python
  replacements were used. Temporary logs/scripts are under `/tmp`.

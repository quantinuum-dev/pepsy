# 2026-09-30 — Proposal probability validation and bounded PEPS chunks

- Scope: user requested trustworthy proposal-only sampling with optional
  amplitude corrections, and review/fixes for chunk memory and inefficiency.
- Branch / baseline: `develop`, `fb31c15`.
- Commit status: working-tree edits only; no staging, commit, or publication.
  Existing sampler backend documentation, the proposal-only handoff additions,
  and untracked backend-audit tests were preserved.

## Changes

- Kept `amplitude_mode="none"` as the explicit no-amplitude path. Documented
  that q is the product of conditionals; sqrt(q) has neither the PEPS phase
  nor its physical norm. Boundary/exact amplitudes remain optional corrections;
  the existing default remains boundary amplitudes.
- Fixed equal-weight result access bypassing validation of saved proposals.
  Mismatched lengths, invalid mantissas/exponents, and zero-probability saved
  draws now raise even in none mode. Scaled/log probabilities avoid underflow.
- Added opt-in `chunk_size="auto"` to collected and streamed draws: at most
  32 groups, reduced to fit the estimated row-cache budget after the initial
  retained row. Exact or uncached proposals use one history at a time.
  An insufficient single-row budget keeps the existing reference fallback.
  Diagnostics record requested and resolved sizes. Existing integer/None
  behavior and defaults are preserved.
- Unsplit exact/reference groups transfer ownership instead of copying the
  network at each site. Completed conditional networks and group boundaries
  are released before optional amplitude contractions; the existing single
  boundary diagnostic snapshot remains. Streaming also releases discarded
  result objects before constructing the next chunk.

Implementation: [sampler](../src/pepsy/sampling/peps.py),
[results](../src/pepsy/sampling/results.py),
[regressions](../tests/test_peps_sampler_chunks.py), and
[API guide](../docs/api/sampling/samplers.md).

## Bounded measurement

A warmed NumPy complex128 3x3 D=2 proposal-only probe used Quimb-MPS futures,
chi=16, chi_prime=4, seed 17, 64 shots, greedy planning, and a 67,040-byte
row-cache budget that fits two estimated histories. Compared the baseline
module from Git with the working tree. One thread per BLAS/OpenMP library.

| Case | Seconds | Python-tracked peak bytes | Network copies | Peak groups |
| --- | ---: | ---: | ---: | ---: |
| Baseline, chunk 32 | 1.156 | 2,281,121 | 952 | 32 |
| Updated, chunk 32 | 1.055 | 1,419,758 | 763 | 32 |
| Updated, auto (2) | 1.944 | 556,903 | 1236 | 2 |

The two fixed-32 cases returned identical configurations and log probabilities.
Automatic chunking used factored caches; fixed 32 used the reference fallback.
These single-run tracemalloc measurements include tracing overhead and are
not process/GPU peak-memory or production speed claims. Smaller chunks reduced
memory but were slower here. Script/results: `/tmp/pepsy_chunks_probe.py` and
`/tmp/pepsy_chunks_probe.json`.

## Dependency audit

Inspected installed signatures for Quimb network copy (shallow by default),
MPO apply (non-inplace), MPS compression and boundary contraction. Versions:
Quimb 1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305,
Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1,
Symmray 0.4.1.dev7+g83fb22865, Torch 2.6.0+cu124.

**Adopt:** existing public copy/contraction APIs with explicit ownership and
cutoff policy; no shim or dependency change. Checked the
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray).
The linked Symmray abelian-array page failed to load; this dense-only task
does not change Symmray dispatch. **Defer:** native tensor-axis sampling and
hard device-memory limits. The automatic size is a conservative estimate.

## Validation

- Pre-edit efficiency selection: 62 passed (118.43 s).
- New chunk/probability regressions plus boundary-amplitude selection:
  48 passed (6.11 s), including NumPy/Torch, DMRG/Quimb-MPS, independent
  likelihood replay, source isolation, streaming reproducibility, forbidden
  amplitude evaluation, boundary lifetime, and extreme probability scales.
- Final new-regression selection, including streaming lifetime: 36 passed
  (4.19 s). Full `ruff check src tests` and `git diff --check` passed.
- Smoke: 92 passed, one pre-existing installed-version mismatch
  (`test_package_version_matches_installed_distribution`: installed 0.4.0
  versus project 0.5.0). No environment installation was changed.
- Broader direct-sampler, efficiency, new chunks, amplitude, shared sampler,
  public-API and package-layout selection: **510 passed, 2 skipped, 1 failed**
  (469.06 s). The only failure was the same installed-version mismatch.
  Both skips require a second JAX device. JAX was explicitly run on CPU;
  Torch coverage in this selection was CPU. Log:
  `/tmp/pepsy-sampler-chunks-tests.log`. The final streaming-lifetime regression
  was added after broad collection and passed in the 36-test final selection.
  No new GPU or full-package-suite result is claimed.

No production simulation or full 4x4 statistical benchmark was launched.

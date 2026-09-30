# 2026-09-24 — CI dependency-profile repair

- Scope: diagnose and repair CI failures after the prior import cleanup.
- Branch / baseline: `develop` / `9d6d7ef`.
- Commit status: included in the commit containing this entry on `develop`;
  hosted verification follows publication.

## Findings and changes

- The [prior handoff](2026-09-24-compatibility-import-audit.md) recorded hosted
  failures despite a passing development-version suite. The CLI still resets
  API connections; the connector exposes job status but not individual logs.
  No browser was connected. Public job pages supplied step-level annotations.
- Reproduced failures with released dependencies installed under `/tmp`, using
  the selected interpreter and isolated import paths. The shared environment
  was not modified. The released core baseline had 158 failures, 2806 passes,
  and 927 skips.
- Fixed unsupported seed forwarding, a single-precision upstream compile
  failure, optional test prerequisites, Python 3.10 TOML parsing, and CI profile
  configuration. See the [dated compatibility assessment](../docs/development/notes/ci_compatibility_2026_09.md)
  for versions, capability decisions, and numerical contracts.
- Public algorithms retain their accuracy settings. The tree eigensolver
  fallback warns and preserves dtype, cutoff, cutoff mode, and rank limit.
  Tests require actual upstream features and exercise missing-feature errors.

## Validation

- Repaired isolated released core full collection: **2858 passed, 1037 skipped,
  zero failures** in 62 seconds.
- Development focused numerical selection: **1643 passed, 4 skipped**; one new
  test initially used `bond_size` instead of `bonds_size`. Corrected the test
  assertion and reran it successfully.
- MPI unit suite plus that corrected assertion: **58 passed**.
- Real MPI integration: **25 passed per rank**, with both two and three ranks.
- MPI benchmark smoke checks completed with both two and three ranks. A cold
  threaded run first stalled in Stim/NumPy native initialization (confirmed by
  process sampling). The benchmark now warms one serial shot per rank before
  its timed threaded work. General cold threaded startup is not claimed fixed.
- Isolated mypy 2.3.1 with Python 3.12 target: no issues in both checked files.
- Forced the Python 3.10 TOML-reader fallback and parsed project metadata.
- Full development-version run: **4614 passed, 121 skipped, zero failures**
  in 561 seconds, with **71.69% coverage** (above the unchanged 60% extended
  gate). Final legacy-backend capability refinements are checked separately
  after this run: **208 passed, 17 skipped** in the final development selection.
- Broader released Quimb/Cotengra/Autoray/Symmray, with installed Torch/JAX
  integrations: first run **4529 passed, 198 skipped, 8 failed**. The remaining
  failures were NumPy-only upstream SRC noise, old Torch randomized-SVD dtype
  handling, missing SDC tests, and the norm-ledger test's upstream rank-read
  spy. The repaired focused selection passed **189 tests, 20 skipped**.
- Ruff, whitespace checks, and **25 local Markdown links** passed. Wheel and
  source distribution built successfully and passed `twine check`. Missing
  coverage and wheel tools were supplied under `/tmp`, not installed into
  the shared environment.
- Final full released-dependency run: **4535 passed, 200 skipped, zero
  failures** in 617 seconds, with **73.01% coverage**. This uses released
  Quimb/Cotengra/Autoray/Symmray and the installed optional Torch/JAX stack;
  it is not an exact reproduction of a fresh Linux runner's optional versions.

## Limits and follow-up

- Local MPI checks use the existing Mac runtime; Linux runner results must be
  checked after publication. A Python 3.10 interpreter was not run locally.
- Passing development dependencies alone does not establish released-profile
  compatibility. Preserve separate results for those environments.
- Check the new hosted run against the published commit; the earlier
  `9d6d7ef` run finished with failed numerical/type/MPI jobs and passing
  package, documentation, and agent-guidance jobs.

# 2026-09-28 — Fix PEPS optimizer targets and sampler result ownership

- Scope: user authorized the proposed optimizer and sampler fixes following the
  [careful review](2026-09-28-peps-optimizer-sampler-review.md).
- Branch / baseline: `develop` / `6486a61`.
- Commit status: working-tree edits only; nothing staged, committed, or
  published. Existing launch journals and active experiment jobs preserved.

## What changed

- Two-site PEPS targets are built without cutoff, bond cap, or path compression.
  Conflicting explicit target gate overrides raise. Warm-start compression
  forwards the selected cutoff mode through normal and routed paths.
- Non-finite and materially negative infidelity estimates raise; tiny negative
  roundoff still clips to zero.
- Prefix-grouped direct PEPS samples now return an independent configuration
  list per shot, preserving shared contractions and amplitude evaluation.
- Added observable numerical and result-ownership regressions; updated owning
  API guides and the changelog.

## Evidence and validation

- Numerical reproduction, upstream audit, and accuracy result:
  [dated note](../docs/development/notes/2026-09-28-peps-optimizer-sampler-fixes.md).
- Focused new optimizer regressions: initial attempt 8 passed, 1 failed due
  the test wrongly equating the installed `tensor_split` default with the
  implicit `PEPS.compress_all` behavior. The corrected test compares explicit
  abs compression with the actual implicit compression.
- Closest complete suites: `python -m pytest -q -o addopts=''`
  `tests/test_optimize_peps.py tests/test_peps_sampler.py` → **299 passed,
  2 skipped**, 43 upstream/compatibility warnings in 223.42 seconds.
  Includes NumPy, Torch, JAX, and available native Symmray paths.
- Duplicate-shot NumPy/Torch/JAX selection: 3 passed.
- Full repository run before the final test correction: **5177 passed,
  105 skipped, 2 failed** in 1529.07 seconds. The failures were the
  pre-existing Pepsy installed-version mismatch (0.4.1 metadata versus
  0.5.0 checkout) and a stale 4x4 cache test assertion. The latter was
  confirmed in isolation, then made to select its intended dense row-cache
  mode explicitly; it passed in isolation afterward (1 passed, 2.63 s).
  No implementation changed after the whole-repository run. The full suite
  was not repeated after that test-only correction.
- `python -m ruff check src tests`, `git diff --check`, and local Markdown
  link checks: passed.

## Remaining limits

- The documented extreme-scale exact-draw limitation is outside this fix.
- Production-size PEPS performance is not verified.
- The `apply_patch` sandbox helper failed with the same bubblewrap mountinfo
  error as earlier sessions. Exact-match checked replacement was used for
  tracked-file edits. The failure did not change test or production processes.

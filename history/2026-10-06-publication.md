# 2026-10-06 — Publish stabilizer defaults and Helix benchmark records

- Scope: user explicitly requested committing and pushing Pepsy and
  `pepsy_examples`.
- Pepsy branch: `develop`; starting local HEAD `d41ea9f`, remote after fetch
  `8f7c896`. Existing local commits `c649aaa` and `d41ea9f` are preserved.
- New local commits: `5452315` changes the effective disentangling default;
  `2713ecf` records native-collapse reviews and Helix benchmarks.
- Publication target: `origin/develop`. This handoff accompanies the merge
  of incoming remote changes before the requested push.

## Merge and validation

- Preserved both histories; incoming work concerns JAX host solvers, MPO
  compression, and backend exports. No source conflict occurred.
- Resolved the sole conflict in `tests/test_peps_sampler_4x4.py` by keeping
  the explicit `row_cache_max_bytes=0` reference so the cache test continues
  to compare cached and uncached samplers.
- Post-merge focused checks: **703 passed, 8 skipped**, 24.54 seconds.
  Selection covers public API/package layout, stabilizer measurement/sampling/
  trajectories, gradient solvers including JAX, MPO automaton/delinearization,
  and the conflicting 4x4 PEPS cache test. Temporary log:
  `/tmp/pepsy-publication-merged-checks.log`.
- Ruff, skill catalog, and whitespace checks passed. Prior complete suite
  before this remote merge: **7214 passed, 526 skipped**; a complete suite was
  not repeated after the merge. No numerical implementation edits were made
  during publication. No-shortcut numerical failures remain documented in
  [the layout benchmark](2026-10-06-helix-layout-benchmark.md).

## Examples repository

- `pepsy_examples` had no local changes or unpublished commits. Fast-forwarded
  `main` from `4bb34f2` to existing remote commit `fbdacf1`; `git push origin
  main` confirmed `Everything up-to-date`. No empty commit was created.
- The device-local Pepsy `AGENTS.override.md` and temporary benchmark artifacts
  were excluded from staging.

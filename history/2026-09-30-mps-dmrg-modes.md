# 2026-09-30 — Remove MPS DMRG1; keep DMRG strictly one-site

- Scope: user clarified that `MpsOptimizer` should expose DMRG, DMRG2 and DMRG3,
  with DMRG doing one-site refinement only. This supersedes the earlier
  [same-day alias change](2026-09-30-mps-dmrg1-alias.md).
- Baselines: Pepsy `develop` / `f01f596`; examples `main` / `3b61b7d`.
- Status: working-tree edits only; nothing staged, committed or published.
  Unrelated pre-existing changes and notebook outputs were preserved.

## What changed

Removed `dmrg1` as an accepted MPS mode and its latch diagnostic. Constructor,
mode changes, per-run mode selection and shot overrides give a migration error.
`dmrg` (including the old `fit` spelling) rejects multi-site block overrides;
DMRG2/3 retain their existing schedules. Mixed history uses `guess_direct_dmrg`.
Updated mode tests, public docs, changelog, MPS/FIT skills/catalog and the
selected MPS notebook. Multi-site regression cases now use DMRG2/3. Other
optimizer classes are unchanged; an MPO documentation comparison is clarified.

See [implementation and audit evidence](../docs/development/notes/2026-09-30-mps-dmrg-modes.md).
The same-task installed dependency/upstream audit is reused without changes.

## New validation

Shared Python 3.12 environment, local Pepsy source first, single numerical CPU
thread, CUDA hidden; no environment modifications.

- `pytest -q -ra -o addopts='' -m 'not slow' tests/test_optimize_mps.py
  tests/test_mps_*.py tests/test_public_api.py tests/test_package_layout.py
  tests/test_sampler.py`: **1084 passed, 43 skipped, 30 deselected,
  1 existing failure** in 50.45 s.
  The failure remains `test_package_version_matches_installed_distribution`
  (installed metadata 0.4.0 vs project 0.5.0). Skips concern CUDA/CuPy and Metal;
  slow stress checks were deliberately excluded in this follow-up.
  Log: `/tmp/pepsy-dmrg-modes-domain.log`.
- `tests/test_symmetric_tensors.py -k 'mps_optimizer and (dmrg or two_site)'`
  with the same marker/options: **11 passed, 193 deselected** in 3.73 s.
  Log: `/tmp/pepsy-dmrg-modes-native-final.log`.
- Confirmed the obsolete Symmray rejection test also failed against committed
  `f01f596` before replacing it with native/fidelity checks; see evidence note.
  Baseline log: `/tmp/pepsy-dmrg-modes-native-baseline.log`.
- `ruff check src tests`, both changed skills' `quick_validate.py`, and the
  12-skill catalog validator pass. Notebook JSON/code parsing passes with no
  `dmrg1` in cell sources; existing outputs are unchanged. Both repositories
  pass `git diff --check`.

The full package suite, accelerator checks, slow stress cases and notebook
simulation were not rerun. Earlier alias-task counts remain historical and
are not validation of this final mode-removal policy.

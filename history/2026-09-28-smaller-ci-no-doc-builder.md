# 2026-09-28 — Remove docs builder and narrow routine CI

- Scope: remove the Sphinx/ReadTheDocs builder, keep user-facing docs as linked
  Markdown, preserve opt-in backend dependencies, and reduce routine pytest.
- Branch / baseline commit: `develop` at `6486a616`.
- Commit status: uncommitted working-tree changes.

## What changed

- Removed the Sphinx CI job, Sphinx configuration, ReadTheDocs configuration,
  and the documentation dependency extra. Markdown docs and their indexes stay
  in the repository.
- Push/PR CI now uses the same `smoke` pytest tier as `python -m pytest -q`.
  The full suite and optional backend matrix remain in nightly CI.
- Added a package metadata check that Torch, JAX/NetKet, Symmray, Stim, MPI,
  solver/layout, Guppy, and plotting dependencies stay out of the base install.

## Validation

- `python -m pytest -q -ra`: **163 passed, 1 skipped** in 34 seconds.
- Ruff and the focused mypy checks passed.
- CI YAML and project TOML parsed; changed Markdown local links resolved;
  `git diff --check` passed.
- No Sphinx build was run; the builder was removed by request. Hosted Actions
  status was not checked in this session.

## Scope retained

- Optional runtime backends remain available through explicit feature extras.
- Domain and slow tests were not deleted; they remain selectable locally and
  in the nightly workflow. Only the push/PR suite was narrowed.

## Follow-up: remove unused aliases and benchmark artifacts

- Removed duplicated backend exports from `pepsy.tensors` and
  `pepsy.tensors.core`; backend helpers are exported by `pepsy.backends`.
- Removed unused aliases for `experimental.mera`,
  `QMeraParametricEnergyOptimizer`, `MpsStabSampler`, `StabilizerMps`,
  `SpinfulFermionHubbard`, and `hrps_to_ttn`. Kept aliases with current
  downstream/example users and documented the removals in the migration guide.
- Preserved the user's deletion of `benchmarks/peps_sampling.py`; removed only
  ignored Sphinx-generated docs output. The `examples/` directory remains.
- Focused API/layout/import checks: **77 passed**. Three additional sampler,
  stabilizer, and qMERA checks: **3 passed**. Ruff and `git diff --check`
  passed.
- Commit status: being committed on `develop`; no push requested or performed.

## Post-commit verification

- Full suite with `MPLBACKEND=Agg`: **5,163 passed, 129 skipped, 1 failed**.
  The only failure was the dense transfer-cache regression test relying on the
  default row-cache mode, which is `factored`. The test now explicitly requests
  `row_cache_mode="dense"`, matching its transfer-mode assertions; isolated
  rerun: **1 passed**. The full suite was not rerun after this test-only fix.
- A full run without `Agg` aborted in Matplotlib's macOS GUI backend during a
  schematic test; the `Agg` run passed that test.
- `python -m ruff check src tests` passed before the test-only fix; focused Ruff
  on the corrected test and `git diff --check` passed afterward.
- Commit status: follow-up test correction ready to commit on `develop`; no push.

## Test-tier follow-up

- Kept all backend and domain regressions in the optional/nightly collection,
  but removed the 69-test backend matrix from the default smoke selection.
  The backend tests remain in the broader `core` and `optional` marker tiers.
- Updated the smoke-test guidance in `CONTRIBUTING.md` to make this boundary
  explicit.
- The corrected full-suite rerun completed: **5,164 passed, 129 skipped** in
  13m08s. The earlier 1-failure/not-rerun statements above describe the state
  before this successful rerun.

## Final review before publication

- Scope: the user approved reviewing and committing the remaining test-tier
  changes, running smoke tests, and pushing `develop`.
- Baseline: `ec6fd1b`, which includes the separately validated MPS refactor
  recorded in [its handoff](2026-09-28-mps-execution-extraction.md).
- Corrected the README's obsolete documentation-builder link and the
  contributing guide's stale claim that the MPI job runs benchmark smoke tests.
- Fresh default smoke run: **89 passed, 2 deprecation warnings** in **26.91s**.
- Full collection: **5,297 tests**. All **69 backend tests** remain collected
  under both `core` and `optional`; nightly clears default `addopts` and retains
  the complete collection, including slow tests. This was a collection check,
  not another full-suite execution.
- Ruff, the two CI mypy targets, the skill catalog, and `git diff --check`
  passed. Routine CI still installs only base dependencies and check tools.
- These test-tier and documentation changes are included in the commit
  containing this update. Publication is authorized; hosted CI results must
  be checked after the push and are not established by these local checks.

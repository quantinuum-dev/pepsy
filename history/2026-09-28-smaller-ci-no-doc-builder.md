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

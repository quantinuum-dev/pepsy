# 2026-09-24 — Python 3.12 baseline and CI diagnosis

- Scope: fix remaining CI issues; user explicitly selected Python 3.12+ and
  requested aligned packaging, CI, and documentation.
- Branch / baseline: `develop` / `0b63985`.
- Commit status: Python baseline committed and pushed as `ff77274`; the
  dependency repair below is a follow-up on `develop`.

## Changes and evidence

- Raised `Requires-Python` to `>=3.12`, removed the development TOML backport,
  and aligned Ruff, mypy, release builds, and maintained installation docs.
  The CI matrix now has one core and one comprehensive Python 3.12 profile.
  Default local pytest still uses the smoke selection.
- The prior hosted run passed core 3.11/3.12, MPI, packaging, docs, type
  checks, and agent guidance. Python 3.10 exited during collection (code 2).
  The Python 3.12 extended suite independently failed after 29 minutes.
  Raising the Python minimum does not establish that the extended failure
  is fixed.
- GitHub's authenticated raw-log endpoint repeatedly reset CLI connections;
  the connector exposed statuses but did not return log contents. Public job
  pages showed only exit codes. Added escaped pytest failure annotations to
  make exact tracebacks visible on those pages in the next run. The extended
  job stops at its first failure; successful runs retain full collection and
  the existing 60% coverage gate.

## Local validation

- Combined core/smoke selection: **1671 passed, 8 skipped** on Python 3.12.
- Wheel and source distribution built and passed `twine check`; wheel
  metadata requires Python 3.12+, excluding 3.10/3.11.
- Focused mypy, Ruff, workflow YAML/matrix checks, changed-document links,
  and whitespace checks passed.
- An isolated deliberately failing pytest run retained its failing exit code
  and emitted one escaped annotation under `GITHUB_ACTIONS=true`; the same
  test emitted no annotation outside CI.

## Remaining work

- Hosted validation of the dependency repair remains pending. The Python
  baseline run passed every other job. Do not infer hosted success from
  local checks or remove upstream numerical compatibility guards solely
  because the Python minimum changed.

## Extended-suite diagnosis and repair

- Recovered the previous run's raw logs using authenticated HTTP/1.1 to get
  the log archive redirect, then downloading the archive without forwarding
  credentials to the storage host. The earlier CLI connection failures were
  a log-access problem, not evidence about the test failure.
- Run `36084946424` had eight failures, all during `import netket` in
  `test_netket_flat_z2.py` and `test_vmc_api.py`. Whole-package coverage was
  **72.36%**, above the 60% requirement.
- NetKet 3.22.4 evaluates `"LocalEstimators" | jax.typing.ArrayLike` at
  import time. JAX 0.11.1 switched `ArrayLike` from `typing.Union` to a
  runtime union; the mixed string/union expression raises `TypeError` on
  Python 3.12. Confirmed in published wheels and official sources:
  [NetKet declaration](https://github.com/netket/netket/blob/v3.22.4/netket/_src/stats/online_stats/operations.py),
  [JAX 0.11.0](https://github.com/jax-ml/jax/blob/jax-v0.11.0/jax/_src/basearray.py),
  [JAX 0.11.1](https://github.com/jax-ml/jax/blob/jax-v0.11.1/jax/_src/basearray.py).
- All five NetKet 3.22 releases contain the declaration, but they import
  with the installed JAX 0.8.2. Excluding NetKet versions alone would be
  misleading: NetKet 3.21 also fails with current JAX, on a removed
  `jax.core.concrete_or_error` import.
- Classified as a **dependency compatibility constraint**: limit JAX to
  `<0.11.1` in `vmc-netket`, inherited by `vmc`. Keep NetKet itself unpinned
  and revisit the JAX bound after an upstream annotation fix. No vendor
  patch or numerical code change is needed.
- Added an extended-CI import check for JAX, NetKet, Symmray, and Torch before
  numerical tests, so incompatible installations fail promptly.
- Isolated `/tmp` installs, using the required Python 3.12 interpreter,
  reproduced NetKet 3.22.4/JAX 0.11.2's exact `TypeError` and successfully
  imported NetKet 3.22.4/JAX 0.11.0. The shared environment was unchanged.
- Fresh resolved NetKet/JAX dependencies plus released Quimb 1.15.0,
  Autoray 0.11.0, Cotengra 0.8.2, and Symmray 0.4.0: both affected test
  modules passed (**57 passed**). Public API/package layout checks passed
  (**58 passed**); Ruff and whitespace checks passed.
- Wheel/sdist builds and `twine check` passed; wheel metadata retains
  `Requires-Python: >=3.12` and excludes JAX 0.11.1/0.11.2 for NetKet VMC.
  A broader local released-dependency run is in progress; it is not yet a
  passing full-suite result.

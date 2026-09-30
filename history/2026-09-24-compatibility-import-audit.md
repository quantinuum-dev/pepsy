# 2026-09-24 — Compatibility import audit and hosted CI

- Scope: commit/push the reliability fixes, check hosted CI, and migrate safe
  internal imports from `tensors.core` to their owning modules.
- Branch / baseline commit: `develop` / `69a85b0`.
- Commit status: reliability changes committed and pushed as `69a85b0`;
  this follow-up is recorded in the separate commit containing this entry.

## Changes and decisions

- The [previous reliability handoff](2026-09-24-reliability-and-stream-parsing.md)
  described uncommitted work. That work is now committed and published.
- Changed five plain re-export import sites: symmetric mapping uses
  `tensors.maps`, symmetric measurements and MPS entropy use
  `tensors.observables`, and MPO norms use `tensors.contractions`.
- Kept the `tensors.core` wrappers for optimizer construction, fidelity, and
  hypercompressed contraction. Their historical patch hooks make them more
  than aliases; migrating those callers requires a separate compatibility
  decision. No public exports were removed.
- Hamiltonian mapper documentation and its type error now use the public
  `pepsy.tensors.OneDMap` spelling. Updated the
  [ownership guide](../docs/development/package_layout.md).
- Added a fresh-process regression that actually resolves symmetric chain
  and MPO mappings without loading `tensors.core`.
- This reduces import coupling. No package-size or runtime speedup is claimed.

## Validation

- Focused API, layout, import, symmetric tensor, entropy, MPO, and contraction
  selection: **412 passed, 1 skipped** before adding the fresh-process test.
- Standalone fresh-process mapping probe passed.
- Full suite, including the new import regression: **4610 passed, 121 skipped,
  zero failures** in 314 seconds. Command:
  `MPLBACKEND=Agg python -m pytest -q -o addopts=''`.
- Ruff, whitespace checks, and all three local documentation links passed.
- The existing upstream audit and unchanged development environment from the
  previous handoff apply; this pass changes imports, not numerical algorithms.

## Hosted CI findings

- [Run 36078364391](https://github.com/quantinuum-dev/pepsy/actions/runs/36078364391)
  checks the exact published reliability commit `69a85b0`.
- Docs, package, and agent-guidance jobs passed. Python 3.10/3.11/3.12 core,
  MPI unit-contract jobs (both matrix entries), and focused type checking
  failed. Extended and VMC jobs were still running at inspection.
- GitHub CLI requests repeatedly failed with connection resets. The GitHub
  connector could read job status, but does not support the individual job
  log or check-annotation endpoints. Failure causes remain unverified.
- Local type-check reproduction was unavailable: the selected environment
  does not have `mypy`. No dependency was installed into that shared environment.
- Next priority is diagnosing hosted CI against its released-dependency
  profiles. Passing the local development-version suite does not establish
  that those CI profiles pass.

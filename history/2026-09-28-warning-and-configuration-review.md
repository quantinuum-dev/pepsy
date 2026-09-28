# 2026-09-28 — Warning cleanup and configuration checks

- Scope: user's requested next steps after the readability pass: commit the
  pending batch, triage Pylint and 790 warnings, check skipped configurations,
  and document BP/MPO/PEPO numerical contracts.
- Branch / baseline: `develop` / `5ae50de`. That baseline is the separate
  local commit of the previously pending readability and diagnostics changes.
- Commit status: this entry accompanies the separate local commit titled
  `Fix avoidable warnings and document numerical contracts`.
  No push or hosted CI run.

## Changes

- Safe ownership of read-only NumPy storage in Torch conversion.
- Installed Quimb keyword selection for cold PEPS sweeps and MPS trajectory
  environments; canonical Hamiltonian builder calls in routine tests.
- Narrowed BP factorization fallback with a regression that preserves
  singular-PSD recovery while exposing unrelated type errors.
- BP helper/accessor docstrings, expectation shapes/normalization/ownership,
  MPO physical axes, and PEPO physical transpose and materialization contracts.
- Test figure cleanup and detached scalar comparisons preserve warning
  visibility. No dependencies, source modules, smoke selection, CI jobs,
  Sphinx tooling, or benchmark directory were added.

## Review and evidence

The [maintenance triage report](../docs/development/notes/2026-09-28-maintenance-triage.md)
contains the warning-cause inventory, exception decisions, ranked large
functions, upstream capability audit, and remaining numerical questions.
Standard scoped Pylint remains nonzero: **1,795 findings, including 71
error-labelled reports** (previously 1,802 / 71). These include dynamic API
inference reports described in the preceding audit; this is not a clean-lint
or exhaustive-correctness claim.

## Validation

- Focused backend/PEPS/trajectory: **143 passed, 7 skipped**.
- Focused BP/Hamiltonian/MPO: **257 passed**.
- Quimb reference keyword checks: **3 passed**.
- Default smoke: **89 passed, 2 compatibility warnings in 19.62s**.
- Ruff, focused CI mypy, and the 12-skill catalog pass.
- New BP example and normalization/input-preservation checks pass.
- Nonsymmetric matrix probes confirm documented MPO/PEPO physical order.
- MPI integration: **25 passed per rank** with both two and three ranks.
- Two logical CPU JAX devices: **2 passed**.
- Metal/CPU controls: **4 passed, 1 skipped**. Metal scalar-ledger execution
  passed; native Metal QR is unsupported. CUDA and CuPy remain unavailable.
- Full suite: **5,171 passed, 129 skipped, 687 warnings in 523.06s**.
  This precedes one final qMERA test-assertion detach; afterward its module
  passed **103 tests with no warnings**, with scalar-conversion warnings
  promoted to errors.
- The exact-batch test associated with two full-run Loky worker notices
  passed an isolated rerun without warnings. The notices remain unexplained.
- All 31 local Markdown link targets in changed documentation exist;
  `git diff --check` passes.

## Remaining work

Quimb's NumPy shape-assignment deprecation is the dominant upstream warning.
Complex-to-real casts need their own numerical-policy investigation. Torch
boundary fallback diagnostics and the ranked large functions remain future
work. GPU and distributed results are limited to the configurations above;
local tests do not establish hosted CI or multi-node behavior.

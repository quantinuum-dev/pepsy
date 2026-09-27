# 2026-09-27 — Remove obsolete examples and benchmarks

- Scope: preserve the user's intentional deletion of eleven standalone example
  and benchmark files, repair documentation and CI references, and commit.
- Branch / baseline commit: `develop` at `bd7a4bf`.
- Commit status: prepared for the accompanying commit; not pushed by this task.

## What changed

- Removed obsolete standalone operator, PEPS sampling, Pauli-MPO, MPI launcher,
  and MPS compression benchmark files selected by the user.
- Replaced current documentation references with maintained API and test-suite
  guidance. Historical notes retain their original evidence while stating that
  the referenced standalone artifacts were removed later.
- Removed the deleted MPI benchmark invocation from nightly CI while retaining
  two- and three-rank MPI integration tests.

## Validation

- Default smoke suite before the documentation cleanup: 163 passed, 1 skipped.
- Public API/package-layout selection: 59 passed.
- Ruff, workflow YAML parsing, changed-file local links, and `git diff --check`
  passed.
- Strict Sphinx build with warnings treated as errors passed after cleanup.

## Remaining scope

- The full numerical and optional-backend suites were not rerun because this
  change removes development artifacts and updates documentation/CI only.

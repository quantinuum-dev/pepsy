# 2026-09-26 — Fix operator construction and lazy application

- Scope: repair the reported Pepsy operator convention and lazy sub-MPO bugs,
  with the associated Gaugy fixes handled in that checkout.
- Branch / baseline commit: `develop` / `c85e3c32db168faa0608be67a8718c30d528ad93`.
- Commit status: recorded in the fix commit containing this entry; publication
  pending. Only this task's files are included.

## What changed

- Dense gate builders use native Quimb output/input conventions and list order.
- `tns_align` connects state inputs to operator `b...` legs, with explicit
  `transpose=True` for intentionally transposed representations.
- Lazy sub-MPO application works with either `inplace_mpo` choice.
- Added independent dense regression oracles and documented migration.

## Validation

- Focused operator and gate suites: 122 passed, 1 skipped; one existing
  ComplexWarning. This includes native Symmray gate coverage.
- Full suite: 4,644 passed, 121 skipped, 733 warnings in 371.10 seconds.
- Ruff across `src tests`, the two-module CI mypy check, and `git diff --check`
  pass. Public API/package checks are included in the full suite.
- Strict documentation build passes with `sphinx -E -W --keep-going`.
  The initial sandboxed attempt could not resolve official intersphinx
  inventory hosts; the network-enabled rebuild succeeded without warnings.

## Decisions and limitations

- Adopt upstream operator semantics; no installed-library edits.
- Old general gate-built operators should be rebuilt from their streams.
- See [dated evidence](../docs/development/notes/operator_conventions_2026_09.md).

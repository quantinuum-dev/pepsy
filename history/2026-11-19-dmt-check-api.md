# 2026-11-19 — DMT skill, API and check costs

- Scope: user requested updating the DMT skill, revising/fixing the API and
  reducing expensive checks. Preserve their warn-and-continue run preference.
- Branch / baseline: `develop`, `f5d31006d02821028c890a04b17e4b15b9683d0b`.
- Commit status: uncommitted. Prior task edits/artifacts are preserved;
  nothing staged, committed or published. Local overrides were not changed.

## Implemented

- Replay defaults `check_regions=False` (`dmt_` prefix on ordinary replay).
  Explicit full checks now also verify Pauli's default triples.
- Shared strict sweep skips repeated full-region scans but keeps the final
  original/output comparison; relaxed per-bond measurements remain available.
- Scalar density-trace guards cover all paths, including the initial ordinary
  replay sweep and manual compression. Explicit Pauli trace/off choices remain.
- Added report metadata for resolved full-check mode and focused regressions.
  Updated two relaxed reporting tests to request their full measurements.
- Updated DMT skill, operations reference, UI metadata, catalog, manifest and
  handwritten API guides. No mathematical truncation/precision change.

## Evidence and validation

See the [record](../docs/development/notes/2026-11-19-dmt-check-api.md) and
linked timing JSON. With full verification retained, L=64, chi=32 five-site
compression improves 2.893 s to 0.342 s (8.46x), with bitwise-identical cores
and independent regional error <=3.62e-15. Fast ordinary replay is 0.842
versus 3.840 s/step for explicit full checks in this CPU example.

Final public/domain selection: 487 passed, 1 skipped, 1 expected failure,
1 deselected. The deselected installed-version test was run initially and
fails on pre-existing metadata 0.5.0 versus project/HEAD 0.6.0. Subsequent
focused check-cost tests: 15 passed. Skill quick validation, catalog validation,
Ruff and whitespace/link checks pass. No full-suite or GPU validation claim.
Refreshed upstream audit and exact versions are in the note; no dependencies
or installed environment files changed.

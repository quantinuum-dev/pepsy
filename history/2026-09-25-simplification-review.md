# 2026-09-25 — Further package simplification review

- Scope: review opportunities to make Pepsy lighter and better organized.
- Branch / baseline: `develop` / `7bde713`, with the preceding Markdown
  cleanup still uncommitted and preserved.
- Status at review: measurements and recommendations; no implementation or
  workflow refactor performed. This record accompanies the later cleanup commit.

## Current evidence

- `tools/measure_imports.py --repeat 3`: local clean-process medians were
  17.85 ms for root import and 18.47 ms for all core namespace imports.
  Sampling, optimizer, and experimental discovery were also about 18 ms.
  None loaded the optional roots monitored by the profiler.
- `tests/test_import_boundaries.py`: **13 passed**. Timings are local and do
  not measure installed dependency size or numerical execution speed.
- The existing 0.5.0 wheel is 1.97 MiB compressed, 8.78 MiB uncompressed.
  The source tree has 187 Python modules and 240,368 lines, including comments
  and docstrings. File size alone does not establish a defect.
- MPS optimizer: 11,970 lines; symmetric tensors: 11,904; stabilizer MPS:
  9,193; BP series: 8,420. MPS `run` spans 963 lines and `_run_dmrg` 879.
- Eight implementation files still import through `tensors.core`. Its
  contraction/fidelity wrappers preserve patch hooks by temporarily changing
  implementation globals, so changing callers requires a compatibility audit.
- Lazy discovery, composed extras, and a numerical base-wheel smoke test are
  already implemented. Several recommendations in the September 24 assessment
  are therefore complete, rather than new work to repeat.

## Recommended order

1. Remove unused PyPI/TestPyPI publishing inputs, jobs, and OIDC permission
   from the release workflow. Keep tag/manual artifact builds and validation.
   This matches the user's GitHub-only distribution choice.
2. Extract a small MPS responsibility first: timing summaries and layout
   report formatting are concrete candidates. Preserve public methods and
   lazy imports; move replay and normalization only after narrower changes
   pass their relevant checks. Existing empty module paths do not prove a
   subsystem has already been extracted.
3. Migrate remaining internal compatibility imports one subsystem at a time,
   preserving documented public aliases and patch-hook behavior.
4. Separate long API guides by task while retaining their entry pages and
   links. Tree layout, replay, and readout are a useful first grouping.
5. Treat symmetry model/state construction as a later focused extraction.
   Follow the owning domain instructions and numerical invariants first.

Keep the existing responsibility-based package hierarchy and five direct
core requirements. Further installation or runtime savings need measured
dependency or workload evidence; these structural proposals do not establish
such savings. `git diff --check` passed.

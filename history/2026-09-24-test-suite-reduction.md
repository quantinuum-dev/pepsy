# 2026-09-24 — Reduce repeated test work

- Scope: approved CI/test reduction, followed by the requested large-file
  reorganization and further equivalent-case pruning.
- Branch / baseline: `develop` / `34221d8`.
- Status: included in the commit containing this entry on `develop`;
  hosted verification follows publication.

## Changes

- One extended CI job combines the existing optional dependency profiles,
  including VMC, and retains full collection with the 60% coverage gate.
  Removed the separate full-suite VMC job. JAX tests occur across MPS, tree,
  operator, and VMC modules, so selecting only VMC markers would lose coverage.
- Python 3.10/3.11 jobs now select `core and not optional and not slow`.
  Python 3.12 keeps its small smoke job as well as comprehensive coverage.
- MPO direct-mode aliases have four constructor-level mapping checks. The
  four operator-side cases run once through the canonical mode, reducing
  numerical replays from 16 to 4 while preserving their dense and norm oracles.
- Split the two optimizer catch-all suites by responsibility. MPS replay went
  from 9277 to 2328 lines; tree replay from 7678 to 2812. Ten focused modules
  own compression, FIT kernels, native arrays, layouts, state invariants,
  normalization, and control/readout checks. Two small helper modules share
  reference builders without importing another test module.
- Removed `fit` from three numerical mode matrices because it normalizes to
  `dmrg`; constructor and replay alias regressions remain. Retained every
  distinct named DMRG schedule, symmetry, and numerical assertion.
- Classified all 182 cases in the extracted native MPS module as optional;
  each of its 20 test functions explicitly requires Symmray. The final core
  selection contains 1433 cases in the development environment.
- Updated contribution guidance and the test ownership map. Package
  implementation, dependency declarations, and numerical tolerances are unchanged.

## First-pass validation

- Collection comparison: **4735 → 4727 cases** in the development environment;
  only the intended 16-case alias family was replaced by eight cases.
- MPO alignment suite: **21 passed**.
- Default smoke suite: **151 passed, 1 skipped**.
- New core selection, development dependencies: **1582 passed, 7 skipped**.
- New core selection, isolated released core dependencies: **1207 passed,
  394 skipped**. Both local profiles used the selected Python 3.12 interpreter;
  Python 3.10/3.11 remain hosted checks.
- CI YAML, four matrix entries, retained interpreter versions, single full
  coverage command, Ruff, and whitespace checks passed.
- The first local core attempt used Matplotlib's interactive macOS backend
  and aborted in the plotting test. Rerunning with CI's `MPLBACKEND=Agg` passed.
- The full numerical suite was not rerun during the first pass. Earlier
  full-suite results are in the [dependency-profile handoff](2026-09-24-ci-dependency-profiles.md).

## Reorganization validation

- Before pruning, collection snapshots confirmed all 4727 cases and their
  markers survived the split. AST comparisons confirmed all 623 MPS/tree
  test functions were unchanged.
- Final collection: **4717 cases**. Exactly ten repeated `fit` cases were
  removed, with corresponding `dmrg` cases retained. All 623 test bodies
  and assertions remain identical. The only marker changes add `optional`
  to the 182 native MPS cases described above.
- Released core selection after the split and alias pruning: **1207 passed,
  387 skipped**, before explicitly classifying the native MPS module optional.
- Full development numerical run: **4596 passed, 121 skipped, zero failures**
  in 322 seconds. This includes the retained stress matrices. The final
  optional marker refinement changes selection only; its collection snapshot
  was checked separately after this run had started.
- Final Ruff, whitespace, and changed-document link checks passed.

No hosted runtime reduction has been measured. Distinct stress/algorithm
combinations remain in the comprehensive CI job; pruning only demonstrated
aliases does not require a separate scheduled workflow.

# 2026-09-26 — Broad module organization and complex sampling correction

- Scope: user requested a broad assessment and coordinated implementation
  instead of another single small extraction.
- Branch / baseline: `develop` / `3ea56fa`, plus the already validated,
  uncommitted [symmetric MPO extraction](2026-09-26-symmetric-mpo-extraction.md).
- Commit status: included in the commit containing this entry, together with
  the symmetric MPO extraction. The user authorized committing and pushing
  `develop`; push verification is reported in the session handoff.

## Implemented

- Audited the largest 30 Python modules and their responsibilities, import
  dependencies, module maps, and tests. Recorded completed splits and other
  potential boundaries in the [organization audit](../docs/development/notes/module_organization_2026_09.md).
- Split sampler families/results, ordinary MPS controls/norm bookkeeping,
  stabilizer advice/frame-layout/shared stream helpers, and BP loop geometry
  into 12 owner modules. Optimizer helpers receive live owners and retain
  subclass dispatch; no extra state holder or mixin hierarchy was introduced.
- Made BP and stabilizer namespace exports lazy after a fresh-process test
  exposed eager loading of the entire implementation. Preserved public names,
  deprecation warnings, historical helper imports, and old pickle globals.
- Updated internal sampling callers and module-global test hooks to their
  owners. Updated module maps, four domain skills plus the FIT source map,
  API guidance, and changelog; catalog/bundle membership is unchanged.
- Preserved legacy generated sampler and BP class links through a scoped
  AutoAPI hook. Runtime imports alone do not preserve generated deep links.

## Sampling defect discovered during validation

An installed-wheel dense oracle exposed biased native probabilities for a
complex MPS. The previous wheel reproduced the mismatch, so it predates this
refactor. Four Torch/NumPy/CuPy sampling/evaluation contractions reversed the
bra/ket orientation of complex right environments. Corrected them and added
independent dense all-configuration tests for NumPy/Torch with physical
dimensions two and three, sampled-weight checks, seed reproducibility, and
source preservation. Complex sample distributions can change.

## Validation

- AST comparison checks **645** original definitions/methods, including full
  classes in the sampling layer. Only the four documented Born-weight
  expressions intentionally change. Public signatures and optimizer method
  docstrings are retained; two sampler return-type links are qualified to
  their new result owner. Moved method docstring indentation is normalized
  for comparison.
- Initial focused numerical/API suites: **719 passed, 7 skipped**.
- Expanded import/subclass/API/stabilizer/all-BP checks: **654 passed,
  1 skipped**. The first new subclass test used an incomplete private-hook
  signature; correcting that test left production method signatures unchanged.
- Sampling suites after the probability correction: **163 passed, 7 skipped**.
- Import tests cover all sampler owner orders, old sampler/BP pickle globals,
  lazy engine loading, and independent control/advice/geometry imports.
- Ruff, focused CI mypy, skill validators, and the 12-skill catalog passed.
- A full pre-correction refactor run passed **4,628 tests, 121 skipped**;
  the final-code full run is recorded separately below.
- Built an sdist and a wheel from it. All **204** wheel Python modules match
  the final source. An installed rebuilt wheel passed MPS controls/norm,
  stabilizer replay/advice,
  all four sampler engines, a BP rho reference, and old pickle imports outside
  the checkout. Build tools and test installations remained under `/tmp`.
- Strict Sphinx build passed without warnings. All **244** historical API
  anchors checked remain available across the affected optimizer, sampler,
  and BP classes; **7,558** rendered local links/anchors resolve across 21
  pages. Import aliases needed explicit AutoAPI rendering; preserving
  the ordinary undocumented-member filter avoided duplicate field descriptions.
  Two result-type links were qualified to remove ambiguous cross-references.
- Final full suite: **4,632 passed, 121 skipped**, 732 warnings, in **328.02
  seconds**. Skips cover unavailable CUDA/CuPy, sandbox Metal, and
  single-process MPI cases. No new GPU or multi-rank validation is claimed.

## Limits

No dependency upgrades or performance claim. CuPy/GPU and multi-rank coverage
remain limited to available runtimes. The audit distinguishes remaining design
review candidates from known defects; it does not assert a functional audit of
every subsystem. Previous local commits are preserved. The commit handoff
reconfirmed that all 204 Python source files match the validated wheel and
that fetching `origin/develop` introduced no new remote commits. No release
is part of this task.

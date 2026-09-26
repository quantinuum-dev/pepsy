# 2026-09-25 — Extract symmetric state ownership

- Scope: continue the deferred cleanup after the previous maintenance and
  branch-promotion work. Selected the model/state separation identified in
  the [simplification review](2026-09-25-simplification-review.md).
- Branch / baseline: `develop` / `22bff20`, initially clean and synchronized
  with local/remote `main` and remote `develop`.
- Commit status: working-tree changes; no commit or push for this refactor.

## Implemented

- Moved `_SymState`, `SymMPS`, and `SymPEPS` into
  `src/pepsy/tensors/symmetric_states.py`. All three class bodies match their
  pre-extraction ASTs; no algorithm, defaults, gates, or backend conversions
  changed. The old module shrank from 11,904 to 10,494 lines.
- Broader AST review found all **199 existing top-level function/class
  implementations** unchanged except the explicit local state import in
  `SymHamiltonian.jw_energy`.
- Public `pepsy.tensors` exports and internal tensor constructors use the new
  owner. `symmetric.py` keeps model/operator/conversion/diagnostic ownership
  and resolves historical state-class attributes lazily. `jw_energy` loads
  its state type at call time to avoid an eager circular dependency.
- Preserved class identity across public imports, legacy pickle class paths,
  native state metadata, and subclass copying. Model discovery does not
  eagerly load state implementation, and importing either module first works
  with Symmray unavailable.
- Updated the API guide, implementation map, changelog, and dated
  [upstream audit](../docs/development/notes/symmray.md).

## Validation

- Existing symmetric tensors, fermion gate cache, and JW gate suites:
  **226 passed, 1 CUDA skip**.
- Public API, package layout, import boundaries, and subclass checks:
  **75 passed** in the initial selection. Four new serialization cases first
  failed because the test passed a native Symmray value to NumPy; explicitly
  converting the contracted native array fixed the test oracle. The final
  serialization/subclass module then passed **all 5 tests**.
- Ruff, focused mypy, skill catalog, and whitespace checks passed.
- Strict Sphinx HTML build passed with an empty diagnostic log. **1,577**
  rendered local links, including fragments, resolved across the affected
  guide, module map, and generated API pages.
- Built an sdist and a wheel from that sdist. All **188 Python modules** in
  the wheel match working-tree source, including the new state module.
  Installed-wheel native MPS/PEPS construction and pickle round trips passed
  outside the checkout, with the installed import path explicitly checked.
- Build tooling and wheel installation were kept under `/tmp`; the shared
  environment and sibling repositories were not changed.
- Full local suite: **4,614 passed, 121 skipped**, 732 warnings, in
  317.40 seconds. The skips remain CUDA/CuPy, sandbox Metal, and single-process
  MPI cases. This run does not establish new GPU or multi-rank coverage.

## Limits

This separates state behavior from model/operator code; it does not claim a
numerical speedup or complete decomposition of every large module. Hardware
validation remains bounded by this Mac's available backends. No release or
remote branch promotion is part of these working-tree changes.

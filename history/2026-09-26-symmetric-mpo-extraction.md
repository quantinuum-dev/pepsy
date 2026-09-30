# 2026-09-26 — Extract symmetric MPO builders

- Scope: user approved symmetric MPO construction as the next module split.
- Branch / baseline: `develop` / `3ea56fa`.
- Commit status: working-tree edits; no new commit, push, or release. The two
  preceding refactor commits remain local.

## Implemented

- Moved 14 assembly, local-term factorization, and charge-sector helpers into
  `operators/_symmetric_mpo.py` (1,319 lines). `tensors/symmetric.py` shrank
  from 4,152 to 2,899 lines. Shared charge, basis, and coordinate mapping
  helpers remain with the Hamiltonian definitions.
- `SymHamiltonian.to_mpo` and `to_pepo` import builders when called. Fermion
  pair observables and native gate identity construction use the owning
  operator module directly. Historical helper imports and pickle globals
  resolve through lazy aliases in `tensors.symmetric`.
- Updated the tensor/operator module maps, API guide, fermion skill,
  changelog, and dated [upstream evidence](../docs/development/notes/symmray.md).
  Catalog membership and upload files are unchanged.

## Validation

- AST comparison: all 108 existing top-level definitions match, including
  the Hamiltonian class, excluding only the two new local import statements.
  All 14 moved function bodies and signatures are unchanged.
- New clean-process regressions check both import orders, old pickle globals,
  optional Symmray absence, and lazy state/model/diagnostic loading.
- Focused symmetry, native PEPO, gate-cache, JW, API, package-layout, and
  import-boundary suites: **314 passed, 1 skipped**, 12 warnings. An initial
  run aborted in macOS's graphical Matplotlib backend; the successful rerun
  uses the headless Agg backend without production changes.
- Ruff, focused CI mypy checks, fermion skill validation, the 12-skill
  catalog, and whitespace checks passed.
- Built an sdist and a wheel from it. All **192** wheel Python modules match
  source. Installed-wheel checks outside the checkout passed for spinless and
  spinful native/JW matrices, mixed-charge MPOs, direct local PEPOs, compression
  metadata, and historical imports. Build tools stayed under `/tmp`.
- Strict Sphinx build passed without diagnostics. **3,610** rendered local
  links/anchors resolve across 10 pages; all **11** existing `SymHamiltonian`
  API anchors remain present.
- Full local suite: **4,622 passed, 121 skipped**, 732 warnings, in 314.54
  seconds. Skips cover unavailable CUDA/CuPy, sandbox Metal, and single-process
  MPI cases; no new GPU or multi-rank validation is claimed.

## Limits

No numerical algorithm, dependency, or performance change is claimed. Other
large modules remain separate future tasks. Remote branches are unchanged.

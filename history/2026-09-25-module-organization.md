# 2026-09-25 — Organize symmetry and MPS helper modules

- Scope: user approved continuing responsibility-based module splits after
  the [state extraction](2026-09-25-symmetric-state-extraction.md).
- Branch / baseline: `develop` / `22bff20`; the preceding state extraction
  was already in the working tree and is preserved.
- Commit status: local, uncommitted changes; no commit, push, or release.

## Implemented

- `symmetric.py`: 10,494 → 4,152 lines in this pass. Hamiltonians, MPO
  construction, charge conversion, and legacy Hubbard streams remain here.
  `symm_fermions.py` now owns fermion models, parameterized gates, lattice
  metadata, and factories (3,175 lines); `symmetric_diagnostics.py` owns
  summaries and drawings (3,413 lines). The previous state extraction remains.
- `mps/optimizer.py`: 11,835 → 11,124 lines. `_streams.py` owns stream
  snapshots, symbolic resolution, and normalization (461 lines).
  `compression.py` owns stateless Quimb adapters and disposable guesses
  (324 lines). Stateful replay, targets, rollback, normalization, and canonical
  metadata remain together on `MpsOptimizer`.
- Public lazy exports point to the owners. Historical implementation imports
  retain identity, and old pickle class/function paths remain loadable.
  Importing compression helpers does not initialize replay or FIT.
- Updated API guides, implementation maps, changelog, and domain skills.
  Catalog and upload manifest entries still identify the same skill packages
  and reference files; no bundle membership changed.
- Reused and extended the same-task [upstream audit](../docs/development/notes/symmray.md).
  No dependency versions, defaults, or numerical policies changed.

## Validation

- AST comparison: all 196 non-facade symmetry definitions, all 20 MPS
  top-level definitions (including the entire optimizer class), and all
  three extracted state classes are unchanged from this pass's starting tree.
- Public API, package layout, and import boundaries: **76 passed**. The old
  test assuming `compression.py` was empty was updated to reflect its new
  implementation ownership; separate coverage enforces its replay independence.
- Native model checkpoint and stream-plan cache/lock regressions: **3 passed**.
- Symmetry, fermion/JW gates, state layout, MPS replay, and all `test_mps_*`
  suites: **1,110 passed, 32 skipped** (unavailable GPU backends).
- Ruff, focused mypy, both changed skill validators, the 12-skill catalog,
  and whitespace checks passed.
- Built an sdist and a wheel from that sdist. All **190 Python modules** in
  the wheel match source. Installed-wheel dense guesses/replay, native state
  summaries, aliases, and model serialization passed outside the checkout.
  Build tools and installation were isolated under `/tmp`.
- Strict Sphinx build passed with an empty diagnostic log; **3,263** rendered
  local links/anchors resolved across nine affected pages. The first attempt
  failed because sandbox networking prevented external inventory downloads;
  the network-enabled retry retained warnings-as-errors.
- Full local suite: **4,619 passed, 121 skipped**, 728 warnings, in
  313.41 seconds. Skips are unavailable CUDA/CuPy, sandbox Metal, and
  single-process MPI cases; no new GPU or multi-rank coverage is claimed.

## Limits

This is a focused decomposition, not a split of every large file. The remaining
MPS class still couples replay and canonical state closely; other optimizer
families were not reorganized. No runtime speedup is claimed. GPU and multi-rank
MPI validation remain limited to the coverage actually executed in this session.

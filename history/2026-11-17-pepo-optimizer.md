# 2026-11-17 — PEPO optimizer first step

- Scope: add a density-operator gate-stream class using existing PEPO kernels;
  protected PEPO DMT remains a separate, unimplemented next step.
- Branch/baseline: `develop`, `dbc9819e`. Changes are uncommitted. Existing
  unrelated sampling edits were preserved.
- `PepoOptimizer` replays complete sandwiches through local SVD, SU gauges,
  or the existing fused full update. Direct/DMRG boundary fitting is available
  in full update; its cache lives within a run. Symmray and DMT are rejected.
- Added complex dense-reference, lossy, repeated-replay, one-site, direct and
  DMRG boundary tests. `tests/test_pepo_optimizer.py` and
  `tests/test_pepo_joint_update.py`: 47 passed, 8 accelerator tests skipped.
  Public API/package checks: 62 passed, one pre-existing metadata mismatch:
  installed Pepsy 0.5.0 vs checkout 0.6.0. A second focused run including
  full-update, SU and public API checks passed 127 tests. Ruff and
  `git diff --check` passed.
- Installed Quimb 1.15.0, Autoray 0.8.4, Cotengra 0.8.3.dev8,
  Cotengrust 0.2.1, Symmray 0.4.1.dev15. Reused the existing guarded legacy
  Autoray namespace shim for mixed NumPy/Torch runs.
- No benchmark, accelerator validation, or full test-suite result is claimed.
  See the later update below, [API](../docs/api/optimizers/pepo.md)
  and [design](../docs/development/notes/2026-11-17-pepo-dmt-design.md).

## Later update: exact-map DMT prototype

- Added `mode="dmt"` for dense NumPy complex128 open PEPOs with exact
  single-layer maps, all L shapes or explicit regions, guarded feasible
  low-rank fits, optional local objective refinement and transactional failure.
- A lossy 2x2 example preserves all four L-shaped densities while discarding
  a four-site correlation; independent dense partial traces verify the maps.
  The closest PEPO/full-update/public suites passed 152 tests, with eight
  accelerator tests skipped. Ruff and diff checks passed.
- Boundary-MPS maps, caching, larger lattices, Torch/CuPy and certified global
  optimality remain unimplemented. A failed nonconvex solve does not prove
  infeasibility. This is a prototype, not the requested scalable algorithm.

## Later update: boundary-MPS regional maps

- Added exact and frozen single-layer boundary-MPS maps; direct compression
  uses Quimb's MPS sweep and `fit_mode="dmrg"` uses Quimb 1D variational FIT.
  Open pair and region indices remain exposed through local identity tensors.
  Regional maps are reused within a pair solve, and duplicate regions are
  removed; numeric boundary caches are not reused across maps or gates.
- On a random 4x4 PEPO, `boundary_chi=1` produced about 0.61 relative map
  error for both methods; `chi=8` reproduced the exact map to numerical
  precision. Independent partial traces checked all 36 L shapes for an
  actually lossy parity-correlated density and for a nontrivial gate target.
- PEPO, full-update and public API suites: 158 passed, 8 accelerator cases
  skipped. Ruff and `git diff --check` passed. Exact output tests took under
  one second each on this machine; they are not a scaling benchmark.
- Maps and open-leg tensors are still explicitly materialized and capped;
  distant regions may leave the whole lattice uncompressed. No independent
  global accuracy certificate or large-system performance claim is made.
  DMT remains NumPy complex128 and open-boundary only.

## Later update: boundary and truncation review

- Changed the DMT direct boundary from Quimb's `mps` sweep to its 1D `direct`
  compressor, matching Pepsy's direct boundary fitting method; a regression
  spies on the actual selected Quimb method. DMRG retains Quimb variational FIT.
- Added analytic complex-factor Jacobians for feasibility and SLSQP refinement,
  finite-difference validation and nonfinite target/map rejection. The method
  remains nonconvex and has no global-optimum or feasibility certificate.
- On a fixed random 4x4 example, relative map errors for direct were 0.609,
  0.054 and ~8e-16 at boundary caps 1, 2 and 4; DMRG was comparable.
  On one small feasibility case analytic versus numerical differentiation
  used 23 versus 86 residual calls, but elapsed times (0.0055 versus 0.0045 s)
  did not demonstrate a speedup. These are small probes, not benchmarks.
- PEPO optimizer, PEPO joint updates, full updates and public API checks:
  161 passed and 8 accelerator cases skipped. Ruff and diff checks passed.

## Later update: map caching and 6x6 tests

- Added a bounded per-run LRU for whole regional maps. Maps are reused for
  repeated active pairs and invalidated when a changed tensor is outside their
  active pair; a new `run()` starts with an empty cache. No strip-level sharing
  across distinct regions or bonds is claimed. Added explicit `boundary_chi`
  `"auto"` heuristic (`max(4, 2 * chi)`) and cache controls/reporting.
- Cached/fresh comparisons cover direct and DMRG boundaries, repeat-pair hits,
  exterior invalidation, eviction and external mutation between runs.
- 6x6 direct/DMRG tests preserve all 100 L regions over two gates for chi 1
  and 4 and show 100 repeated-pair cache hits. An independently checked,
  genuinely lossy parity-density case compresses an initial bond 2 to 1 and
  preserves every L region. These structured cases are not scaling proofs.
- Random 6x6 bond-2 regional-map probe versus exact contraction: direct errors
  at boundary caps 1, 2, 4 and 8 were 1.013, 0.345, 0.0174 and 2.2e-15;
  DMRG gave 0.970, 0.370, 0.0174 and 1.6e-15. A finite cap is not an accuracy
  certificate and the heuristic is not necessarily sufficient.
- Focused PEPO/full-update/public-API suites passed 172 tests with 8 accelerator
  skips, using `OPENBLAS_NUM_THREADS=1`; Ruff and `git diff --check` passed.
  A structured 6x6 two-gate direct-boundary probe took 0.51 s with 100 map
  cache hits versus 0.88 s with cache disabled. These are single-run timings.

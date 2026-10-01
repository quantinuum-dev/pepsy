# 2026-09-30 — Integrate backend-aware automaton MPO delinearization

- Scope: implement default-on, two-pass rank reduction for dense MPO builds.
- Branch / baseline commit: `develop` / `a233a40`.
- Commit status: included in the MPO delinearization commit prepared on
  2026-10-01 after updating `develop` to `7773d2e`.

## What changed

- Added two-pass suffix/prefix rank reduction on NumPy, Torch, CuPy, and JAX
  dense arrays. Automaton arrays are reduced before Quimb materialization when
  no builder backend converter is present; converted backend MPOs are reduced
  before the final automaton SVD cap.
- Shared dense-array operations use Autoray dispatch (`ar.do`) for reshape,
  transpose, axis moves, SVD, column selection, QR, solves, and reconstruction;
  host reads use `ar.to_numpy`. NumPy uses pivoted QR. Torch, CuPy, and JAX use
  backend SVD to decide rank, then backend-native pivot selection, QR, and
  solve. This keeps factor values on-device and avoids differentiating through
  singular vectors. Only scalar rank, norm, and pivot decisions move to the
  host because they control data-dependent output shapes and Python loops.
- Enabled delinearization by default for dense `ham_tn.to_mpo` and
  `MPOAutomaton.to_mpo` calls. `delinearize=False` opts out. Symmray keeps its
  backend-specific compression path.
- Fixed the reverse sweep's row-rank check so it reduces a virtual dimension
  larger than the local physical configuration count.
- Updated API guides, the Unreleased changelog, and regression tests.

## Validation

- Activated `/Users/rezah/envs/genpy`; focused `test_ham.py`,
  `test_structural_compression.py`, and `test_mpo_automaton.py` suite:
  **118 passed, 1 skipped**. CuPy skipped because CuPy/CUDA is unavailable in
  this environment.
- Torch backend test confirms operator preservation and finite matching
  gradients. JAX tests confirm eager rank reduction and traced gradient flow;
  under tracing, dynamic rank reductions are skipped.
- Ruff and `git diff --check` passed.
- A seeded 8-site anisotropic long-range model reached every exact
  operator-Schmidt rank checked; its dense MPO differed from the raw build by
  relative error `5.0e-15`. With `max_bond=8`, the relative dense truncation
  error was unchanged at `0.1080`.

## Performance observations

Median wall times are local CPU measurements. The 8-site row uses the same
`max_bond=8` cap in both routes; the other original rows disable truncation.

| Workload | Default automaton | Automaton + delinearize | Bond result |
| --- | ---: | ---: | --- |
| 6×6 square Heisenberg + field | 0.106 s | 0.087 s | no change; max 20 |
| 8-site random long-range spin model, cap 8 | 0.016 s | 0.010 s | same capped bonds and relative error `0.1080` |
| 32-site random long-range spin model, 120 terms | 0.546 s | 0.454 s | max 69→42; sum 1239→749 |

The NumPy QR optimization reduced a paired five-run 32-site build from `0.462 s`
to `0.377 s` (18% faster), with final max bond 42 and summed bonds 763 in both.
On a separate 6×6 nearest-neighbor case with cap 16, the default term strategy
took 4.50 s without delinearization and 4.59 s with it; the automaton strategy
took 0.039 s and 0.031 s, respectively. The large timing difference comes from
the build strategy, while delinearization itself adds little cost to this term
route.

## Limits

- The 2-pass rank reveal is lossless within its conservative residual guard; it
  is not a global optimality guarantee for every MPO.
- Rank decisions are data-dependent host-side shape decisions. JAX tracing
  skips rank reduction to preserve static shapes and gradient flow.
- CuPy support is covered by a conditional test but could not be executed here.

## 2026-10-01 — publication checks

- Pulled five remote commits to `7773d2e` and restored the local MPO changes.
  Git combined both sets of changelog entries; the other local files were
  restored unchanged. The pre-pull stash remains as a backup.
- Re-ran the three focused suites with `MPLBACKEND=Agg`: **118 passed,
  1 skipped** (CuPy is not installed). The first attempt without `Agg` aborted
  in Matplotlib's macOS GUI backend during a schematic drawing test.
- Ruff and `git diff --check` passed. The full suite was not run.
- User authorized committing this work and pushing `develop`.

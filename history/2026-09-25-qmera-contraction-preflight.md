# 2026-09-25 — qMERA contraction preflight

- Scope: estimate qMERA contraction FLOPs and peak memory before optimization,
  cache Cotengra paths, and clarify the notebook/API.
- Branch / baseline commit: `develop` / `4f62cb0`.
- Commit status: uncommitted and unstaged. Earlier qMERA 2D/Torch and unrelated
  Tree working-tree edits remain present; the sibling example `qmera/` folder
  is untracked.

## What changed

- Added `QMeraBuilder.estimate_contraction_cost` and the owning qMERA report
  API. It estimates forward complex FLOPs and peak live tensor bytes per cone,
  plans only the paths used by the selected normalization mode, and memoizes
  repeated topology estimates.
- Primed unsliced paths in `QMeraContractionPathCache` for compiled loss reuse.
  Cotengra remains responsible for sliced-tree caching.
- The TFIM notebook prints the estimate before optimization and passes the
  returned cache to Pepsy's optimizer. Its source outputs were preserved.
- Updated the API page, changelog, and
  `docs/development/notes/qmera_contraction_preflight_2026_09.md` with metric
  scope and limitations.

## Validation

- `tests/test_optimize_qmera.py`: 99 passed. One loky worker warning occurred
  in the Torch full-graph test.
- Public API/layout: 57 passed, one known installed-version assertion
  deselected (`0.4.1` distribution metadata vs `0.5.0` checkout).
- Ruff over `src tests` and `git diff --check`: passed.
- Final four-site TFIM notebook executed into `/tmp/pepsy_examples_executed/`:
  preflight printed before 25 JAX Adam steps; final energy `-5.1873339645`,
  exact reference `-5.2262518595`, local-term sum agreed, and Torch AOT
  energy/gradient check passed. No source notebook outputs changed.
- Odd retained 2D `(1, 5)` one-term preflight and compiled energy matched the
  direct local energy with `contraction_opt="greedy"`.

## Limits / next measurements

- FLOPs and peak bytes are Cotengra path estimates for forward dense local
  contractions, not measured wall time or process/GPU memory. Native Symmray
  block costs require their own estimator and currently raise an explicit
  `NotImplementedError`.
- A representative larger 2D benchmark could compare these estimates with
  actual runtime and memory before using them as resource predictions.

# 2026-06-22 - Optimizer layout and Quimb MPS boundaries

- Milestone: optimizer package cleanup + Quimb MPS sweep boundaries
- Branch / commit: `develop` (committed in this snapshot)

## What changed
- Reworked `src/pepsy/optimizers/` into package namespaces:
  `mps/`, `mpo/`, `peps/`, `sweep/`, plus `_shared.py` and existing
  `global_opt.py`.
- Kept public optimizer imports working through `pepsy.optimizers` while tests
  and docs now target implementation leaves such as
  `pepsy.optimizers.peps.optimizer`.
- Added `README.md` / `PLAN.md` notes under `src/pepsy/optimizers/`,
  `src/pepsy/optimizers/peps/`, and `src/pepsy/optimizers/sweep/`.
- Added overview READMEs for `src/pepsy/boundary/` and `src/pepsy/tensors/`.
- Added `QuimbMpsBoundaryStore` and boundary-engine selection helpers in
  `src/pepsy/optimizers/sweep/environments.py`.
- Added `SweepOptimizer(boundary_engine={"auto", "dmrg", "quimb-mps"})`.
  `auto` keeps dense inputs on Pepsy `BdyMPS`/`CompBdy` and routes
  Symmray-looking inputs to Quimb MPS environments.
- Added `PepsOptimizer(boundary_engine=..., boundary_options=...)`, forwarding
  the resolved boundary policy into sweep cleanup and keeping normalization /
  infidelity metric defaults consistent with the selected engine.

## Why
- `optimizers/` had grown into several large flat modules (`mps.py`, `mpo.py`,
  `peps.py`, `sweep.py`). The new layout makes the API and future extraction
  points clearer.
- Symmray-backed PEPS needs a boundary path that does not depend on Pepsy's
  DMRG/FIT boundary updates. Quimb already provides MPS boundary contraction
  and environment routines, so `SweepOptimizer` can reuse those through a
  provider-style adapter.
- `PepsOptimizer` is the outer PEPS gate-stream driver, so it needs to expose
  the same boundary policy cleanly instead of hiding Quimb MPS routing inside
  ad hoc `sweep_kwargs`.

## How it was validated
- `pytest -q tests/test_optimize_peps.py` -> 20 passed.
- `pytest -q tests/test_optimize_peps.py tests/test_optimize_sweep_plot.py tests/test_public_api.py tests/test_package_layout.py` -> 467 passed.
- `pytest -q tests/test_optimize_mps.py tests/test_optimize_mpo.py tests/test_optimize_peps.py tests/test_optimize_sweep_plot.py tests/test_optimize_global.py tests/test_gradient_solver.py tests/test_public_api.py tests/test_package_layout.py` -> 596 passed.
- `git diff --check -- src/pepsy/optimizers docs/api/optimizers/peps.md tests/test_optimize_peps.py tests/test_optimize_sweep_plot.py docs/api/optimizers/sweep.md docs/api/optimizers/mps.md docs/api/optimizers/mpo.md docs/development/package_layout.md tests/test_optimize_mps.py tests/test_optimize_mpo.py tests/test_gradient_solver.py` -> passed.
- Trailing-whitespace scan over new README/PLAN files -> passed.
- Tiny manual Quimb smoke built a 2x2 random PEPS with
  `SweepOptimizer(..., boundary_engine="quimb-mps")` and refreshed y-axis
  environments successfully.

## Decisions / findings
- The canonical package is still `pepsy`; `pepsy_examples` remains examples
  and external testing, and `tc_gauge` is a downstream time-compression
  consumer.
- Use conventional `README.md` / `PLAN.md` names, not `READ.me` / `PLAN.me`.
- Keep dense defaults on Pepsy `BdyMPS`/`CompBdy`; use Quimb MPS automatically
  only for Symmray-looking inputs unless the user explicitly requests
  `boundary_engine="quimb-mps"`.
- Explicit user boundary choices win: `boundary_engine="dmrg"` is preserved
  even for Symmray-looking PEPS.
- Current Quimb MPS environment refreshes are conservative full-axis refreshes
  after local updates. Correctness first; optimize later.

## Next step (do this first next time)
- Continue the optimizer split by moving gate routing, warm-start building,
  sweep/global dispatch, and trace helpers out of the large `peps/optimizer.py`
  and `sweep/optimizer.py` files into the placeholder modules already created.

## Open questions / blockers
- Add deeper numerical comparisons between dense Pepsy and Quimb MPS boundary
  paths.
- Add optional Symmray regression coverage guarded by `pytest.importorskip`.
- Run downstream smoke checks in `tc_gauge` after the Pepsy API settles.
- Quimb environment exponent handling still needs careful stripped and
  non-stripped tests.

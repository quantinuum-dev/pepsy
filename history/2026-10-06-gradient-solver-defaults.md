# 2026-10-06 — Gradient solver defaults and native backend validation

- Scope: simplify the sibling Gaugy joint notebook's solver cell and move
  necessary backend checks into Pepsy.
- Branch / baseline commit: `develop` / `8f7c896`.
- Commit status: working-tree edits only; no staging, commit or publication.
  Concurrent PEPS optimizer changes and existing history files are separate.

## Changes and findings

- The notebook now constructs `GradientOptimizer(solver=SOLVER,
  n_steps=MAXITER, log_every=1, progress=True)` without a solver-specific
  option ladder. Pepsy already has solver dispatch and numerical defaults;
  those defaults were preserved rather than adopting the notebook's strict
  tolerances globally. The notebook README and solver API explain optional
  overrides and potentially different stopping points.
- Native Torch/JAX solvers reject the other backend's arrays before conversion,
  optional optimizer imports or objective execution. Mixed backend mappings
  are rejected consistently. SciPy/NLopt retain backend inference, and
  NumPy/scalar inputs retain their existing conversion paths.
- A bounded default JAX Adam notebook check exposed an existing bookkeeping
  bug: a pre-update loss was paired with post-update parameters. Restoring
  the reported best could therefore increase the actual returned cost.
  The solver now saves the evaluated state and evaluates its last update
  once before selecting the final state. `final_loss` matches returned
  parameters with or without best-state restoration. The extra forward
  evaluation is included in `n_evals`; nonfinite final values cannot replace
  a finite best state.
- Defaults are numerical policies, not objective-specific automatic tuning.
  A five-step default Adam run need not improve a near-solution start.

## Validation

Shared Python 3.12, CPU, one BLAS/OpenMP thread. Installed versions:
Torch 2.6.0+cu124, JAX 0.10.2, SciPy 1.17.1, NLopt 2.11.0, Optax 0.2.8.

- Before fixes: seven backend-validation regressions and four overshoot
  result-consistency regressions failed; two existing mixed-host guards passed.
- `python -m pytest -q -o addopts='' tests/test_gradient_solver.py
  tests/test_gradient_solver_jax.py`: **78 passed in 11.23 seconds**.
  Includes native backend rejection, default host solves, explicit overrides,
  last-update selection and nonfinite final-update recovery.
- Downstream `test_joint_notebook.py`: **28 passed in 116.42 seconds**,
  with 12 existing NumPy/Torch conversion deprecation warnings. This covers
  SciPy/NLopt, both backends/objectives and all three expansions using defaults.
- Default Torch L-BFGS notebook optimizer/comparison cells passed on three
  open sites, two slices, cutoff two and five steps. The subsequent JAX Adam
  check exposed the bug above.
- After correction, all **27 notebook code cells** executed with default
  JAX Adam on CPU, three open sites, two slices, cutoff two and 30 steps.
  Local cost decreased from `9.68779434e-05` to `3.75854137e-05`; best loss
  matches returned parameters and final comparison values are finite.
- `python -m ruff check src tests` and `git diff --check`: passed.
- No fresh CUDA validation or full-suite claim. Notebook source experiment
  settings were preserved; temporary smoke output is under `/tmp`.

See the [solver API](../docs/api/solvers/gradient.md) for defaults and result
semantics. Previous JAX host-solver evidence remains in the
[2026-10-01 record](2026-10-01-jax-host-solvers.md).

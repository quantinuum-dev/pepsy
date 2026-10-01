# 2026-10-01 — Fix joint gauge notebook review findings

- Scope: user authorized rechecking and fixing the two notebook review findings.
- Branch / baseline: Pepsy `develop` / `5eadd9f`; Gaugy Examples `main` /
  `b204aab`, with substantial pre-existing working-tree changes preserved.
- Commit status: all changes remain uncommitted and unstaged; nothing published.

## Implemented

- [GradientOptimizer's SciPy adapter](../src/pepsy/solvers/gradient.py) returns
  SciPy's termination message instead of its initialized `maxiter` label.
  Pepsy's explicit `patience` and `bad_max` callback reasons retain precedence.
  Parameter selection, objective values, and optimization behavior are unchanged.
- Updated the [solver API guide](../docs/api/solvers/gradient.md) and changelog.
- The [joint gauge notebook](../../gaugy_examples/pauli_gaugy/gauge_optimization/joint_gauge_optimization.ipynb)
  compares compiled autodiff gradients against central differences of
  `compiled.loss_vector` on the selected backend/device at the optimized
  parameters. It checks V-only and G-only directions at two step sizes, saves
  the diagnostic, and raises on disagreement. Updated its adjacent README.
- Added [solver regressions](../tests/test_gradient_solver.py) and
  [notebook regressions](../../gaugy_examples/pauli_gaugy/gauge_optimization/test_joint_notebook.py).
  The latter execute the actual diagnostic cell, forbid the native checker,
  and verify detection of incorrect V and G compiled gradients separately.

## Fresh validation

- Environment: existing Python 3.12 environment, SciPy 1.17.1; CUDA hidden and
  JAX on CPU. SciPy's installed `minimize` signature and official
  [OptimizeResult documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html)
  confirm that `message` describes termination and statuses vary by method.
- Before the solver fix, all five new real-SciPy termination cases failed
  (L-BFGS-B/BFGS convergence, iteration/evaluation limits, and line-search
  failure); both existing-control preservation cases passed.
- `python -m pytest -q -o addopts='' tests/test_gradient_solver.py tests/test_public_api.py tests/test_package_layout.py`:
  **90 passed**, two compatibility deprecation warnings.
- Examples' `test_joint_notebook.py` and `test_helper.py`: **13 passed**,
  including Torch/JAX positive checks and injected incorrect compiled gradients.
- Gaugy `test_paulig_compiled.py` and `test_cluster_optimization_api.py` with
  `-k 'not inductor'`: **35 passed**, two Inductor checks deselected.
- Executed every notebook code cell with temporary settings `L=4`, open
  boundaries, time 0.3, depth 2, cutoff 2, CPU, three optimizer steps, for
  Torch/SciPy and JAX/Adam. Both improved the objective, completed exact
  validation and compiled gradient checks, and matched native reference
  gradients. Results stayed under `/tmp`; stored notebook outputs and existing
  run records were preserved. The scalar quadratic reproduction now reports
  SciPy convergence after one iteration instead of `maxiter`.
- Pepsy `python -m ruff check src tests`, Ruff on the new notebook test,
  notebook format/code validation, and both repositories' `git diff --check`:
  passed.

## Limits

- The full-size L=12 CUDA run, Inductor, and full repository suites were not run.
- Native `plan.check_gradients` remains a valid check of its native evaluator;
  no Gaugy package algorithm needed a change. The finite-difference SciPy
  solver is outside these two findings and was not changed.
- Historical run stop labels cannot be reconstructed from the old saved
  records and were not rewritten.

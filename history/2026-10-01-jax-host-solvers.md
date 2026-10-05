# 2026-10-01 — Joint notebook JAX/Torch L-BFGS support

- Scope: make the Gaugy joint notebook's public Pepsy optimizer work with
  JAX and Torch for L-BFGS and NLopt LD_LBFGS; test actual optimization and
  direct-circuit validation.
- Branch / baseline: `develop`, `608b664`.
- Commit status: working-tree edits only; no staging, commit, or publication.
  Existing sampling/MPI/trajectory edits are separate work and preserved.

## Changes

- SciPy/NLopt runners use backend-specific callbacks, preserving existing
  Torch controls and adding JAX real/complex autodiff, dtype/device/shape,
  optional-dependency isolation, and second-order SciPy callbacks.
- Renamed the shorthand table that was overwritten by a second SciPy method
  table. Bare `lbfgs` now selects SciPy L-BFGS-B as documented; explicit
  `torch-lbfgs` selects Torch. Fixed both result-return APIs when Torch is absent.
- Updated solver API/annotations and changelog. The sibling notebook exposes
  `lbfgs`, `scipy-lbfgs` and `LD_LBFGS` for either backend, handles signed costs,
  preserves its initial Torch vector, and disables JAX GPU preallocation
  before initialization. Its final local/direct/exact comparison remains
  evaluation only, with no gradient-check/archive/result-writing cells.

## Validation

- `tests/test_gradient_solver.py tests/test_gradient_solver_jax.py
  tests/test_public_api.py tests/test_package_layout.py`: **116 passed** on
  CPU, with two existing compatibility-alias deprecation warnings.
  Covers convergence to known minima, complex coordinates, signed costs,
  bounds, both public result APIs without Torch, and Hessian/HVP methods.
- Sibling `test_joint_notebook.py`: **28 passed** on CPU. The new 24-case
  matrix uses both backends, both requested solvers, local/global objectives,
  and connected-log/polymer/exact expansions, executing the notebook solver
  and final validation cells. Four existing independent trace comparisons pass.
- **Eight fresh CUDA Run All notebooks passed**, using L=4, open boundaries,
  time=.3, depth=2, cutoff=2 and budget=5. Both backends/solvers/objectives
  decrease training costs and run the exact dense reference; no CPU fallback.
  Corresponding Torch/JAX direct costs agree within 3e-12 in this matrix.
- `python -m ruff check src tests`, example test lint, notebook-format validation,
  and `git diff --check` passed. The full repository suite was not run.
- **Two additional JAX CUDA Run All notebooks passed** at the saved settings
  L=12, periodic boundaries, time=2, depth=10 and cutoff=3, using budget=2
  for each solver. Both decrease training and exact dense direct costs.
- A fresh process with `sys.modules['torch']=None` imported the public solver
  API and optimized a JAX quadratic with both `lbfgs` and `LD_LBFGS` to zero.

## Evidence and limits

The saved notebook error was a `KeyboardInterrupt` inside JAX Adam. A small
CPU JAX Adam Run All already passed during reproduction. The old notebook
explicitly rejected JAX L-BFGS, while Pepsy's generic callbacks required Torch;
these restrictions are now removed for SciPy/NLopt.
The CUDA matrix is a bounded compatibility check, not a claim that 1000-step
experiments converge. SciPy budgets iterations; NLopt budgets evaluations.
GPU preallocation settings require a fresh kernel if JAX already initialized.
Temporary execution copies/logs are under `/tmp`, outside the source notebook.

Installed versions, upstream checks and packing rationale are in the
[maintenance note](../docs/development/notes/2026-10-01-jax-host-solvers.md).

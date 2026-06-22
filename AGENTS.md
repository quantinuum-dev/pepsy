# AGENTS.md

This file defines default behavior for coding agents working in this repository.

## Scope

- Applies to the entire `pepsy` repository unless overridden by nested agent instruction files.
- Prefer minimal, targeted changes that preserve existing APIs and style.

## Project Context

- Main package code: `src/pepsy/`
- Tests: `tests/`
- Docs: `docs/`
- Examples: `examples/`
- The sibling `/home/reza.haghshenas@quantinuum.com/tensy` repository uses
  Pepsy as its tensor-network package; keep that consumer in mind for API and
  behavior changes.
- Local/generated artifacts may appear in `build/`, `docs/_build/`, `__pycache__/`, `cash*/`, `ctg_cash/`, and `store/`; do not use these as source-of-truth code.

## Startup Checklist

At the start of a new task:

- Check `git status --short` and treat existing changes as user-owned unless you made them.
- Look for nested instructions with `rg --files -g 'AGENTS.md'`.
- Orient from source and tests, not cache/build output: `find src/pepsy -maxdepth 2 -type f -name '*.py' | sort`.
- Read the closest tests before editing. For public API or import-path work, read `tests/test_public_api.py`, `tests/test_package_layout.py`, and `docs/development/package_layout.md`.
- Prefer new package namespaces for imports. Do not add old flat modules such as `pepsy.core`, `pepsy.gates`, `pepsy.sampler`, or `pepsy.optimize_sweep`.

## Architecture Map

- `pepsy.backends`: backend inference/conversion helpers, package-wide backend defaults, torch/JAX/CuPy linalg registration.
- `pepsy.tensors`: `OneDMap`, tag validation, product/identity constructors, contraction optimizers, observables, `tn_norm`, and `tn_fidelity`. Many leaf files here are facades over `tensors/core.py`.
- `pepsy.operators`: gate matrices, `gate`/`gate_simple` dispatch, MPO/PEPO builders, and `ham_tn` Hamiltonian helpers.
- `pepsy.boundary`: PEPS norm/overlap setup (`build_bra_ket`), boundary environments (`BdyMPS`), sweeps (`CompBdy`), `contract_boundary`, `normalize`, and `infidelity`.
- `pepsy.solvers`: `GradientOptimizer`, `FDSolver`, `optimize_packed_params`, finite-difference adapters, and canonical solver-name handling.
- `pepsy.optimizers`: higher-level `GlobalOptimizer`, `SweepOptimizer`, `MpsOptimizer`, `MpoOptimizer`, and `PepsOptimizer` workflows.
- `pepsy.sampling`: `MpsSampler`, `VecSampler`, `PepsBpSampler`, and result dataclasses.
- `pepsy._internal`: private formatting and small utility helpers only.

## Core Workflows

- Boundary contraction flow: `build_bra_ket(ket, bra?) -> BdyMPS(...) -> contract_boundary(...)`.
- `build_bra_ket` tags the ket in place, adds `KET`/`BRA` tags, validates `X*`, `Y*`, and `I*` tags, and reindexes colliding bra internal indices with an `_*` suffix.
- `contract_boundary` returns `BoundaryContractResult`; use `res.cost` and `res.fidel`, not tuple unpacking.
- `BdyMPS.mps_b` stores keys like `Y0_l`, `Y0_r`, `X0_l`, `X0_r`. `BdyMPS.chi` reports the largest current boundary bond, and `expand_bnd(chi, inplace=True)` retunes existing boundaries.
- `normalize(...)` mutates the input state in place and requires `chi` unless a boundary object or `{"bdy": ...}` holder is supplied.
- `infidelity(...)` returns a dict with `infidelity`, `norm`, `norm_target`, `overlap`, and reused/created boundary handles.
- PEPS-like networks should carry lattice tags `X{i}`, `Y{j}`, `I...`; physical outer indices conventionally use `k...` for ket legs and `b...` for bra/operator-output legs.
- Gate streams should use canonical bundled entries like `[(gate, where), ...]`. The ambiguous single bundled alias `(gate, where)` is intentionally rejected.
- User-provided gate tensors are not backend-coerced automatically; keep gate tensors and tensor-network arrays on compatible backends.
- `OneDMap` supports `snake`, `snake-row-major`, `row-major`, `col-major`, `hilbert`, `hilbert-row-major`, and `diag` modes. PEPO conversion is restricted; check `tests/test_ham.py` before changing mapper behavior.

## Public API Rules

- Top-level `pepsy` exports are lazy and managed in `src/pepsy/__init__.py` via `_SYMBOL_MODULES`, `_MODULE_EXPORTS`, and `__all__`.
- If adding/removing a public symbol, update the owning subpackage `__all__`, top-level `src/pepsy/__init__.py` when appropriate, docs under `docs/api/`, and `tests/test_public_api.py`.
- Old flat import modules are intentionally removed and covered by `tests/test_package_layout.py`.
- Preserve convenience top-level symbols such as `pepsy.BdyMPS`, `pepsy.rx`, and `pepsy.SweepOptimizer` unless the task explicitly requests an API break.

## Numerical and Backend Notes

- Default contraction helper is usually `build_optimizer(progbar=False)` or string/object `contraction_opt="auto-hq"`.
- `build_optimizer` and `build_compressed_optimizer` intentionally avoid seed kwargs; tests assert this.
- Canonical solvers include `torch-adam`, `torch-lbfgs`, `torch-adamw`, `torch-radam`, `torch-nadam`, `scipy`, `nlopt`, `fd-adam`, `fd-scipy`, `fd-nlopt`, `jax-adam`, `jax-adamw`, `jax-sgd`, and `jax-rmsprop`.
- `SweepOptimizer` expects canonical solver names; do not reintroduce legacy aliases like `scipy_lbfgs` or `nlopt_lbfgs`.
- External solvers such as SciPy/NLopt flatten params to CPU NumPy `float64` internally and then cast back to original dtype/device.
- Keep optional dependencies optional. Use `pytest.importorskip(...)` style in tests when behavior depends on torch, scipy, nlopt, jax, cupy, or optax.

## Editing Rules

- Do not refactor unrelated code.
- Keep public interfaces stable unless the task explicitly requires a breaking change.
- Add brief comments only where logic is non-obvious.
- Preserve naming patterns and module structure used nearby.

## Python and Environment

- Use Python 3.11 environment for commands and tests.
- Run commands from repository root when possible.
- Install local development dependencies with `python -m pip install -e '.[dev,docs]'` when needed.

## Common Commands

- Run all tests: `pytest -q`
- Run a focused test file: `pytest -q tests/test_name.py`
- Run a focused test case: `pytest -q tests/test_name.py::test_case_name`
- Public API/layout smoke tests: `pytest -q tests/test_public_api.py tests/test_package_layout.py`
- Check Python syntax/static issues: `python -m pyflakes src tests`
- Build documentation: `sphinx-build -W -b html docs docs/_build/html`

If numba, matplotlib, or Python cache directories cause local environment noise, prefer temporary cache locations such as:

```sh
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mplconfig PYTHONPYCACHEPREFIX=/tmp pytest -q
```

## Code Style

- Prefer existing PEPSY helper functions and local patterns over introducing new utilities.
- Keep tensor-network naming, boundary naming, and optimizer argument names consistent with nearby code.
- Avoid changing numerical tolerances, default bond dimensions, or solver choices unless the task is specifically about accuracy, convergence, or performance.
- Keep compatibility with the supported package range in `pyproject.toml`.

## Examples and Notebooks

- Keep examples lightweight and deterministic where practical.
- Do not modify generated notebook outputs unless explicitly requested.
- When changing public examples, verify that imports use current namespaces and public API names.
- If a notebook or helper still shows an old flat import path, migrate it deliberately instead of copying that pattern into new code.

## Validation

- For code changes, run focused tests first (closest test files), then broader tests if needed.
- If behavior or API changes, update documentation in `docs/` and/or `README.md`.
- If unable to run tests locally, state this explicitly in the final report.

Focused validation guide:

- API/layout changes: `pytest -q tests/test_public_api.py tests/test_package_layout.py`
- Boundary setup, normalization, infidelity, or sweeps: `pytest -q tests/test_prepare_boundary_inputs.py`
- Tensor constructors, backend defaults, contraction helpers, or observables: `pytest -q tests/test_core_seed.py`
- Gate routing/builders: `pytest -q tests/test_gate.py`
- Hamiltonian or lattice mapping: `pytest -q tests/test_ham.py`
- Solver changes: `pytest -q tests/test_gradient_solver.py`
- Optimizers: `pytest -q tests/test_optimize_global.py tests/test_optimize_sweep_plot.py tests/test_optimize_mps.py tests/test_optimize_mpo.py`
- Sampling: `pytest -q tests/test_sampler.py`
- Docs/API behavior changes: run focused tests plus `sphinx-build -W -b html docs docs/_build/html`

## Safety and Boundaries

- Never delete user data or caches unless explicitly requested.
- Avoid destructive git operations.
- Do not modify generated/build artifacts unless the task explicitly asks for it.

## Pull Request / Change Summary Expectations

When finishing a task, include:

- What changed
- Why it changed
- How it was validated (tests/commands)
- Any risks or follow-up suggestions

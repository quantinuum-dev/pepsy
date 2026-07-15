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
- Gaugy and Tensy are separate packages/repositories. Do not recreate Gaugy,
  Tensy, or other sibling-project folders inside this Pepsy repository.
- Keep Pepsy source, tests, docs, and examples focused on Pepsy itself. Do not
  add `gaugy` or `tensy` imports, package code, notebooks, examples, or docs
  here unless the user explicitly asks for a cross-package integration artifact.
- When work needs Gaugy or Tensy behavior, make the Pepsy side expose a clean
  public API and leave package-specific code in the corresponding external
  repository.
- Local/generated artifacts may appear in `build/`, `docs/_build/`, `__pycache__/`, `cash*/`, `ctg_cash/`, and `store/`; do not use these as source-of-truth code.

## Startup Checklist

At the start of a new task:

- Check `git status --short` and treat existing changes as user-owned unless you made them.
- Look for nested instructions with `rg --files -g 'AGENTS.md'`.
- Orient from source and tests, not cache/build output: `find src/pepsy -maxdepth 2 -type f -name '*.py' | sort`.
- Read the closest tests before editing. For public API or import-path work, read `tests/test_public_api.py`, `tests/test_package_layout.py`, and `docs/development/package_layout.md`.
- Prefer new package namespaces for imports. Do not add old flat modules such as `pepsy.core`, `pepsy.gates`, `pepsy.sampler`, or `pepsy.optimize_sweep`.

## Branching & Workflow

- After a small coherent batch of package changes, run the relevant focused
  validation, commit only the files you changed, and push to the configured
  upstream. Do this incrementally instead of waiting for many unrelated edits
  to accumulate.
- Never include unrelated or user-owned work in these automatic commits, and
  do not push changes that fail validation unless the user explicitly asks.

## Architecture Map

- `pepsy.backends`: backend inference/conversion helpers, package-wide backend defaults, torch/JAX/CuPy linalg registration.
- `pepsy.tensors`: `OneDMap`, tag validation, product/identity constructors, contraction optimizers, observables, `tn_norm`, and `tn_fidelity`. Many leaf files here are facades over `tensors/core.py`.
- `pepsy.operators`: gate matrices, `gate`/`gate_simple` dispatch, MPO/PEPO builders, and `ham_tn` Hamiltonian helpers.
- `pepsy.boundary`: PEPS norm/overlap setup (`build_bra_ket`), boundary environments (`BdyMPS`), sweeps (`CompBdy`), `contract_boundary`, `normalize`, and `infidelity`.
- `pepsy.solvers`: `GradientOptimizer`, `FDSolver`, `optimize_packed_params`, finite-difference adapters, and canonical solver-name handling.
- `pepsy.optimizers`: higher-level `GlobalOptimizer`, `SweepOptimizer`, `MpsOptimizer`, `MpoOptimizer`, and `PepsOptimizer` workflows, plus the `stabilizer_tn` subpackage (`MpsStabOptimizer`, `STNState`: stim-tableau + coefficient-MPS STN simulator with basis-updating measurement, reset, and magic-state injection).
- `pepsy.sampling`: `MpsSampler`, `VecSampler`, `PepsBpSampler`, and result dataclasses.
- `pepsy._internal`: private formatting and small utility helpers only.

### BP and simple-update gauge workflow

- The experimental BP integration is isolated in `src/pepsy/bp/`. The supported
  top-level workflows are `pepsy.one_norm_bp`, `pepsy.gauge_all`, and
  `pepsy.gauge_all_simple`; keep its remaining symbols under `pepsy.bp` until
  they are individually promoted.
- `gauge_all_simple` is the single simple-update gauge path. Its optional
  `RelayGaugeOptions` enables convergence acceleration. It must preserve the
  represented tensor network exactly when it mixes an external bond gauge:
  compensate the two adjacent core tensors before returning `(core, gauges)`.
  Relay memory and DIIS candidates are projected back to nonnegative,
  L2-normalized singular-value gauges.
- `gauge_all` is the SU <-> D1BP orchestrator. It may share conversions,
  schedules, and relay controls with SU and BP, but must not collapse their
  distinct numerical update loops into one implementation.
- `one_norm_bp` is the primary plain 1-norm BP runner. It supports L1BP,
  HV1BP, and D1BP; use `method="d1bp"` for the SU-compatible directed-message
  representation.
- Relay/SU state, DIIS history, and warm starts are keyed by the stable external
  bond ids, so this path intentionally rejects `fuse_multibonds=True`. Its
  `schedule="parallel"` mode schedules edge-coloured, non-overlapping bond
  batches on threads for CPU NumPy networks; retain the serial route for other
  backends.
- For D1BP warm starts from simple update, use
  `simple_update_core_and_gauges_from_messages` and
  `run_d1bp_from_simple_update_gauges`; validate the post-update residual rather
  than assuming a fixed sweep count means convergence.

### MPS optimizer workflow

- Read `.github/skills/mps-optimizer/SKILL.md` before changing
  `pepsy.optimizers.mps.optimizer`.
- For repeated layout-aware evolution, install a layout once with
  `MpsOptimizer.apply_layout(...)`. It stores `logical_order` as
  position-to-logical-site labels and intentionally never swaps the MPS back.
  Use `logical_site`, `position`, `remap_sample`, and `to_dense` for readout.
- A product-state reorder is free exactly when `p.max_bond() == 1`. An
  initially entangled state must raise by default; `allow_lossy_reorder=True`
  permits one caller-controlled-cutoff reorder only.
- Treat `info_c["cur_orthog"]` as algorithm state. Pass it through Quimb gate,
  canonicalization, control-event, and normalization paths. Temporary target
  MPS copies must use isolated metadata and must not overwrite the live cache.
  Prefer a tracked one-site canonical norm over a full doubled-network norm.
- `mode="exact"` is a dense TensorNetwork path and does not use canonical
  metadata. Leaving exact mode must rebuild an MPS before any MPS operation;
  persistent layouts cannot be switched into exact mode silently.
- Control events that change MPS length, especially `cap`, are incompatible
  with persistent layouts. Measurement/reset bookkeeping remains in logical
  site labels.

### Stabilizer tensor-network workflow

- Read `.github/skills/stabilizer-tensor-networks/SKILL.md` and its method/API
  references before changing `MpsStabOptimizer` or `STNState`.
- Preserve `|psi> = C|p>`: Clifford gates update the Stim tableau `C`; physical
  non-Clifford operators are frame-mapped through `C`; `submpo` acts directly
  in the coefficient frame.
- Preserve the coefficient-MPS canonical-centre tracker through Quimb updates.
  Use its one-site tensor norm, including the MPS exponent, for local norms and
  unitary truncation diagnostics.
- `track_infidelity` records sparse cumulative `1 - ||p||^2` samples only for
  normalized unitary segments. It is not exact overlap fidelity or discarded
  SVD weight. Do not renormalize unitary evolution or sum `infidelities`.
  Non-unitary matrices and coefficient-frame sub-MPOs emit no sample and
  invalidate the proxy until a normalized projective collapse resets it.
- `norm_diagnostics()` combines completed/current segment survival factors:
  `total_survival_proxy` is their product and `total_norm_proxy` is its square
  root. `geometric_mean_norm` is only the per-segment geometric mean. Never
  multiply physical Born probabilities into these compression proxies; progress
  reports `Ntotal` and `Itotal` for the total norm and infidelity proxies.
- A selected Kraus `TrajectoryEvent` outcome is a normalized trajectory
  boundary. Snapshot the preceding unitary segment before applying its
  non-unitary matrix, then normalize, reset the proxy, and commit the boundary
  record (`kind="trajectory_kraus"`). This is how later unitary evolution
  resumes valid STN norm tracking.
- Keep dense Pauli decomposition bounded by
  `max_pauli_decomposition_qubits`; prefer named gates, Pauli rotations, or a
  coefficient-frame sub-MPO over opting into uncontrolled `4**k` enumeration.

## Upstream Tensor-Network Substrate

- Treat `autoray`, `cotengra`, `cotengrust`, and `quimb` as first-class
  dependencies and prefer their APIs over local reimplementations wherever they
  cover the needed tensor-network, contraction-planning, and backend-dispatch
  behavior.
- Use `quimb` tensor-network objects and methods for Tensor/TensorNetwork,
  MPS/MPO/PEPS/PEPO construction, contraction, boundary, gate, geometry,
  sampling, and optimization workflows when practical. Keep Pepsy wrappers
  thin and focused on Pepsy conventions, compatibility, and stable public APIs.
- Use `cotengra` for contraction path/tree optimization, hyper-optimization,
  slicing, subtree reconfiguration, compressed contraction planning, and
  reusable optimizer objects. Forward cotengra-compatible options rather than
  inventing parallel path-search configuration inside Pepsy.
- Use `cotengrust` through cotengra's public acceleration paths for Rust-backed
  greedy, random-greedy, optimal, and subtree-reconfiguration path-finding.
  Keep cotengra as the main optimizer interface unless a task specifically
  needs a lower-level cotengrust primitive.
- Use `autoray` for backend inference and backend-agnostic array operations,
  including creation, conversion, `einsum`/`tensordot`, reshaping, conjugation,
  linalg, and registered backend-specific gradient/linalg fixes. Preserve
  compatibility with NumPy, Torch, JAX, CuPy, and other autoray-compatible
  arrays when optional dependencies are installed.
- Preserve compatibility with the supported dependency ranges in
  `pyproject.toml`; handle version-specific upstream behavior with feature
  detection and focused regression tests.
- Do not vendor or copy upstream internals. If Pepsy needs a workaround for an
  upstream behavior, isolate it behind a small adapter, document why it exists,
  and test it against the closest public upstream API.
- `symmray` is the planned symmetric/block-sparse integration path. Design new
  tensor, gate, boundary, and optimizer code so Symmray-style arrays can flow
  through quimb/autoray paths where possible, keep `symmray` optional unless it
  becomes a declared dependency, and use `pytest.importorskip("symmray")` for
  Symmray-specific tests.

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

- Use the Python 3.12 environment for commands and tests:
  `source ~/envs/py312/bin/activate`.
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
- Write examples compositionally: expose small setup/build/evolve/measure steps
  that compose public APIs instead of copying internal implementation details.
- For Pepsy examples, prefer Pepsy public functionality wherever possible; drop
  to `quimb`, `autoray`, or `cotengra` only when the example specifically needs
  lower-level tensor-network, backend, or contraction-planning control.
- Do not add Gaugy or Tensy examples/notebooks under `examples/`; those belong
  in their own repositories.
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
- MPS layout/canonicalization review: `pytest -q tests/test_optimize_mps.py tests/test_optimize_mpo.py tests/test_symmetric_tensors.py`
- Stabilizer tensor networks: `pytest -q tests/test_stabilizer_tn.py tests/test_stabilizer_tn_stress.py`
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

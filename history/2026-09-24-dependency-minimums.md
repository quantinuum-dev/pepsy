# 2026-09-24 — Verify dependency minimums

- Scope: user requested verification of declared minimum versions against
  the APIs Pepsy actually uses.
- Branch / baseline: `develop` / `6536b72`.
- Commit status: implementation and validation accompany this commit.

## Changes

- Raised unsupported runtime/optional dependency floors based on published
  source, Python support, and isolated lower-bound probes. Detailed versions,
  failures, and limits are in the [dependency audit](../docs/development/notes/dependency_minimums_2026_09.md).
- Core CI now generates its minimum constraints from `pyproject.toml` and
  keeps its existing test selection. Extended CI still resolves current
  releases. Added `pip check` to both test profiles; no additional full-suite
  job, numerical implementation, or test case was introduced.
- Shared `genpy` was not changed. Isolated wheel/dependency probes used the
  required Python 3.12 interpreter and temporary directories.

## Validation

- Exact five core floors: **1370 passed, 319 skipped**, 2188 deselected.
- Exact NetKet/JAX/Flax/Optax floors: **57 passed**.
- Torch 2.4: **60 backend tests passed, 1 skipped**; **49 VMC tests passed**.
- Stim 1.13: **14 passed**; NLopt 2.9: **7 passed**; Matplotlib 3.9:
  **8 passed**; Nevergrad 1.0.3: **1 passed**; Autograd 1.7: **1 passed**.
- Public API/package metadata: **58 passed**. Ruff, whitespace, workflow
  constraint generation, local links, wheel/sdist build, and `twine check`
  passed. Wheel requirements matched every core/optional declaration.
- A fresh resolver dry run accepted the exact core constraints with current
  development tools. No claim of every optional-version cross-product or
  every historical transitive/tool minimum is made.
- Exact mpi4py 3.1.5 was checked against upstream Python 3.12 support notes,
  not rebuilt locally. Cotengrust 0.1 needed an unavailable Rust toolchain on
  this Mac; its floor was retained and its Cotengra fallback remains tested.

## Hosted status

- The preceding repair's [run 36088818562](https://github.com/quantinuum-dev/pepsy/actions/runs/36088818562)
  completed successfully, including extended CI. This closes the pending
  hosted result in the [earlier handoff](2026-09-24-python312-baseline.md).
- Hosted validation of this minimum-version CI change has not yet completed.

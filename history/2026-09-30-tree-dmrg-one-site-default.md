# 2026-09-30 — Default one-node refinement for Tree DMRG

- Scope: user requested the TreeOptimizer equivalent of default one-site DMRG
  refinement and a report of Tree DMRG defaults.
- Branch / baseline: `develop` / `f01f596`.
- Status: working-tree edits only; no staging, commit or publication. Existing
  changes in Pepsy and sibling repositories were preserved.

Changed TreeOptimizer's constructor block default from two to one; documented
initialization, iteration/pass semantics and stopping controls. Explicit larger
blocks and named schedules remain available; no TreePeps/standalone FIT or
notebook changes in this task. Updated Tree skill guidance/catalog and changelog.
See [implementation and audit evidence](../docs/development/notes/2026-09-30-tree-dmrg-one-site-default.md).

## Validation

Shared Python 3.12 environment, local source first, CPU-only subprocesses with
one numerical thread; no environment modifications.

- Initial focused dense/mode/explicit-warmup checks: **12 passed**.
- Final default-specific dense/native checks: **12 passed, 193 deselected**
  in 5.34 s, including four newly added native cases.
  Log: `/tmp/pepsy-tree-one-site-default-final.log`.
- Broad Tree/FIT/API checks: `pytest -q -ra -o addopts='' -m 'not slow'
  tests/test_optimize_tree.py tests/test_tree_*.py tests/test_public_api.py
  tests/test_package_layout.py`: **1498 passed, 101 skipped, 1 existing failure**
  in 231.02 s. Failure: `test_package_version_matches_installed_distribution`
  (installed metadata 0.4.0 versus project 0.5.0), already observed in the
  earlier MPS work. Skips concern hidden CUDA/CuPy and disabled JAX x64.
  This run collected before the four new native default regressions were
  added; those all passed in the separate final selection above.
  Log: `/tmp/pepsy-tree-one-site-domain.log`.
- Ruff `src tests`, Tree skill validator, 12-skill catalog validation, affected
  local documentation links and `git diff --check` pass.

No GPU performance or full-package-suite claim. Numerical coverage here uses
NumPy, Torch CPU and native Symmray; other accelerator paths are not validated
by the CPU run. Historical handoff counts are not reused as new results.

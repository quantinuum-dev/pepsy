# 2026-09-28 — DMRG replay stages and connected-amplitude documentation

- Scope: user's approval to apply the Quimb-inspired readability priorities:
  connected-amplitude contracts, then MPS and MPO DMRG orchestration.
- Branch / baseline: `develop` / `f410fa0`, preserving all prior uncommitted
  work, including the [Tree-PEPS diagnostics fix](2026-09-28-tree-peps-bond-diagnostics.md).
- Commit status: working-tree changes; not staged, committed, or pushed.

## Changes

- Extracted `_run_dmrg_single_window` and `_run_dmrg_batch_window` in each
  existing optimizer module. Public signatures, defaults, existing hooks,
  target/guess ownership, seed arithmetic, rollback, and diagnostic timing
  remain intact. No source modules or dependencies were added.
- MPS driver body: **843 → 250 lines**. MPO driver body: **553 → 300 lines**.
  These counts exclude leading docstrings and indicate navigation changes,
  not a runtime speedup. MPS and MPO helpers retain distinct norm, scheduling,
  and fallback contracts.
- Expanded both Torch connected-amplitude docstrings and added a small CPU
  example and contract table to the VMC guide. Executable bodies are unchanged.
  Documented the boundary cache's parameter-version key and the need to clear
  it when beginning a fresh autograd graph.
- Updated the optimizer/VMC module maps and readability notes. No tests or
  CI settings were added or changed in this pass.

## Validation

- The four extracted bodies match their pre-edit ASTs. Expanding the helper
  calls reconstructs both entire optimizer modules' original executable ASTs;
  argument forwarding and return bindings are checked. Both Torch method
  bodies/signatures are unchanged after excluding their docstrings.
- Initial existing optimizer suites: **390 passed, 11 warnings in 4.19s**.
- The new documentation example runs and its amplitudes match direct model
  evaluation. Two successive fresh autograd graphs, with the documented cache
  clearing, match direct model gradients at `atol=rtol=1e-9` on the small
  dense CPU case.
- Expanded existing MPS, MPO, native symmetry, and Torch VMC domain suites:
  **1,310 passed, 38 skipped, 29 warnings in 184.44s (3m04s)**, exit 0.
  The skips require unavailable CuPy/CUDA/Metal.
- Ruff, the two CI mypy targets, all 54 local Markdown links in changed
  guides/handoffs, and whitespace checks pass. Targeted Pylint checks for
  undefined and possibly/unconditionally uninitialized variables pass in
  both changed optimizer modules, exit 0; no new suppressions were added.
- Default smoke: `MPLBACKEND=Agg python -m pytest -q` → **89 passed,
  2 compatibility-alias warnings in 20.61s**, exit 0.
- Existing fresh-process import profiler, three samples per profile with
  `NUMBA_DISABLE_JIT=1`: root/core/sampling/optimizers/experimental load none
  of the checked optional dependency roots. Medians were 20–22 ms on this
  machine during validation; this is not a cross-machine performance promise.
- Full combined working-tree suite: `MPLBACKEND=Agg python -m pytest -q -ra
  -o addopts=''` → **5,168 passed, 129 skipped, 787 warnings in 514.00s
  (8m34s)**, exit 0. This includes the prior uncommitted Tree-PEPS diagnostics
  fix and readability changes. Skips require unavailable CuPy/CUDA/Metal,
  multiple MPI processes, or the explicit two-device JAX configuration;
  those paths were not validated by this run.

## Limits

This continues the existing compatibility audit without changing upstream
calls, numerical kernels, or optional dependency boundaries. Native and
backend behavior is checked through the existing suites. GPU/distributed
coverage remains subject to the available environment. Other Pylint findings
remain a separate review backlog; no package-wide clean-lint claim is made.

# 2026-10-06 — PepsOptimizer default behavior review

- Scope: explain the current default class and review its correctness; no fixes.
- Branch / baseline commit: `develop` / `8f7c896`.
- Commit status: review note and this handoff are uncommitted working-tree
  additions. Nothing staged, committed, or published. Existing files preserved.

Read the complete optimizer, relevant gate/boundary/sweep paths, API guides,
closest tests and historical PEPS reviews. Recorded current behavior, concrete
reproductions, installed dependencies and limits in the
[review report](../docs/development/notes/2026-10-06-peps-optimizer-default-review.md).

Confirmed: approximate normalization plus fixed unit-norm evaluation can fail
on a default 4x4 identity update; `bond_dim` silently bypasses exact-target
protection; string-index gates receive invalid routing options; sweep FIT
diagnostics are dropped; and the documented temporary mode override persists.
The Symmray default-solver caution is stale. These remain unfixed.

Fresh checks: **329 passed** across `test_optimize_peps.py`,
`test_optimize_global.py`, and `test_prepare_boundary_inputs.py`; Ruff and
diff checks passed. Small NumPy and Torch CPU dense-reference probes checked
target accuracy, actual refinement, normalization, dtype and input isolation.
No full suite, GPU, or broad PEPO/cyclic validation was performed.

Proposed next work, subject to user scope: address exact-target alias protection
and consistent evaluation normalization first, then API/diagnostic issues.

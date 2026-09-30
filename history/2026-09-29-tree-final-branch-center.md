# 2026-09-29 — Retain the final center after branched direct compression

- Scope: implement the user-approved optimization from the
  [routing audit](2026-09-29-tree-routing-canonical-audit.md), with correctness
  checks for subsequent updates and native fermionic paths.
- Branch / baseline commit: `develop`, `b343ced`, with the earlier session's
  norm/backend and Kraus repairs and concurrent unrelated work present.
- Commit status: working-tree edits only; nothing staged, committed or
  published. Existing changes were preserved.

## What changed

- `TreeOptimizer._compress_subtree` retains the center at the final visited
  tensor for branching direct/DM compression. A recursive flag preserves
  every return needed for another sibling or ancestor branch and omits only
  the final return chain to the hub. Cut order is unchanged.
- Existing TTN wrappers maintain canonical-region and `left_inds` metadata.
  The recursion reports the actual terminal center. Path compression and
  successive compression algorithms are unchanged.
- Added `tests/test_tree_branch_execution.py` and focused device coverage in
  `tests/test_tree_gpu_backend.py`. Updated the public replay guide and
  changelog. The
  [implementation note](../docs/development/notes/2026-09-29-tree-final-branch-center.md)
  records the invariant, tradeoff, and reused upstream audit.

## Validation after the implementation

All commands used the existing Python 3.12 environment and single-threaded
BLAS/OpenMP. CPU-only runs hid CUDA; device runs exposed it explicitly.

- `python -m pytest -q -o addopts='' tests/test_optimize_tree.py tests/test_tree_*.py`
  — **1,361 passed, 83 skipped, 20 warnings**, 159.81 s. Includes native
  Symmray, NumPy/Torch/JAX CPU paths, path/branch routing, FIT, measurement,
  scale/stabilization and layout checks. GPU-dependent cases skip here.
- The initial new branch regression selection — **42 passed**, 8.61 s.
  These cases are also included in the broad run above; do not add the counts.
  They compare exact cut/return traces and completed states with an explicit
  reference that returns after every child. They cover binary/ternary trees,
  lossy/lossless direct/DM, canonical proofs, the local norm metric, native
  graded updates, and mixed-support sequences with DMRG `guess-direct`.
- Two Torch gradient checks were added after the broad run had collected
  tests and run separately: **2 passed, 42 deselected**, 2.30 s. Lossy and
  lossless branch state gradients match the reference sweep.
- New branch/device selection in `test_tree_gpu_backend.py` with
  `JAX_PLATFORMS=cpu` and CUDA exposed — **8 passed, 148 deselected**,
  23.33 s: direct/DM on Torch CPU, Torch CUDA, JAX CPU and CuPy.
- The same two JAX cases with `JAX_PLATFORMS=cuda` and
  `JAX_DEFAULT_MATMUL_PRECISION=highest` — **2 passed, 154 deselected**,
  15.63 s. Production precision settings were not changed.
- Initial path/native/canonical selection — **105 passed, 2 skipped**,
  9.48 s; subsumed by the broad run.
- Repository `python -m ruff check src tests` and `git diff --check` passed.

Logs: `/tmp/pepsy-tree-final-center-domain.log`,
`/tmp/pepsy-tree-final-center-regressions.log`,
`/tmp/pepsy-tree-final-center-gradients.log`,
`/tmp/pepsy-tree-final-center-backends.log`,
`/tmp/pepsy-tree-final-center-jax-gpu.log`, and
`/tmp/pepsy-tree-final-center-initial.log`.

## Decisions and limits

- The completed update agrees with the old sweep up to roundoff. Necessary
  interbranch returns still occur. The saved return count equals the distance
  from the final tensor to the hub; this is not a measured wall-clock speedup.
- A different incoming center can change a later path sweep's direction and
  finite-bond trajectory. Exact multi-gate comparisons use sufficient bond
  dimension; finite-bond sequences check canonicality and local norm behavior.
- Local compression infidelity remains the user's requested cheap
  canonical-norm metric with log accumulation. No diagnostic target,
  overlap contraction, timer or backend-specific numerical kernel was added.
- Full-package and full-GPU suites were not rerun for this tree-only change.
  No blockers remain within the authorized scope. Concurrent operator work
  is outside this handoff's validation claims.

## Second review at the user's request

Re-read the recursive traversal, all `_compress_subtree` callers, TTN edge
center transitions, FIT guess handoff, normalization and stabilization. No
implementation defect was found and production code was unchanged in this
follow-up. Every earlier child returns to its parent, including the last child
of a subtree whose ancestor has more work. Only the final descent omits its
return chain. Callers read the actual state-owned center rather than assuming
the old hub endpoint.

One test-reference defect was found and fixed: `_returning_branch` always
preserved sub-cap bonds, ignoring the explicit `preserve_subcap=False` used by
ordinary gate replay. The earlier small-cutoff cases did not expose this.
Adding chi=64/cutoff=0.1 reproduced **two expected reference-test failures** on
binary/ternary NumPy direct cases. Correcting the reference's cutoff policy
resolved them without changing production code or tolerances.

Additional permanent checks cover:

- Nonzero cutoff with bonds already below the cap.
- Uneven trees, unary nodes, irregular structural ids, permuted qubit labels,
  a physical root, exterior legs, and four different initial hubs including
  a leaf. Both cutoff policies and explicit bond/cutoff overrides are checked
  for NumPy/Torch direct/DM against the corrected reference.
- Lossy branched updates with stabilization, stored exponents ±400, and
  infidelity tracking enabled/disabled. The retained center preserves working
  norm, normalized state direction, exponent, and local compression loss.

Fresh CPU validation, CUDA hidden and BLAS/OpenMP single-threaded:

- Branch regression file before the final scale cases: **84 passed**, 10.95 s.
- `python -m pytest -q -o addopts='' tests/test_tree_branch_execution.py
  tests/test_tree_path_execution.py tests/test_tree_canonical_regions.py
  tests/test_tree_native.py tests/test_tree_submpo.py
  tests/test_tree_fit_messages.py tests/test_tree_unitary_stability.py`
  — **360 passed, 2 skipped, 1 warning**, 61.68 s; includes all **100** current
  branch cases. The warning concerns approximate private MPO expectation
  compression in an existing native test.
- Repository Ruff and `git diff --check` passed. Full-package and GPU tests
  were not repeated in this follow-up; the earlier GPU results above apply
  to the unchanged implementation.

Logs: `/tmp/pepsy-tree-final-center-review-gap.log` (expected failures before
repairing the reference), `/tmp/pepsy-tree-final-center-review-regressions.log`,
and `/tmp/pepsy-tree-final-center-second-review.log`. Only tests and this
handoff changed in the second review; everything remains uncommitted.

# 2026-10-08 — CuPy full update and commuting-strip scheduling

- Scope: user requested CuPy support, reuse of boundary MPS and partial
  contractions along rows/columns, and better scheduling within a gate depth.
  The subsequent question about adding local DMRG refinement was investigated
  as a design question; that refinement is proposed, not implemented.
- Branch / baseline: `develop`, `fe11e24`.
- Commit status: working-tree edits only; nothing staged, committed or pushed.
  Earlier full-update edits and concurrent backend/MPS/JAX changes remain.
  This task did not edit those unrelated backend/MPS files.

## Implemented

- Full-update accepts dense CuPy complex64/complex128 alongside Torch, using
  Pepsy's existing conversion helpers and Autoray/Quimb native operations.
  Shared Quimb/QR ALS already supported CuPy. Native state/dtype/device are
  preserved, and public runs bind CuPy operations to the state's device.
- Exact CuPy snapshots on the GPU provide mutation revisions for the existing
  outer boundary and inner strip caches. Only comparison booleans reach the
  host. Weak references release snapshots, alias writes invalidate correctly,
  and storage/comparison overhead is documented. Torch counters are unchanged.
- Row/column ordering accepts disjoint fixed non-diagonal two-qubit gates.
  Overlapping gates commute only when both are diagonal. Conflicts and
  unsupported/single-site/trainable gates preserve necessary order.
  Complete one orientation before turning to transverse strips; input order
  remains the default. This is a heuristic, not a globally optimal schedule.
- API docs and changelog updated. Detailed measured evidence and the proposed
  strip-refinement design are in the
  [CuPy/strip note](../docs/development/notes/2026-10-08-full-update-cupy-and-strip-refinement.md).

## Validation

Activated the existing `py312` environment. CuPy 14.1.1 used a real NVIDIA
RTX A5000. BLAS/OpenMP/Numba were limited to one thread; Cotengra planning was
bounded or used greedy. No dependency changes.

- Combined affected PEPS, shared reduced-update, CuPy, gate-order and cache
  suites: **431 passed**, 66 warnings, 114.74 seconds.
- Subsequently added CuPy `eff`/`dmrg2` boundary checks: **2 passed**,
  11 deselected, 33 upstream warnings, 3.77 seconds.
- Default smoke: **94 passed**, two warnings, 32.14 seconds.
- Ruff and `git diff --check`: passed.
- Three 4×4 CuPy TFIM runs cover cached/fresh real time and imaginary time.
  Cached/fresh vectors agree to 9.38e-9 in phase-aligned norm. Both cache
  levels register reuse, norms stay within 1.6e-15, and imaginary energy falls.
- An intermediate run exposed a decorator regression on test DummyState
  objects after adding the device context. The wrapper now inspects optional
  tensor metadata safely; the final 431-test run includes those tests.
- Default complex64 boundary cutoff can produce a rank-truncation plateau
  despite increasing chi; convergence correctly reports failure. Reference
  comparisons use zero cutoff. No production threshold was weakened.
- No multi-GPU or large-D performance study. The preceding full-suite attempt
  reproduced 24 baseline BP failures on unchanged HEAD; it was not repeated
  and is not represented as passing by these focused checks.

Logs: `/tmp/pepsy_cupy_final_regression.log`,
`/tmp/pepsy_cupy_dmrg_tests.log`, `/tmp/pepsy_cupy_smoke.log`.
GPU reproducer: `/tmp/pepsy_itf_4x4_cupy_review.py`; outputs under
`/tmp/pepsy_itf_4x4_cupy_results/`.

## Proposed strip refinement

Recommend one optional fixed-D, one-site ALS pass on a completed strip,
against the untruncated target for that block. Reuse norm and candidate–target
overlap boundary contractions. Keep the best candidate and expand to adjacent
strips only after measuring the benefit. Existing `SweepOptimizer` uses
gradient solves for PEPS tensors; its DMRG controls fit boundary MPS. Native
strip ALS therefore needs an explicit adapter and active-strip restriction.
Refitting the compressed state to itself cannot recover the original error.
No new refinement option was added in this task.

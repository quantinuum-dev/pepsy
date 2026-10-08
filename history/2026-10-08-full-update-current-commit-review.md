# 2026-10-08 — Review of current PEPS full-update commits

- Scope: review PEPS full-update implementation for bugs and inefficiencies;
  no implementation fixes requested.
- Branch / baseline: `develop`, `ad8ed05`; inspected `fe11e24`, `6de7551`,
  and `ad8ed05` and their integration with shared reduced ALS/boundaries.
- Commit status: this review handoff is uncommitted; nothing staged or published.
  The tree was initially clean. Concurrent changes later appeared in the MPS
  optimizer/API guide, changelog, and `tests/test_mps_traced_backend.py`;
  those files were not changed by this review.

## Confirmed findings

### P2 — Strip refinement bypasses two-site gate conversion

`optimizer.py:2485` builds the block target from the original gate payload.
`ReducedPair` converts its own gate to the PEPS backend/dtype/device, but that
converted gate is not propagated to the block-target path. Consequently,
enabling `full_update_kwargs={'refine_sweeps': 1}` breaks gates accepted with
refinement disabled.

Reproduction: seed-24 normalized Torch complex128 2x2 D2 PEPS, one gate
`diag(exp([-0.3j, 0.3j, 0.3j, -0.3j]))` on `((0,0),(0,1))`, retained chi 2,
direct boundaries, boundary/normalization chi 16, convergence disabled,
four ALS iterations, independent metric/acceptance checks disabled.

- NumPy gate, refinement off: succeeds; refinement on: `TypeError`,
  `tensordot(): argument 'other' (position 2) must be Tensor, not numpy.ndarray`.
- Torch complex64 gate, refinement off: succeeds; refinement on:
  `RuntimeError: both inputs should have same dtype`.
- Matching Torch complex128 gate succeeds with both settings.

Proposed correction: convert once at the driver boundary and use the same
native payload for pair reduction and the exact block target. Add mixed
backend/dtype refinement regressions. No device-mismatch reproduction was run.

### P2 — Single-site full update ignores target normalization

`optimizer.py:2478` continues after applying a single-site gate without
honoring `normalize_target`. The ordinary sweep branch explicitly normalizes
at the corresponding point. Final normalization cannot compensate when it
is intentionally disabled.

Reproduction: normalized seed-24 Torch complex128 2x2 product PEPS, single
gate `2*I` on `(0,0)`, `non_unitary=True`, `normalize_target=True`,
`normalize_final=False`, direct boundaries, normalization chi 16,
convergence and independent metric/acceptance checks disabled. Full-update
returns norm 2; ordinary sweep returns norm 1. This also affects the
`non_unitary=True` default target-normalization policy when no explicit
`normalize_target` override is supplied.

Proposed correction: apply the resolved target-normalization policy to
single-site updates, preserving the scale of any active refinement target.

### P2 — Smart scheduling does cubic predecessor-copy work

`_gate_order.py:139-149` visits every remaining pair for each emitted pair,
materializing `list(predecessors[i])` even when the first inspected ancestor
already proves that gate unavailable. For alternating noncommuting XX and ZX
rotations on one bond, the dependency graph is dense and these repeated
copies require cubic work. The entire traversal is also run for both preferred
orientations, even though this example has only one legal order.

CPU compilation timings (no tensor updates), same process, shared NumPy gate
objects, angle 0.1: 100 gates 0.021 s; 200 gates 0.086 s; 400 gates 0.378 s;
800 gates 1.877 s; 1600 gates 10.100 s. These are single-run measurements.
Instrumenting one `_schedule` invocation counted 84,575 / 671,650 / 5,353,300
copied predecessor indices at 100 / 200 / 400 gates, confirming approximately
eightfold growth per doubling separately from noisy wall-clock timing.

Proposed correction: maintain readiness incrementally, avoid copying blocked
predecessor sets, and reconsider only affected candidates after emission.
The documented quadratic dependency storage does not explain this additional
cubic traversal cost. Shorter queues or `gate_order='input'` avoid this case.

## Validation

Activated the existing `~/envs/py312`, with one BLAS/OpenMP/Numba thread and
`LOKY_MAX_CPU_COUNT=2` for pytest. Installed versions: Quimb
1.15.1.dev90+g6a3906cbe, Autoray 0.11.1.dev14+g014a3f69a, Cotengra
0.8.3.dev8+g8954240f2, Symmray 0.4.1.dev15+g0374aaa3c, Torch 2.6.0+cu124,
CuPy 14.1.1. Inspected installed public ALS and PEPS gate signatures.
No dependency changes or compatibility shims proposed.

- Full-update, strip-refinement, gate-order, smart-gate-order, and environment
  reuse modules: **108 passed**, one warning, 21.98 s.
- Boundary-convergence, shared BP reduced-update, and CuPy full-update modules:
  **85 passed**, 47 warnings, 17.02 s.
- Combined: **193 passed**, no skips; includes Torch CPU and CuPy tests.
- Targeted probes reproduced the failures and profiled scheduler behavior.
- `git diff --check` passed. No Python implementation/test edits; no new lint,
  default-smoke, or full-repository suite run.

Temporary probes: `/tmp/pepsy_review_probes.py`,
`/tmp/pepsy_review_more_ad8ed05.py`, and
`/tmp/pepsy_review_scheduler_counts_ad8ed05.py`. Inputs and results above
remain the durable evidence if temporary files disappear.

Earlier review findings were checked against the current fixes rather than
reported again. Passing existing tests does not cover the newly reproduced
cases. Findings remain unfixed; larger-scale numerical accuracy and long-time
GPU evolution were not validated in this review.

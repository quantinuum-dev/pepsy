# 2026-10-06 — FIT inside PEPS sweep mode

Read-only implementation review on `develop`, baseline `8f7c896`, including
the existing uncommitted PEPS fixes. No production or test code changed in
this review. Followed the maintainer router and tensor-fitting skill; reused
the unchanged environment/upstream audit from the earlier PEPS review.

## Algorithm and defaults checked

The dense sweep path has two different optimizations:

1. `CompBdy` compresses the norm and target-overlap environments into boundary
   MPSs. It constructs a disposable SRC guess, then fits against the boundary
   target network. The default calls `FIT.run_eff`, with one-site updates,
   alternating `RL` directions, ten directional sweeps, and no tolerance stop.
   One-site fitting preserves the guess ranks. Optional two-/three-site
   updates can grow visited bonds up to the requested boundary cap.
2. `SweepOptimizer` optimizes an entire PEPS row/column against those fixed
   environments using NLopt `LD_LBFGS`, `maxeval=50`, and best-iterate restore.
   This is distinct from FIT's one-site boundary update. The objective is
   `1 - |<candidate|target>|^2 / (norm_candidate * norm_target)`.

The default PEPS driver supplies target norm one for unitary evolution;
boundary FIT canonical-center norms are not measurements of the PEPS target
norm. The driver normalizes the output using its normalization cap. Finite
boundary caps make this an approximate normalization, not an exact guarantee.
The default sweep schedule uses one y/x cycle, with an initial forward pass
and four backward/forward round trips on each axis. Dense automatic boundary
routing uses FIT; native Symmray automatic routing uses Quimb MPS boundaries.

Inspected target/guess separation, canonicalization, effective tensor
writeback, changing-direction environment caches, exponent handling, local
bra conventions, solver parameter restoration, best-state tracking,
normalization ownership, boundary diagnostics, and termination rules.

## Measured checks

- PEPS optimizer/batching, boundary-input, FIT hot-path, and fermionic-boundary
  suites: **374 passed**, 558 warnings, 42.89 seconds.
- `tests/test_mps_fit_kernels.py -k 'run_eff or full_chain_runs_validate'`:
  **22 passed**, 79 deselected, 2.78 seconds.
- Independent exact-reference probe: normalized random 2x3 D=2 complex128
  Torch PEPSs, seeds 291/293, boundary cap 32, SRC seed zero, ten FIT sweeps,
  greedy contractions. Replaced the local solver with a deterministic probe
  that perturbs and applies the current row/column parameters. Across all
  11 updates in one forward/backward/forward traversal of each axis,
  including reused environments after earlier slices changed, local loss
  differed from dense normalized fidelity by at most **1.11e-16**.
  Complex Torch directional derivatives differed from centered finite
  differences (step 1e-5) by at most **4.28e-12**. Perturbations used Torch
  seeds `301 + update_number`, with complex Gaussian entries scaled by .02.

The exact-reference probe checks local objectives, gradients, and environment
reuse, not NLopt convergence. Real local-solver behavior is covered by the
focused suites. No full-suite, GPU, large-lattice, or universal convergence
claim follows from these results.

## Confirmed issues

### Invalid initial sweep score is classified as convergence

`SweepOptimizer._run_global_sweeps` accepts every initial loss below 1e-10,
including substantially negative boundary artifacts. The PEPS wrapper then
rejects the negative result before the outer acceptance/rollback check.
The previously reproduced real 3x3 example has valid outer infidelity
7.43e-5 but inner loss -0.03043. This review reconfirmed the unchanged code
path; the numerical reproduction is earlier evidence in the
[second review](2026-10-06-peps-second-review.md).

Proposed: invalid estimates must not signal convergence; preserve the warm
start and report unsuccessful cleanup, or use an explicit environment retry.
Do not clip a substantial negative value to zero.

### Local solver output is applied before validating its loss

`_optimize_axis_slice_with_current_env` checks the initial loss for finiteness
and nonnegativity, but after solving it computes `applied_loss` and applies
the returned parameters without the corresponding guard. The best-state
tracker rejects invalid scores only after the state was already modified.

Controlled fault injection on the same 2x3 fixture returned NaN parameters
from `_optimize_packed_params`. Initial loss was **0.9982623872181167**;
the method returned `loss_final=NaN`, no `invalid_loss` flag, and left
nonfinite tensors in the live state. This is a confirmed missing defensive
check, not a reproduction of default NLopt spontaneously producing NaNs:
NLopt's own finite checks and best-iterate restore reduce that risk. A
later full-sweep best-state restore can help only if execution reaches it.

Proposed: reject nonfinite/invalid returned candidates before writeback and
retain the current valid slice, with an explicit diagnostic. Meaningful
regressions should cover both rejected output and unchanged live state.

## Accuracy and reporting limitations

Ten fixed FIT sweeps do not prove boundary convergence. One-site FIT cannot
repair insufficient boundary rank by increasing it; boundary caps and
optional block updates control that approximation. Best local losses across
different approximate environments are diagnostic estimates, not guaranteed
globally ordered fidelities; the PEPS driver's outer acceptance check matters.
Known-target norm one additionally depends on input normalization accuracy.

The fitting skill's blanket statement that `run_eff` is independent of
adaptive stopping is stale relative to the implemented and tested optional
`rtol` path. Default fixed-sweep behavior remains consistent. This review
does not change skill policy or algorithm behavior.

Only this evidence note and its history handoff were added. All proposed
fixes remain unimplemented; prior working-tree changes were preserved.
Nothing staged, committed, or pushed.

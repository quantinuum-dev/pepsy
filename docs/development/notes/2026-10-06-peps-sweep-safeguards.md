# 2026-10-06 — Preserve valid PEPS states on invalid sweep estimates

Implemented the two safeguards authorized after the
[FIT/sweep review](2026-10-06-peps-fit-sweep-review.md). Branch `develop`,
baseline `8f7c896`; changes remain uncommitted. Existing PEPS work and
concurrent gradient-solver edits were preserved.

## Implemented behavior

- A nonfinite or substantially negative initial sweep infidelity stops
  cleanup with `success=False`, `converged=False`, and
  `termination_reason="invalid_initial_loss"`. The raw value stays in
  `loss_before`; no invalid estimate is exposed as an accepted final loss.
  No local updates occur. The existing local tolerance of 1e-10 also
  distinguishes early-exit roundoff from invalid negative estimates.
- The PEPS driver propagates failure diagnostics and retains its normalized
  warm start with `reason="optimizer_failed"`, including when acceptance
  checks or fidelity measurement are disabled. Failed cleanup performs no
  additional postcheck or output normalization.
- Before evaluating returned local parameters, the sweep checks finiteness
  of native parameter-tree leaves. Reductions stay on the array backend;
  only one combined boolean is transferred. A nonfinite or substantially
  negative returned loss is rejected before slice writeback. Diagnostics
  distinguish `nonfinite_parameters` from `invalid_candidate_loss` and keep
  the retained slice's loss separate from the rejected candidate's loss.

No cap growth, extra target-norm contraction, FIT stopping change, or NLopt
budget change was introduced. Boundary accuracy remains the caller's
responsibility. Only failure handling and near-zero early-exit consistency
changed; contraction/compression kernels and upstream registrations did not.
The unchanged environment/upstream audit was reused.

## Validation

`tests/test_peps_sweep_safeguards.py` adds 23 cases:

- Invalid initial scores (-.03, NaN, positive/negative infinity) preserve
  tensors and never enter a sweep; valid near-zero scores still converge.
- NumPy/Torch local NaN/Inf parameters and invalid scalar losses preserve
  every tensor and do not enter best-state storage.
- The real 3x3 D=2, seed-1 RZZ(.01), boundary-cap-1 reproduction now returns
  the same normalized state as `optimize=False`. Exact outer metrics are
  used. All four combinations of fidelity measurement and acceptance checks
  enabled/disabled pass and report explicit failed cleanup.
- U1/U1U1 fermionic native NumPy/Torch block trees remain unchanged after
  injected nonfinite solver output, including block keys and index metadata.

Fresh domain selection: **431 passed**, 560 warnings, 54.42 seconds across
the safeguard tests (first 19), real boundary-engine numerical regressions,
PEPS optimizer/batching, boundary preparation, fermionic boundaries, public
API, and package layout. The four subsequently added native safeguard cases
passed separately: **4 passed**, 19 deselected, 2.79 seconds. Total distinct
passing cases: **435**. No skipped cases in those selections.

The initial native test fixture incorrectly assumed `SymPEPS.random` used
Torch by default; it actually returned NumPy blocks. Corrected the fixture
to request explicit Torch conversion and cover both backends. No production
change was required for that test-construction failure.

`python -m ruff check src tests`, review-link checks, and `git diff --check`
passed. No full-suite or GPU claim. Existing real default-schedule tests for
Quimb MPS, DMRG, and DMRG2 passed after the safeguards.

Updated the PEPS/sweep API guides and changelog. Nothing staged, committed,
or pushed. Other earlier findings (global normalization ownership and stored
retry-option forwarding) are outside this authorized two-fix scope.

# 2026-10-08 — Strip ALS refinement and local fidelity records

- Scope: implement the approved optional local ALS/DMRG-style refinement
  after two-site full updates, and efficient per-gate fidelity accumulation.
- Branch / baseline: `develop`, `fe11e24`.
- Publication: initially implemented as working-tree edits; the user later
  authorized final review, commit, and push. See publication checks below.
  Earlier full-update work is included; concurrent backend/MPS changes are
  preserved separately and excluded from this commit.

## Implemented

- New private `_strip_update.py` adapts public Quimb native ALS to an active
  row/column, reusing norm/overlap boundary MPS and prefix/suffix contractions.
  An exact block target survives pair truncations; repeated bonds and strip
  or single-site barriers close the block. Fixed exterior arrays and rank.
- `refine_sweeps=0` defaults off; positive values enable bounded alternating
  passes, with `refine_rtol` matching pair dtype tolerance policy. Best-state
  rollback, adaptive block calibration, final normalization/remeasurement,
  timing, and explicit step reports are integrated in the driver.
- Local pair fidelities are retained for every two-qubit gate with no extra
  global metric contraction. Rejected ALS reports the retained warm start.
  `accumulate_local_infidelity=True` optionally records a stable product;
  strip-target scores remain distinct. Unmeasured post-refinement final
  scores are `None`.
- Public API guide, changelog, regression tests, and detailed
  [numerical evidence](../docs/development/notes/2026-10-08-full-update-strip-refinement.md)
  updated. The prior proposed-only note links to this implementation.

## Validation and findings

- Activated existing `py312`; real NVIDIA RTX A5000 CuPy tests, native Torch
  CPU, complex64 and complex128. No dependency or upstream modifications.
- Refinement/full-update focused tests: **60 passed**, one warning, 17.92s.
- Combined affected tests: **448 passed**, 98 warnings, 111.58s. Log:
  `/tmp/pepsy_strip_combined.log`. Includes shared reduced-update, CuPy,
  scheduling, caching, adaptive boundaries and sweep regressions.
- Default smoke: **94 passed**, two warnings, 34.83s; log:
  `/tmp/pepsy_strip_smoke.log`. Ruff and `git diff --check` passed.
- 4×4 TFIM experiments confirm normalized bounded-rank evolution. One strip
  pass improves accepted local fits but slightly worsens final D=2 real-time
  circuit infidelity (~0.1% relative). D=2 imaginary and D=4 real-time cases
  improve. Keep refinement opt-in; the local product is not a global bound.
- D=2 refinement adds approximately 1.7× elapsed time in these short runs;
  large-D scaling remains unverified. Invalid boundary estimates/candidates
  in the D=4 run were safely rejected. Full details and temporary reproducers
  are in the numerical note.
- Quimb selections drop parent exponents. The adapter restores strip scale
  and explicitly transfers relative local-equation scale to the RHS before
  calling public ALS. Dense-reference tests cover this integration detail.
- The earlier full-suite attempt reproduced 24 baseline BP failures on
  unchanged HEAD; no new full-suite run or passing claim is made here.

## Remaining limitations

Refinement covers one active strip, not neighboring strips or the full lattice.
Dense local eigensystem cost and exact target growth can dominate at larger D.
No multi-GPU or differentiable full-update support was added. No blocker to
the requested opt-in implementation or fidelity bookkeeping remains.

## Final publication review

- Separated physical norm validity from the user-selected ALS stopping
  tolerance: a large `refine_rtol` cannot authorize a substantially complex
  norm. Nonfinite imaginary norms and overflow-sized overlaps also reject
  refinement safely. Three regression cases cover these checks.
- Staged only PEPS full-update, shared reduced-ALS/cache changes, their tests,
  and related documentation. Staged only the relevant changelog entries;
  unrelated MPS/backend edits remain in the original checkout.
- Created an isolated detached checkout from `fe11e24` and applied the staged
  patch. Verified imports resolve to that checkout, then ran the combined
  affected suite: **451 passed**, 98 warnings, 115.64s. Default smoke:
  **94 passed**, two warnings, 34.05s. Ruff and whitespace checks passed.
  This verifies the change without depending on unrelated local backend edits.
- Logs: `/tmp/pepsy_publish_regression.log`, `/tmp/pepsy_publish_smoke.log`.
  No new full-suite or larger-scale numerical claims beyond those above.

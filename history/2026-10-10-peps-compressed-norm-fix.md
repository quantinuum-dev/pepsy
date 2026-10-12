# 2026-10-10 — Live campaign exposed a pre-solve Hermitian-norm mismatch

- Scope: check the user's background 4×4 sweep comparison and repair a
  confirmed failure to perform the intended site optimization.
- Branch / baseline: `develop`, `b3e7985`; existing uncommitted work preserved.
- Commit status: working tree only, no staging, commit, or push.

## Finding

The campaign `sweep_4x4_20261010_191323UTC` remained alive and saved valid
states, but mature D=2 site-CG/site-L-BFGS runs rejected all 192 strips in a
timestep before performing any site solves. Whole-column L-BFGS continued
to update. Thus process health and checkpoint creation did not establish
that the requested solver comparison was taking place.

At chi=8, compressed norm environments had relative imaginary components
around 1e-5 to 7.5e-4. The local equations already used `(N + N.H)/2`, but
the initial scalar check rejected `A.H N A` whenever its imaginary part
exceeded 1e-10. Earlier small-system tests mostly had effectively exact
boundaries and missed this mismatch.

## Correction

[_als.py](../src/pepsy/optimizers/sweep/_als.py) now uses
`Re(A.H N A) = A.H ((N + N.H)/2) A` for both initial normalization and
candidate scoring. This changes neither the Hermitian solve nor its
regularization/defaults. Nonfinite inputs, nonpositive real norms, and
out-of-range normalized overlaps remain rejected. The independent physical
target-norm check remains strict.

Added both-axis regressions for explicit/matrix-free CG and L-BFGS with a
complex norm environment whose Hermitian quadratic form matches an exact
reference, plus nonfinite/nonpositive norm checks. API docs and changelog
describe the behavior.

The examples runner also requires an actual applied sweep update before
labeling an exact-overlap-accepted candidate as an accepted refinement.
Previously roundoff from renormalizing an unchanged state could mark a
failed sweep accepted. Exact final-overlap rejection remains in place.

## Evidence

- ALS suite: **154 passed**, 31 warnings, 20.98 seconds.
- Mature 4×4 D=2, chi=8 checkpoint replay: each of explicit CG, matrix-free
  CG, and matrix-free L-BFGS performed 128 site solves, all accepted, with
  zero invalid strips. Exact local fidelity improved from 0.9993552299754384
  to 0.999437133845773 / 0.9994371338457788 / 0.9994371367869491 respectively.
  Both CG variants used 1031 total iterations; L-BFGS used 2116 evaluations.
  This is one-gate correctness evidence, not a trajectory speed ranking.
- Broader PEPS/sweep/boundary/refinement/API regression: **400 passed**, 54
  warnings, 100.90 seconds. Downstream trajectory/restart/scheduling checks:
  **13 passed**, 53.72 seconds. Total: **567 targeted checks passed** across
  the three runs. Ruff and `git diff --check` passed in both repositories;
  the full repository suite was not run.

The unchanged-environment upstream audit from the earlier ALS task was
reused. No dependency or contraction dispatch changed; this adopts the
same explicit Hermitian mathematical contract in scalar diagnostics.

## Campaign handling

The original supervisor and three invalid site workers were temporarily
suspended during diagnosis. After validation, all original workers were
stopped and their campaign/status records marked `superseded`; original
source snapshots and outputs are preserved. The corrected campaign is
`sweep_4x4_20261010_192437UTC`, supervisor PID 3649558. All 16 jobs restart
from the initial product state with a new frozen snapshot, avoiding mixed
implementation histories. Its first four D=2 workers are running, and the
frozen worker source contains the correction. `LATEST_SWEEP` selects it, so
the comparison notebook loads corrected results on refresh. D=3/4/5 remain
queued; chi=2D², four workers, eight threads per worker, and t=5 are unchanged.
The other completed baseline campaigns remain unchanged.

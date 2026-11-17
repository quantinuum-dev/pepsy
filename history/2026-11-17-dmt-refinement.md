# 2026-11-17 — Optional fixed-border DMT refinement and 2D cover

- Scope: improve DMT bond approximation beyond the paper's conservative
  rank allocation, and improve joint selection of protected directions for
  overlapping mapped 2D densities. The user's attached revision motivated
  a guarded Schur candidate plus constrained alternating refinement.
- Branch / baseline: `develop`, `328979f4`.
- Status at implementation handoff: working-tree edits only; nothing staged,
  committed or pushed. The publication scope below supersedes that snapshot.
  Preserved existing sampling, test-harness and changelog edits, historical
  notes and the earlier uncommitted DMT skill. No tutorial/sibling repo edits.

## Implemented

- Shared kernel supports `truncation="optimized"` while retaining paper DMT
  by default. Keep a feasible paper seed; test compatible-border SVD with
  conditioning/range checks; alternate equality-constrained native factor
  solves. Accept only feasible nonincreasing one-cut errors within numerical
  roundoff. Final QR must retain feasibility and canonical isometry.
- `max_refinement_sweeps=0` tests only SVD seeds; default eight is the limit
  when the optimized route is selected. Rank-deficient/rectangular corners
  and incompatible pseudoinverses are handled explicitly. No random restarts,
  finite marginal penalties, global fit, positivity or optimum certificate.
- `region_selection="min_cover"` uses bipartite matching/vertex cover to
  minimize the combined physical-string protection count across regions.
  This is a sufficient hard-constraint construction, not virtual-rank or
  globally minimal-region optimization. Original smaller-half default kept.
- Forwarded options through both MPO storage bases, standalone compression,
  Pauli gates/persistent evolution, manual MpoOptimizer compression and gate
  replay, including remapped routed gates. New report fields distinguish
  seed/refinement choice; optimized `residual_rank` is None.
- Added independent objective, rank, isometry, dense marginal, replay,
  backend and gradient regressions; mathematical skill/reference, API guides,
  changelog and [benchmark example](../examples/dmt_refinement_benchmark.py).

## Validation and evidence

- Initial focused DMT/MPO/replay selection: **325 passed, one pre-existing
  expected failure**, 202.27 s. This preceded the final gradient fix below.
- New refiner suite after removing in-place updates and using Torch's
  structured pseudoinverse: **22 passed**, 18.41 s. Then added an unresolved
  pivot/zero-corner case and explicit canonical-isometry assertions; the
  expanded results follow.
- Final expanded refiner, ordinary MPO/replay, backend and API/layout
  selection: **227 passed, one skipped, one existing version-metadata
  failure**, 90.34 s. This includes all **23 new refiner tests**, canonical
  isometries, unresolved/zero corners and the combined-backend complex
  gradient regression. All numerical checks in this selection passed.
- Found and fixed complex autograd failures from in-place residual updates
  and manually differentiated singular-vector pseudoinverses. Torch uses
  native pinv with explicit rtol/atol; a protected-observable derivative is
  compared to its exact linear gradient, including incompatible-corner ALS.
- Ran the complete suite: **104 passed, 7 skipped**, stopped at existing BP
  `test_loopunion_selects_both_plaquettes_and_clips_at_edge[where0-expected0]`
  because a Torch Autoray namespace receives a NumPy array. This reproduces
  the prior documented environment failure; no new DMT code runs there.
- Initial API/layout attempt: **35 passed**, stopped at existing metadata
  mismatch (installed Pepsy 0.5.0 versus checkout 0.6.0). No dependency or
  installation changes were made to mask it.
- Ruff `src tests` plus the benchmark, individual skill metadata, 13-skill
  catalog, mathematical probe with actual kernel, and diff checks passed.
  Mathematical probe's complex six-site marginal defect: 2.52e-15.
  The eight edited/new DMT Markdown guides/references resolve 48 local links.
- 6×6/chi96 and 8×8/chi128 positive rank-136 mixtures obey all caps, with
  independent plaquette defects at most 7.94e-15. Two refinement sweeps
  improve measured full-sweep relative HS errors from .7732 to .5457 and
  .7226 to .3013; single-run times increase from 2.134 to 2.974 seconds and
  7.461 to 9.857 seconds. These are observations, not a general trajectory
  error guarantee or statistically established speed comparison.
- [Derivations, installed/upstream audit, all method measurements and
  reproduction](../docs/development/notes/2026-11-17-dmt-refinement.md).

## Limits

The exact Schur optimum is conditional on compatible, resolved protected
data and zero residual cutoff. General alternating refinement is local.
One-cut nonincrease does not compare different entire-sweep trajectories.
Cutoffs seed the optimized solve; refinement may reuse rank up to the cap.
String-count minimization does not minimize state-dependent protected ranks.
Large-square evidence here is compression, not a complete large-square
gate layer. Adaptive host decisions remain; no fully JIT, GPU performance,
native symmetry or loopy PEPO claims. Existing fast defaults are unchanged.

## Publication scope

The user subsequently requested commit and push to `develop`. This change
includes the DMT implementation, tests, benchmark, owning API guides, DMT
changelog entry, mathematical skill/catalog routing and both dated DMT
handoffs. Unrelated JAX sampling code/docs/changelog, test-profile edits,
older unrelated history and device-local policy are excluded. Existing
numerical validation above is reused; publication checks include staged
diff/whitespace inspection and confirming the remote branch after push.

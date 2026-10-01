# 2026-10-01 — Review combined metric and gate fixes

- Scope: user requested another review after the metric/gate fixes.
- Branch / baseline: `develop` / `7773d2e`, with the preceding fixes still
  uncommitted. No implementation or test changes in this review.
- Commit status: this review handoff is uncommitted; nothing staged or pushed.

## P2 regression: optional metric chi is rejected

The new `_resolve_metric_chi` unconditionally validates its selected value
through the optimizer's positive-integer cap validator. A per-call mapping
`chi=None` therefore raises before reaching the metric, although low-level
metrics accept this for exact contraction and for existing DMRG boundary
handles. Named optimizer caps remain positive integers; an explicit optional
low-level mapping cap is a distinct existing API behavior.

Reproduced with a NumPy 2×2 product PEPS, each local vector `[2., 0.]`:

- `opt.normalize(method="exact", chi=None)` previously returns the scaled
  old norm, but now raises `TypeError: normalize_chi must be an integer.`
- `opt.estimate_infidelity(state, state.copy(), method="exact", chi=None,
  norm=None, norm_target=None)` previously returns zero, but now raises
  `TypeError: evaluation_chi must be an integer.`

The baseline comparison extracted the two metric methods from committed
`HEAD` with AST, compiled them against the current module globals and bound
them to a separate optimizer instance. No checkout or tracked-file changes,
installed-library edits, or different environment were needed. Current helper
methods remain unchanged for that baseline instance; its stored mappings do
not contain chi, so the baseline's fallback cap evaluation remains valid.
Both real baseline metric contractions succeeded; current public calls fail.

Suggested fix: preserve explicitly selected mapping `chi=None` through metric
resolution, while continuing to validate positive numeric named caps and pair
entries. Cover exact metrics and existing-boundary reuse, and distinguish an
explicit optional cap from an absent option when resolving run records and
delegated defaults. No fix was made in this review-only turn.

## Fresh validation

- Activated `envs/py312` with CUDA hidden and JAX on CPU for probes/checks.
- The cap, exact-target, constructor-override and final-compression regression
  selection passed **14 tests**, 216 deselected, one warning, 2.87 seconds.
  It does not cover the optional-cap reproducer.
- `git diff --check`: passed.
- No fresh full-suite or Ruff run. The preceding 535-pass CPU selection and
  CUDA memory failures remain earlier validation, not new results here.
- Existing working-tree fixes, documentation and review artifacts preserved.

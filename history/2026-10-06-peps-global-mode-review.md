# 2026-10-06 — Review global PEPS mode

- Scope: inspect and test global mode; report correctness and defects.
- Branch / baseline: `develop`, `8f7c896`.
- Commit status: new uncommitted review notes only; existing edits preserved.
  No implementation/test edits, staging, commits, or pushes.

Global defaults are LD_VAR2/1200 evaluations with Torch autodiff through
Quimb MPS contractions. A real NumPy 3x3 default run improved exact
infidelity .12069 → .04920 and preserved unit norm/bond cap. Fresh focused
tests: 193 passed. Exact/MPS/CTMRG small-system gradient probes passed.

Confirmed: Torch driver runs fail because Quimb returns NumPy arrays before
comparison to the Torch target; global candidate normalization is duplicated
and ignores normalize_final=False; NLopt exception recovery extracts the last
vector rather than the best one (controlled failure with real evaluations).
Native U1/U1U1 standalone runs improve but also return NumPy blocks.

See the [detailed report](../docs/development/notes/2026-10-06-peps-global-mode-review.md)
for reproduction fixtures, defaults, lower-priority findings, and limits.
No fixes implemented. No full suite, GPU, broad complex64, or hyper-mode
validation. Review links and diff checks passed.

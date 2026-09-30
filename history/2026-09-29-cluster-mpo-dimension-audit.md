# 2026-09-29 — Verify and explain cluster MPO dimensions

- Scope: check the 1D notebook's MPO dimensions and clarify fixed versus
  compact channels for the requested ITF plus NNN model.
- Baselines: Gaugy Examples `main` at `dbabafa`; Pepsy `develop` at `c1ff0f8`.
- Status: working-tree edits only; nothing staged, committed, or published.
  Existing unrelated changes in both repositories were preserved.

## Changes and findings

- Expanded the [notebook](../../gaugy_examples/pauli_gaugy/cluster_1d.ipynb)
  and [guide](../../gaugy_examples/pauli_gaugy/README.md) explanations. Plot
  labels now say **uncompressed (fixed)** and **SVD-compressed**, and explicitly
  describe stored dimensions rather than minimum ranks.
- Added per-bond storage checks, independent Schmidt-unfolding diagnostics,
  truncation lower bounds, and direct fixed/compressed matrix comparisons.
  Saved verified outputs for the affected cells; other outputs were preserved.
- At p=5, t=0.06, largest stored bonds are 649 versus 35 for OBC and 1737
  versus 56 for PBC. Maximum direct matrix difference over the ten audited
  cases is `2.43e-13` relative Frobenius error. Large fixed dimensions retain
  redundancy; compact dimensions are not guaranteed to be minimal.
- Kept the [corrected NN + NNN model](2026-09-29-cluster-1d-restore-nnn.md),
  package source, and compression settings unchanged.

## Validation and limits

Full notebook execution passed. Six focused fixed-factorization and compression
tests passed, including NumPy and Torch checks. All 106 math expressions passed
strict KaTeX rendering, notebook Ruff passed, and the bond plot was visually
inspected. Detailed measurements, test names, environment, and upstream audit
are in the [dimension audit](../docs/development/notes/2026-09-29-cluster-mpo-dimensions.md).

No full-suite, larger-system, or native Symmray numerical validation was run.
Fixed/compressed agreement concerns `C_p`; finite-cluster error against the full
exponential remains. No unresolved blocker for the requested notebook audit.

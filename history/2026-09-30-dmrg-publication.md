# 2026-09-30 — Publish the DMRG session changes

- User authorized committing and pushing the completed session work.
- Baselines: Pepsy `develop` / `f01f596`; examples `main` / `3b61b7d`.
- This handoff accompanies the authorized Pepsy publication commit.
  The examples commit is `a48c6f6`; final push outcomes are reported in the
  session response and can be verified against the remotes.

The Pepsy change includes MPS DMRG1 removal and strictly one-site DMRG,
Tree's one-node default and legacy DMRG1 warning, scoped regressions,
API/skill guidance, and the earlier GPU audit. Unrelated operator, sampling,
roughening, and example-removal edits were excluded, including their entries
in the shared changelog. Previous dated handoffs describe their original
working-tree state; this publication follows the user's later authorization.

The notebook commit contains the mode/default updates and anchored plot
labels. Pre-existing local CuPy selection and other simulation outputs stay
uncommitted. Published plot previews were regenerated from the tracked
notebook's displayed Torch summary values; their limited displayed precision
is documented in the notebook. The user's working notebook remains unchanged
by this selective staging. JSON/code parsing, point/label correspondence,
visual inspection and staged diff checks passed; no simulation was rerun.

## Validation of the exact staged Pepsy content

Exported the index to an isolated directory under `/tmp`, so unrelated
working-tree implementation changes could not influence these checks.
Used the shared environment, staged source first, one CPU numerical thread,
hidden CUDA and temporary caches.

- MPS domain plus Tree API consistency/parity/composition and public API/layout:
  **1167 passed, 37 skipped, 31 deselected**, 97.62 s.
  Excluded slow tests and the previously confirmed installed-metadata version
  mismatch (0.4.0 versus project 0.5.0). Accelerator/Metal checks skipped.
  Log: `/tmp/pepsy-dmrg-staged-validation.log`.
- Tree default and explicit-block regressions: **13 passed, 192 deselected**,
  3.44 s, including U1/U1U1 and NumPy/Torch CPU cases.
- Native MPS DMRG regression selection: **11 passed, 193 deselected**.
- Ruff `src tests`, 12-skill catalog and staged diff checks pass.

Earlier broad Tree checks and real GPU audit results remain separately scoped
in the session handoffs. No new full-package or GPU execution claim is made
for this publication check.

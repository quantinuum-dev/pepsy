# 2026-09-29 — Final trajectory audit and optimizer publication

- Scope: the user requested commit/push, then a further review of
  `run_trajectory_shots` for MPS/tree and Tree DMRG iteration defaults before
  publication. The commit/push authorization remains active.
- Branch / baseline: `develop`, `b343ced`.
- Commit status at audit: related optimizer changes staged selectively;
  concurrent operator/package changes remain outside this commit.

## Findings and changes

The MPS stabilization repair and direct/DMRG2 trajectory checks pass. Three
ordinary-tree defects were confirmed and repaired: inherited stabilization
altered trial branch norms, finite-chi trial compression biased Born weights,
and rare importance-sampled branches could remain unnormalized below the
default epsilon. See the
[implementation note](../docs/development/notes/2026-09-29-tree-trajectory-born-weights.md)
for concrete before/after examples and the reused upstream audit.

Tree probabilities now use exact local Gram expectations on a private TTN
wrapper, with common stored scales stripped. The selected branch still uses
the user's compression settings; positive tree branches normalize with eps=0.
Public noise documentation, tree trajectory skill guidance and changelog were
updated. The cheap ordinary gate infidelity metric is unchanged.

## Iteration defaults verified from implementation

- TreeOptimizer defaults to `fit_n_iter=4` for each fitted gate window. Each
  iteration comprises two directional passes, so the budget permits eight.
- `fit_min_iter=2`, `fit_patience=1`, and `fit_rtol="auto"`: tolerance is 1e-5
  for float32/complex64 and 1e-9 for higher precision (1e-3 for 16-bit data).
  Warm-up/refinement transitions can defer stopping; a one-node region uses
  one exact update. `fit_rtol=None` disables tolerance stopping.
- Configure `fit_n_iter` in the TreeOptimizer constructor, including inside a
  trajectory factory. It is not a replay `run_kwargs` option. Actual completed
  iterations are in `get_fit_diagnostics()["iterations"]`.
- A deterministic probe observed four iterations for dmrg/dmrg1/dmrg2/dmrg3
  and two for mix. These are example convergence outcomes, not fixed counts.
  Standalone TreeFIT's default n_iter=6 is separate from TreeOptimizer's four.

## Validation

- Before repair, new probability regressions: **12 failed, 4 passed**.
  MPS comparisons already passed; ordinary-tree failures reproduced the two
  Born-weight defects and the zero-outcome stabilization exception.
- Initial repaired trajectory module: **128 passed**. Eight additional rare
  branch regressions then covered MPS/tree, direct/DMRG2 and both strategies.
- CPU trajectory, tree stability/measurement/API, and MPS normalization/control
  selection: **427 passed, 1 skipped**, 50.03 s. The final staged snapshot is
  checked separately before publication, including the branch-center suite.
- Exact local probability backend selection: **36 passed, 156 deselected**,
  9.56 s: Torch CPU/CUDA, JAX CPU, CuPy; one-, two-, three-site supports in
  nontrivial order; stored exponents -400/0/400; dense reference probabilities;
  no live data, proofs, center, RNG or history mutation; no trial replay or
  dense state conversion. Initial complex64 imaginary-roundoff failures were
  resolved in the known-Hermitian readout without relaxing test tolerances.
- JAX CUDA with highest matmul precision: **9 passed, 183 deselected**, 8.31 s.
- Repository Ruff, tree skill validation, catalog and diff checks passed.

Logs: `/tmp/pepsy-trajectory-final-audit-before.log`,
`/tmp/pepsy-trajectory-final-audit-after.log`,
`/tmp/pepsy-trajectory-final-audit-domain.log`,
`/tmp/pepsy-trajectory-final-audit-backends-fixed.log`, and
`/tmp/pepsy-trajectory-final-audit-jax-gpu.log`.

No full-package or full-GPU run was added in this follow-up. Earlier validation
and its limits remain in the linked optimizer repair handoffs. The unrelated
concurrent operator changes are outside these validation/publication claims.

## Final staged-snapshot verification

Archived the selectively staged Git tree into a separate temporary directory,
then set PYTHONPATH to that snapshot and verified the imported package path.
The combined trajectory, tree stability/measurement/API, MPS normalization/
controls, all 100 branch-center cases, and public API/package-layout tests
reported **580 passed, 1 skipped, 4 warnings**, 59.32 s. This verifies that the
optimizer commit works without the unrelated operator edits in the shared
workspace. No numerical files changed after that check. The detailed log is
`/tmp/pepsy-optimizer-staged-validation.log`.

All 26 selected files, including only optimizer entries from the shared
changelog, are prepared for the user-authorized commit to develop and push to
origin/develop. Historical handoff statuses above describe their original
review points; the publication commit and final conversation report identify
the resulting revision.

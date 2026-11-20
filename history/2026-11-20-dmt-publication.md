# 2026-11-20 — Commit and synchronize DMT work

- Scope: user requested committing pending Pepsy changes and pulling/pushing
  the current branch into synchronization.
- Branch / baseline: `develop`, `eaedf789`.
- Publication: this handoff accompanies the commit of the preceding DMT
  region, robustness, Torch interval and lattice Fig. 2 work. Git history
  and remote refs record the final publication state.

Retained the earlier session records and benchmark evidence, including failed
and budget-limited controls. The device-local `AGENTS.override.md` is excluded.
No numerical behavior was changed during publication.

Fresh validation: the nine new test modules plus public API/package layout
produced 142 passes and one known installed-version failure:
`test_package_version_matches_installed_distribution` sees installed metadata
0.5.0 versus project version 0.6.0. Ruff passed for `src`, `tests` and the five
new example scripts; `git diff --check` passed. No full-suite rerun is claimed;
earlier scoped results and baseline limitations remain in the linked handoffs:
[region review](2026-11-20-dmt-region-review.md),
[robustness](2026-11-20-dmt-robustness.md),
[Torch intervals](2026-11-20-dmt-torch-interval.md), and
[Fig. 2](2026-11-20-dmt-lattice-fig2.md).

Remote `develop` advanced during validation. Rebased the new local commit onto
`1e7e3d12` without conflicts. After integration, public interval maps, ordinary
DMT replay, MPO optimization and error-mitigation tests produced 268 passes
and three skips; Ruff `src tests` and whitespace checks passed again.

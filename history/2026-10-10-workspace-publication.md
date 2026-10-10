# 2026-10-10 — Publish accumulated workspace changes

- Scope: user requested committing and pushing Pepsy, Gaugy and their examples.
- Branch / baseline: `develop`, `8d53933`; implementation snapshot `3f60b2f`,
  remote integration `13d58f1`. The merge changes documentation only.
- Preserved the existing full-site/layer PEPS, JAX traced-backend, sum-FIT and
  fidelity-audit work. Earlier dated records describe its implementation.
- Fresh checks: sum FIT, traced MPS, layer/strip refinement, public API and
  package-layout selection: **161 passed, 1 skipped**, 3 warnings, 27.48 s.
  Ruff and whitespace checks passed. No full-suite or new GPU validation claim.
- Downstream FullUpdate checks: **22 passed**. Four existing bubble entrypoint
  failures also reproduce against the original `8d53933` package: obsolete
  `dmrg1` and incompatible one-site `dmrg` block-size settings. These unrelated
  runners were not changed. See `/tmp/publish-bubble-original.log`.
- Commits are prepared for the explicitly requested upstream push; final push
  status is reported after Git confirms it. Simulation runtime records remain
  local in Pepsy Examples, following its repository instructions.

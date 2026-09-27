# 2026-09-26 — Publish cluster API dependencies with remote work preserved

- User requested the completed cluster API refinements be committed/pushed.
- Local `develop` was `cf1d84c`; remote advanced to `a13031b`. The first
  push was safely rejected. Merge both histories without rewriting commits.
- Only conflict: additive Unreleased changelog entries. Retained both the
  local boundary/cluster entries and remote sampler/qMERA/MPS entries.
  No manual implementation edits; remote code is preserved unchanged.
- After merge: projector, flat-backend, public-API, layout and Torch-SVD
  selection **136 passed**, 8 warnings, 1.53 s. Full Ruff passes. Downstream
  Gaugy cluster API/PEPO checks **28 passed**, 11 s. Diff check passes.
  These are focused merge checks, not a claim of rerunning the full suite
  against all newly fetched subsystems.
- This note accompanies the merge commit for publication to origin/develop.
  Device-local policy and generated artifacts are not staged.

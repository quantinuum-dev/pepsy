# 2026-10-01 — TreeSampler factor default

- Scope: user requested making factor the default for the intended large-bond
  workload, continuing the authorized TreeSampler commit/push work.
- Branch / baseline: `develop` at published commit `25e9e18`.
- Commit status: included in the commit titled `Make exact factor sampling
  the TreeSampler default`, continuing the user's commit/push authorization.
  Publication targets `origin/develop` and is checked against the remote ref
  at handoff. Unrelated solver, MPI, PEPS and trajectory edits and their
  handoffs are excluded, including unrelated hunks in the shared changelog.

## Changes and decision

The constructor now defaults to exact factor sampling for dense states.
Explicit `strategy="standard"`, native Symmray sampling, chunk/cache/workspace
budgets and backend/thread defaults retain their previous behavior. Updated
the API guide, source docstring, implementation map and scoped changelog
entries. The existing default-settings matrix now tests the implicit factor
route against explicit standard and dense reference probabilities across
NumPy, Torch CPU/CUDA and CuPy. Comparisons elsewhere retain explicit standard
references. Density/vector contraction-specific tests retain their standard
scope rather than changing structural assertions to match factor.

The [decision and environment record](../docs/development/notes/2026-10-01-tree-factor-default.md)
links the earlier measured tradeoffs. Promotion follows the user's request;
the small-bond memory counterexample remains valid. Old handoffs describing an
opt-in strategy or standard default are historical, not the current API.

## Validation and limits

New focused validation: **512 passed, two skipped**, four existing compatibility
warnings in 101.25 seconds. Covers both sampler strategies, scale/validity,
gradients, seeded draws, native backend/device preservation and Symmray,
canonical regions, entropy, unitary stability, public API and package layout.
The known skips require CuPy subnormal-source arithmetic support and a second
CUDA device; one-device Torch/CuPy checks passed. Ruff, relative documentation
links and whitespace checks pass. Exact commands and environment versions are
in the linked decision record. No new production-checkpoint, peak-memory or
full-suite claim is made.

# 2026-10-06 — Default JAX configuration for global PEPS optimization

- Scope: make the previously verified JAX configuration the default when
  choosing JAX in `PepsOptimizer(mode="global")`.
- Branch / baseline commit: `develop` / `8f7c896`.
- Commit status: working-tree edits only; nothing staged, committed, or
  pushed. Earlier PEPS work and unrelated solver edits preserved.

Global optimizer options now resolve before loss defaults. JAX selects
JIT, cutoff zero, and unstripped loss contractions. Explicit global options
retain precedence, including JIT opt-out. Added orchestration and real JAX
gradient/optimization regressions, API documentation, and changelog entry.

See [implementation and evidence](../docs/development/notes/2026-10-06-peps-global-jax-defaults.md).
This adopts a working configuration; explicit JAX exponent stripping still
has the previously reported gradient/tracing limitation. No standalone
GlobalOptimizer defaults or numerical kernels changed.

Validation: **273 passed** across focused runs (173 PEPS/global/safeguards,
2 JAX integration cases, 98 batching/public API/package layout). An initial
JAX test-fixture compatibility error was corrected before the successful
numerical rerun. Ruff, documentation links, and diff whitespace checks pass.
No full suite or GPU run; detailed scope is in the linked evidence.

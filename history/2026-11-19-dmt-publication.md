# 2026-11-19 — Publish accumulated DMT validation and API work

- Scope: user explicitly requested committing and pushing the accumulated
  DMT/MPS work, skill/API updates and benchmark evidence.
- Branch / baseline: `develop`, `f5d31006d02821028c890a04b17e4b15b9683d0b`.
- Commit status: included in this publication commit; Git records its hash.
  Earlier handoffs' uncommitted status describes those earlier turns.

## Included work

Preservation warning/error policies, cheaper replay verification, one final
strict-sweep region check, indexed crossing-region plans, NumPy exact-gate
and norm-diagnostic fixes, DMT skill/API documentation, focused tests, and
reproducible 1D/paper/larger-window/physical-2D benchmarks. Scientific plots,
numeric results and the extracted figure-data fixture are intentional evidence
linked by the dated notes. The local `AGENTS.override.md` remains ignored and
is excluded from publication. No dependency or environment changes.

See [2D results](../docs/development/notes/2026-11-19-dmt-2d-dynamics.md),
[verification costs](../docs/development/notes/2026-11-19-dmt-check-api.md), and
[larger windows](../docs/development/notes/2026-11-19-dmt-window-scaling.md).

## Pre-publication validation

- Ruff: `src`, `tests`, and all newly added benchmark scripts pass.
- DMT skill validator and 13-skill catalog pass; all 24 existing changed/new
  Markdown files' local targets resolve; whitespace checks pass.
- Fresh targeted selection covering MPS gate precision, DMT check costs,
  preservation policy and 2D dynamics: **54 passed, 1 skipped, 3 failed**.
  All three failures also reproduce using the original `HEAD` compression
  module loaded separately, without the working-tree changes:
  `test_small_gate_factorization_preserves_gradient_and_state_dtype[torch]`,
  `[jax]`, and `[jax-x64]` in `tests/test_mps_gate_precision.py`. Contraction
  selects NumPy for gradient-carrying backend arrays, producing a Torch
  `RuntimeError` or JAX `TracerArrayConversionError`. This audit establishes
  pre-existing failures; it does not change the solver or weaken assertions.
- Earlier scoped results remain documented in their own handoffs: 487 DMT/API
  tests and 161 2D/refinement/replay/lattice tests passed, with their stated
  exclusions/skips. The existing package-version metadata mismatch
  (installed 0.5.0, project 0.6.0) remains. No clean full-suite claim.

Fetched `origin/develop` and confirmed it matched the local baseline before
publication. No unrelated work was staged; benchmark findings retain the
positivity, Hermiticity, trace and finite-size limitations reported earlier.

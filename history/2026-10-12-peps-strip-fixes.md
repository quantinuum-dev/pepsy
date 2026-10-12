# PEPS strip sweep review fixes

- Scope: fix the two issues in the [strip review](2026-10-12-peps-strip-review.md).
- Branch / baseline: `develop`, `192561a`, ahead of origin by two commits.
- Status: working-tree edits only; nothing staged, committed, or published.

Strip dispatch now uses the existing `_run_global_sweeps` lifecycle. The
private runner accepts a first-cycle strip index and can return fitted
boundary guesses before rebuilding environments for the retained best state.
This preserves the gated-strip traversal and inter-batch guess reuse while
sharing failure reporting, progress, normalization, diagnostics, and cycles.

Failed sweeps return no optimizer infidelity, including when an explicitly
requested final diagnostic is nonfinite. The driver can therefore retain
the warm start. Explicit conflicting axes are rejected; row/column ordering
styles still permit both gate orientations. Requested per-visit boundary
norms are retained in `sweep_history`. Honoring `renormalize=True` exposed and
fixed scalar logging of scaled `(mantissa, exponent)` normalization results.

New numerical regressions cover all-invalid visits with monitoring,
acceptance, and final-diagnostic switches; normalization and progress;
multiple cycles; and mixed gate orientations. Existing gradient-specific
tests now explicitly select NLopt, its initialization policy, and relevant
caps. Their behavioral assertions remain intact.

## Validation

The same isolated interpreter and dependency versions as the review were
used. Broad regression: **676 passed**, covering strip batches, sweep solvers,
performance/speed/safeguards, the PEPS driver, batching, boundary convergence,
full update, and public/package APIs. The command was:

```text
python -m pytest -q -o addopts='' tests/test_peps_sweep_defaults.py \
  tests/test_peps_strip_batches.py tests/test_peps_sweep_*.py \
  tests/test_optimize_peps.py tests/test_peps_optimizer_batching.py \
  tests/test_peps_boundary_convergence.py tests/test_peps_full_update.py \
  tests/test_public_api.py tests/test_package_layout.py
```

After the last mixed-orientation and failed-final-diagnostic refinements,
the defaults/safeguards selection passed **63 tests**. Ruff on `src tests`
and `git diff --check` passed. These overlapping domain selections are not
a full extended-suite run.

The upstream maintenance check consulted the official Quimb changelog,
Autoray repository, Cotengra documentation/changelog, and Symmray repository.
The Symmray array documentation URL was unavailable. Classification: **defer**
upstream changes; these fixes use existing local orchestration and dispatch
without changing dependency behavior or versions.

Long-run performance remains unmeasured.

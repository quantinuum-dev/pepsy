# 2026-10-06 — Fix the PEPS review findings

- Scope: user requested fixing the issues identified in the follow-up review.
- Branch / baseline: `develop`, `8f7c896`.
- Commit status: working-tree implementation/test/doc edits; nothing committed
  or pushed. Concurrent solver edits were preserved.

## Implemented

Delegated sweep diagnostics reuse the selected target norm, already-normalized
warm starts skip duplicate constructor normalization, and unknown objective
norms are measured once when optimization is enabled without fidelity checks.
Explicit diagnostic/normalization overrides remain. Standalone SweepOptimizer
behavior is unchanged. Public behavior is documented in the
[API guide](../docs/api/optimizers/peps.md).

The [evidence note](../docs/development/notes/2026-10-06-peps-review-fixes.md)
details implementation choices, regressions, and development failures. These
fixes address the three actionable findings in the
[review](2026-10-06-peps-followup-review.md). Finite-chi normalization accuracy
remains a numerical limitation; invalid fidelity checks are retained.

## Fresh validation

Activated Python 3.12 environment; BLAS/OpenMP threads set to one.

```text
python -m pytest -q -ra -o addopts='' \
  tests/test_peps_optimizer_batching.py tests/test_optimize_peps.py \
  tests/test_optimize_global.py tests/test_prepare_boundary_inputs.py \
  tests/test_public_api.py tests/test_package_layout.py
```

**429 passed**, 51 warnings, no skips, 27.21 seconds. This supersedes the
earlier 424-test selection for the current implementation. Ruff, local
documentation links and `git diff --check` passed.

Coverage includes real sweep/global cleanup, all driver/delegated target
contraction calls, native Torch U1 fermionic unknown norms, scalar/paired
caps, override preservation, dense reference fidelity and output norms.
No full package suite, GPU run, or broad PEPO/cyclic validation.

# PEPS strip sweep review

- Scope: review the claimed monitoring, moving-boundary reuse, ALS caching,
  and default sweep policy. No implementation changes.
- Branch / baseline: `develop`, `192561a`, ahead of origin by two commits.
- Status: existing working-tree changes preserved; this review is uncommitted.

The default traversal and cache checks passed, but the strip dispatch has two
unresolved regressions in `_optimize_with_sweep`:

- If every local loss is invalid, `best_loss` remains infinity. The new result
  wrapper forwards it to `_clean_optimizer_infidelity`, which raises
  `ValueError: PEPS infidelity must be finite.` before the driver can retain
  the warm start. The existing all-skipped safeguard test exposes this;
  an independent probe forcing fidelity 1.25 reproduces the exception.
- Direct `optimize_axis` dispatch bypasses stored run controls. A probe setting
  `renormalize=True`, `normalize_boundaries=True`, and
  `collect_boundary_norms=True` observed `renormalize=False` and omission of
  the other two controls. It also bypasses the run-level progress callback.
  Explicit options must be forwarded, or unsupported options rejected.

Fresh validation used the existing isolated environment documented in the
[implementation handoff](2026-10-12-peps-strip-global-alignment.md):

```text
python -m pytest -q -o addopts='' tests/test_peps_sweep_defaults.py \
  tests/test_peps_strip_batches.py tests/test_peps_sweep_als.py \
  tests/test_peps_sweep_performance.py tests/test_peps_sweep_safeguards.py
170 passed, 9 failed, 2 skipped
```

The failures were three `test_precheck_reuse_requires_matching_metric_policy`
cases (legacy `maxeval` options now reach ALS), the all-skipped driver
safeguard (exception above), `test_clipped_outer_precheck_still_invokes_sweep`
(missing reused initial-loss field), and four
`test_real_coarse_boundary_failure_returns_warm_start` cases (old warning
expectation). Default-policy assumptions in these tests need review separately
from the demonstrated all-invalid implementation failure. No full-suite pass
is claimed. Ruff on `src tests` and `git diff --check` passed.

Installed versions: Quimb `1.15.1.dev91+gb3bf45a50`, Autoray
`0.11.1.dev14+g014a3f69a`, Cotengra `0.8.3.dev8+g8954240f2`, Symmray
`0.4.1.dev18+g44fc5ae53`. No dependencies changed. Long-run performance remains
unmeasured.

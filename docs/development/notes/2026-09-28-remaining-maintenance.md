# 2026-09-28 — Boundary diagnostics and numerical orchestration

Baseline: `develop` at `6750b19`, which commits the
[complex-cast corrections](2026-09-28-complex-cast-review.md).
This follow-up covers the user's remaining diagnostics, readability, warning,
hardware, and publication tasks. It adds no dependencies or test tiers.

## Implemented

- Boundary VMC separates compiled dispatch, cached parent/window construction,
  and eager evaluation. Cache writes and alternative retries remain on the
  caller thread; worker threads explicitly inherit no-grad measurement mode.
  Group order, repeated target rows, diagonal reuse, final forward batching,
  device/dtype conversion, and parameter-version invalidation are retained.
- Recovered connected and proposal exceptions record their stage and error
  string. Each call counts all exceptions and retains at most eight records,
  each with at most 400 error characters. Records contain no exception,
  traceback, or tensor objects. Sparse cutoff retry retains its latest cause.
  Existing numerical recovery policies and final-error propagation remain.
- BP extracts the unexcited scalar norm/gate contraction. Numerical cache
  keys, graded projector selection, cost rejection, contraction order, and
  normalization remain with their existing policies. A required baseline
  still raises if its contraction exceeds the budget.
- VMC multi-observable result assembly is shared by modern and legacy sampling;
  statistics, chain axes, profile copying, and ESS stopping are unchanged.
- Generic dense PEPO fitting has a separate residual-fit helper. Rank order,
  deterministic seeds, warm starts, convergence criterion, and complete
  lower-order subtraction remain unchanged.

Source spans (including docstrings) decreased as follows. These are navigation
metrics, not speed or numerical-accuracy measurements:

| Function | Before | After |
| --- | ---: | ---: |
| Boundary `connected_amplitudes` | 459 | 180 |
| BP `_contract_open_scalar_edges` | 396 | 317 |
| VMC `estimate_observables` | 412 | 363 |
| PEPO `_add_generic_cluster_levels` | 327 | 276 |

## Upstream warning review

The environment and capability audit from the
[cast review](2026-09-28-complex-cast-review.md#upstream-audit) is unchanged.
The refactors move Pepsy orchestration around the same upstream calls.
Classification: **adopt** local diagnostics and clearer ownership;
**defer** upstream warning fixes. No installed libraries were modified.

The NumPy 2.5 warning reproduces without importing Pepsy:

```python
import numpy as np
from quimb.tensor.optimize import Vectorizer

vectorizer = Vectorizer([np.arange(6.0).reshape(2, 3)])
arrays = vectorizer.unpack()
```

With warnings enabled, the installed Quimb reports its `array.shape =
info.shape` assignment at `quimb/tensor/optimize.py:174`. Reconstructed values
match the input. The reviewed
[upstream implementation](https://github.com/jcmgray/quimb/blob/main/quimb/tensor/optimize.py)
still uses that assignment. An upstream reshape change needs upstream review;
there is no Pepsy-owned replacement of this optimizer operation in this pass.

The remaining once-per-process Torch scalar warning was previously localized
to Cotengra's zero-factor check in
[`contract.py`](https://github.com/jcmgray/cotengra/blob/main/cotengra/contract.py).
It is separate from the resolved complex-to-real casts. No warning filters,
dependency downgrades, or contraction substitutions were introduced.

The installed Loky warning occurs when a worker exits while work remains and
the executor replenishes its worker pool (`process_executor.py:787`). This
explains the warning condition, but does not distinguish timeout from memory
pressure in the earlier run. The previously affected exact-batch test passed
in isolation. The full-suite follow-up below monitors recurrence without
altering worker policies.

## Hardware limits

A fresh host probe, outside sandbox device restrictions, reports Torch 2.9.1,
CUDA unavailable, CuPy absent, and Metal available. Native Metal QR raises
`NotImplementedError` for `aten::linalg_qr.out`. CUDA/CuPy numerical paths
therefore remain unvalidated here; CPU fallback is not GPU validation.
Earlier MPI and two-logical-CPU JAX evidence is retained in the
[maintenance triage](2026-09-28-maintenance-triage.md), not counted as a new run.

## Validation

- BP open series, native Symmray BP, and dense cluster suites: **139 passed,
  1 warning in 20.89s**. This includes adaptive loop fitting and rank growth.
- VMC local energy, API, compilation, and convergence: **64 passed, 1 warning
  in 16.27s** before the final cutoff-diagnostic addition.
- All local-energy diagnostic tests after that addition: **14 passed in 0.96s**.
  Injected failures cover serial/threaded primary windows, context creation,
  alternative windows, compiled reuse, raw/log proposals, bounded storage,
  call reset, and sparse-cutoff retry options.
- An isolated before/after comparison executes the committed boundary method
  and the refactored method on the same seeded 2×2 PEPS. Float64 and complex128
  target values and every parameter gradient agree at `atol=rtol=1e-12`.
  Inputs include diagonal and repeated targets and two parents. Grad-enabled
  execution correctly leaves the worker-thread count at zero.
- Ruff and the two focused CI mypy targets pass. Scoped Pylint design checks
  still report large functions and local-variable counts; this is gradual
  simplification, not a clean whole-package Pylint claim.
- Full-suite, smoke, and publication results are recorded in the
  [session handoff](../../../history/2026-09-28-remaining-maintenance.md).
- Final full suite: **5,185 passed, 129 skipped, 685 warnings in 512.58s**.
  There are no complex-to-real warnings. Compared with the preceding run,
  Quimb shape warnings varied from 601 to 605 and Loky worker notices from
  two to one; all other warning counts are unchanged. The worker notice
  again accompanies the passing NumPy exact-batch phase-pass test. Its
  cause remains unconfirmed, and no warning was suppressed.
- Default smoke: **92 passed, 2 compatibility warnings in 18.47s**.
  Ruff, focused mypy, 28 local Markdown link targets, and `git diff --check`
  pass. The new diagnostics tests belong to the extended domain suite.

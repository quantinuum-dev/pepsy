# 2026-10-06 — Global PEPS ownership and recovery fixes

User authorized fixes following the
[global-mode review](2026-10-06-peps-global-mode-review.md). Branch `develop`,
baseline `8f7c896`. These changes remain in the working tree; earlier PEPS
work and concurrent gradient-solver changes were preserved.

## Implemented

1. Both GlobalOptimizer entry points restore each output tensor's backend,
   dtype, and device from the corresponding input tensor before normalization.
   They reuse `infer_backend_converter_from_sample`, including its native
   Symmray block conversion; no independent conversion implementation was
   added. This fixes the real Torch candidate/target mismatch in the driver.
2. Both entry points accept `normalize=True` (standalone default). The PEPS
   driver explicitly disables delegated normalization, including fallback,
   and owns candidate normalization through its existing `normalize_final`,
   `normalize_kwargs`, and `normalize_chi` controls. Independently configured
   standalone `normalize_kwargs` are now honored by `normalize()` as well.
3. Global NLopt retains the best finite evaluated vector and restores it
   after normal completion or a caught NLopt exception. Evaluation history
   is preserved unchanged; `final_loss` identifies the restored vector's
   score before optional normalization. The driver uses this score rather
   than the final trial's history entry. Recovery without any valid iterate
   retains the input and reports failure; the driver then retains its warm
   start, even if acceptance checking was disabled.
4. `optimization_info`, propagated into the PEPS step's `optimizer_result`,
   records recovery, termination, returned loss, and exception details.
   Recovered valid candidates still pass through ordinary outer acceptance.

Algorithm and iteration defaults are unchanged: global LD_VAR2/1200;
sweep LD_LBFGS/50. No extra objective or target-norm contractions were added
for best-iterate tracking. Existing fidelity objective semantics, finite-cap
approximations, and known unitary target-norm policy were preserved.

## Upstream compatibility evidence

Reused the unchanged installed-environment/upstream audit. Inspected the
installed Quimb `TNOptimizer` signature, `optimize_nlopt`,
`vectorized_value_and_grad`, and `get_tn_opt` during this task. Quimb's NLopt
route does not invoke its callback; it appends each evaluated scalar loss
while its vectorizer holds the corresponding trial. `get_tn_opt` extracts
the current vector and converts output arrays to NumPy.

Classification: **adopt** the existing backend converter;
**compatibility shim** for best-vector recovery. The shim is scoped to the
per-run loss-history instance, capability-gated on a NumPy vectorizer and
the extraction method, and tested through actual objective evaluations.
It copies a vector only on improvement, does not wrap traced loss/gradient
functions, and does not modify installed code. It depends on Quimb's loss
append/current-vector correspondence; re-audit that contract on upgrades.
If tracking is unavailable during an exception, retain the input rather
than claim a best iterate was recovered.

## Validation

Added 17 regressions in `tests/test_global_optimizer_safeguards.py`:

- Two real global RZZ batches on 3x3 PEPSs, both NumPy and Torch, with final
  normalization enabled/disabled; verify input ownership, output type/device,
  improvement, retained rank, exactly one candidate normalization when
  requested, and no delegated normalization.
- Both optimization entry points preserve Torch complex64/complex128 dtype.
- Controlled worse-trial completion and NLopt failure restore the best
  evaluated state while retaining raw history. The review reproduction now
  returns loss 6.66e-16 instead of the last trial's .987431843617645.
- Failure before a valid evaluation and a finite score paired with NaN
  parameters preserve the input and report failure.
- Driver diagnostics and returned score remain correct with final measurement
  disabled, both with and without a valid recoverable iterate.
- Independent standalone normalization options are honored.
- Real U1/U1U1 fermionic optimization retains Torch blocks, dtype, index and
  charge-block metadata, with finite improved results.

Fresh broad focused run: **479 passed**, 1576 warnings, 64.74 seconds, across
the new regressions, global optimizer, PEPS sweep safeguards, boundary-engine
numerics, PEPS optimizer/batching, boundary preparation, fermionic boundary,
public API, and package layout suites. No skipped cases in this selection.
Warnings include existing NumPy shape deprecations, compatibility aliases,
and rank-deficient Torch QR diagnostics; no test failures.

Earlier incremental checks in this implementation turn: 193 existing tests,
15 initial regressions, and two additional driver regressions passed before
the final combined run. Do not add these duplicate counts to 479.

Ruff (`src tests`), local documentation links, and `git diff --check` passed.
No full repository suite or GPU run; device restoration uses the existing
converter and CPU device preservation is exercised. No universal convergence
or finite-boundary accuracy guarantee is implied.

Updated API docs and changelog. Nothing staged, committed, or pushed.

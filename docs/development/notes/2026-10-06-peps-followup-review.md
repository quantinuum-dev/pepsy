# 2026-10-06 — Review after batching and normalization changes

Scope: explain and review the current uncommitted PepsOptimizer implementation
on `develop`, baseline `8f7c896`. This review makes no implementation changes.
It corrects the scope of the earlier
[known-target-norm validation](2026-10-06-peps-known-target-norm.md): those tests
observed only the outer driver's metric calls, not the delegated sweep calls.

## Findings

### 1. Target norm avoidance is incomplete in the default sweep path

`SweepOptimizer._approx_infidelity_loss` calls `self.infidelity` without
passing `self.target_norm`. `infidelity` defaults `norm_target=None`, causing
a target norm contraction. This happens in the sweep's initial and final
diagnostics, even though the outer PepsOptimizer checks supply target norm
one and local slice objectives use the stored known norm.

A real 2x2 product-state probe with a random two-qubit unitary, chi=1 and
the default sweep schedule returned the following actual metric calls:

| Caller | Supplied target norm | Target norm contracted |
| --- | --- | --- |
| PepsOptimizer precheck | 1 | No |
| SweepOptimizer initial diagnostic | None | Yes |
| SweepOptimizer final diagnostic | None | Yes |
| PepsOptimizer postcheck | 1 | No |

Thus the requested elimination of target norm contractions is **not fully
implemented**. Besides extra cost, local objectives and sweep diagnostics can
use different target normalization estimates. The previous claim that the
whole optimization avoids these contractions was too broad. Proposed fix:
propagate the selected known norm through internal sweep diagnostics while
preserving explicit standalone metric semantics; test all contraction callers.

### 2. Disabling measurements can leave a nonunitary local objective misnormalized

With `non_unitary=True`, `normalize_target=False`, and
`measure_infidelity=False`, the driver forwards target norm None. The sweep's
`set_target_norm` converts None to one. Actual nonunitary targets need not
have norm one, so their local loss is incorrect.

Reproduction: 2x2 product input, chi=1, a gate equal to twice the unitary used
above. With measurement enabled there are zero invalid local updates. With
measurement disabled all **20** scheduled slice updates are skipped with
invalid local infidelity approximately **-2.667**. Sweep diagnostics still
measure target norms and return a plausible loss (~0.08324766), concealing
the lack of local refinement in a superficially successful result.

This is an optional-path bug, not the user's default unitary path. Proposed
fix: resolve unknown target norms independently of diagnostic measurement,
or require an explicit known norm for this combination. Unit-norm unitary
runs should continue to avoid target contraction.

### 3. The warm start is normalized twice before refinement

The driver normalizes the compressed warm start, then passes
`renormalize_state=True` to SweepOptimizer. Its constructor calls normalize
on the same state object. Instrumented calls with zero sweep cycles were:
driver initial state, driver warm start, sweep same warm start, driver final
candidate. The repeated warm normalization is an avoidable contraction,
not a demonstrated state-correctness failure. Proposed fix: skip delegated
initial normalization for the already-normalized warm start, preserving
explicit overrides and standalone SweepOptimizer behavior.

### 4. Known unit target norm remains an accuracy assumption at finite chi

This is an intentional cost/accuracy tradeoff, not a new implementation bug.
Exact unitary gates preserve exact unit norm. Finite-boundary normalization
only approximates that premise. The existing 4x4 D=2 identity regression
still raises at default caps with an estimated negative error of ~-4.638e-4;
normalization/evaluation caps 32 resolve the dense-reference check. Increasing
only evaluation accuracy need not repair inaccurate incoming normalization.
The unitary default does not verify gates' unitarity. Nonunitary inputs must
be declared, and caller-managed initial normalization must actually be valid.

## Behavior that matches the design

Initial normalization; ordered gate absorption; shared-site auto stopping;
2D dense diagonal/RZZ and 4D general target budgets; exact target construction
with singleton overflow; compression of a target copy to D; normalized warm
starts; tolerance-based refinement entry; local NLopt LD_LBFGS with maxeval=50;
final normalization and pre/post acceptance with warm-start restoration are
implemented and covered by the existing focused checks.

Budgets are not padding requests or hard peak-memory bounds. Native/traced/
trainable gates conservatively use the 4D path; only eligible nearest-neighbor
dense diagonal gates receive the exact 2*current_bond rank ceiling. Stored
fidelity traces are batchwise proxies, not exact full-circuit fidelity.

## Validation and reproducibility

Fresh read-only checks: **155 passed**, 4 warnings, 19.51 seconds across
`tests/test_peps_optimizer_batching.py` and `tests/test_optimize_peps.py`.
The preceding 418-test selection is earlier evidence, not a fresh run here.
The new findings above were established by additional real-contraction probes
and call-path inspection; passing the existing tests does not cover them.

Probes used NumPy complex128 2x2 product state |0000>, `contraction_opt='greedy'`,
chi=1, and a unitary generated by QR of a 4x4 complex Gaussian matrix from
`np.random.default_rng(31)`. Metric instrumentation wrapped both
`pepsy.optimizers.peps.optimizer.boundary_infidelity` and
`pepsy.optimizers.sweep.optimizer.boundary_infidelity`, inspecting actual
`norm_target_result` values. Normalization instrumentation wrapped the driver's
`boundary_normalize` and sweep's `peps_normalize`, comparing object identities.
No numerical function was replaced by a fabricated return value.

Reused the unchanged installed-environment and upstream audit from the
[original review](2026-10-06-peps-optimizer-default-review.md#upstream-audit).
Proposed corrections are **deferred** in this review. No dependencies, source
code, tests, or concurrent solver edits were changed. No full suite, GPU run,
or broad PEPO/cyclic validation. Nothing committed or pushed.

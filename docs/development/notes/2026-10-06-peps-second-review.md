# 2026-10-06 — Review after delegated norm fixes

Scope: review the current uncommitted implementation on `develop`, baseline
`8f7c896`. No implementation or test edits in this review. The preceding
[three fixes](2026-10-06-peps-review-fixes.md) remain verified; the findings
below concern additional option paths and coarse-environment handling.

## Confirmed remaining findings

### Global mode normalizes the returned candidate twice and ignores the opt-out

`PepsOptimizer._optimize_with_global` always populates `normalize_kwargs`,
including when its input mapping is empty. Both GlobalOptimizer optimization
entry points then normalize their output internally. The PEPS driver applies
another normalization if `normalize_final=True`. With `normalize_final=False`,
the delegated normalization still happens.

A real 2x2 chi=1 probe with two global objective evaluations and
`accept_if_improved=False` recorded:

- `normalize_final=True`: initial input, warm start, global candidate, same
  candidate again in the driver.
- `normalize_final=False`: initial input, warm start, global candidate.

This is an optional-global-mode flag bug and avoidable contraction cost.
It does not invalidate the previous fix to duplicate *sweep warm-start*
normalization. Proposed correction: let the PEPS driver own final output
normalization and preserve explicit delegated overrides deliberately.

### A coarse sweep score can abort before the accurate acceptance check

`SweepOptimizer._run_global_sweeps` treats any initial loss below 1e-10 as an
early exit, including substantially negative values. Its best-state recorder
rejects that negative value, so the result has `best_loss=None` and a negative
`loss_after`. The PEPS wrapper's `_clean_infidelity` raises before the outer
postcheck or warm-start rejection policy can run.

Reproduction uses a 3x3 complex128 random PEPS, D=2, seed=1, an RZZ matrix
`diag(exp(-0.01j * [1,-1,-1,1]))` on ((1,1),(1,2)), `boundary_chi=1`, greedy
contraction paths, and exact initial normalization/outer fidelity methods.
The valid outer precheck reports **7.430547819620159e-5**. The coarse initial
sweep diagnostic reports **-0.03043056470497363** and the driver raises
`PEPS infidelity is substantially negative.` Only the precheck completes.
Reproduced with both zero cycles and the default sweep schedule, since the
failure precedes any local updates.

This is a robustness limitation, not evidence of a wrong accepted state.
Exact outer checks do not fix an inaccurate optimization environment.
Increasing `boundary_chi` is the appropriate accuracy control. Proposed
correction: distinguish invalid internal scores from convergence and return
the saved warm start / an explicit failed-cleanup status, or use a deliberate
environment retry policy. Do not silently clip the negative score.

### Stored retry overrides leak into the boundary metric

`PepsOptimizer(..., infidelity_kwargs={'evaluation_max_retries': 0})` followed
by `estimate_infidelity(state, state)` raises
`TypeError: peps_infidelity() got an unexpected keyword argument 'evaluation_max_retries'`.
The wrapper merges stored options into the boundary call without consuming
this wrapper-only option. Per-call keyword binding handles it differently.

This is a lower-priority configuration inconsistency. The supported direct
constructor option `evaluation_max_retries=0` remains the workaround. Proposed
correction: either consume wrapper options consistently before forwarding or
reject them at configuration time with guidance to the direct option.

## Validation and limits

Fresh selection: **166 passed**, 36 warnings, 23.02 seconds across
`tests/test_peps_optimizer_batching.py` and `tests/test_optimize_peps.py`.
These include the delegated-contraction guard, native unknown target norms,
scalar/paired caps, real sweep/global cleanup and dense reconstruction checks.
The preceding 429-test selection remains earlier evidence, not a fresh run.

Additional probes used real contraction and optimization functions with
recording wrappers; no fabricated numerical return values. The global fixture
is the 2x2 complex product state and seeded QR unitary described in the
[earlier review](2026-10-06-peps-followup-review.md). The coarse-environment
fixture is specified above. Reused the unchanged installed-environment and
upstream audit; no dependency changes or new compatibility shims.

No full suite, GPU check, or broad PEPO/cyclic validation. Fixes above are
**proposed**, not implemented in this review. Only review notes were added;
concurrent solver edits are untouched. Nothing committed or pushed.

# 2026-10-06 — Skip unitary target norm contractions

User requested removing the enlarged target's norm measurement when evolving
normalized retained states unitarily. This supersedes the default target
measurement in the [preceding normalization note](2026-10-06-peps-unitary-targets.md).

## Implemented policy

`PepsOptimizer.run()` supplies `norm_target=1.0` to both pre/post fidelity
checks for unitary runs with final normalization enabled. The same norm is
forwarded to local/global cleanup. The boundary metric's existing known-norm
path avoids constructing and contracting the target double layer. Retained
output normalization with `normalize_chi` and candidate norm measurement are
unchanged; so are batching and `LD_LBFGS` / `maxeval=50` defaults.

Constructor/run `infidelity_kwargs` overrides retain precedence. An explicit
`norm_target=None` measures the norm, and a numeric/scaled value supplies a
known norm. Nonunitary runs and `normalize_final=False` default to measurement.
Direct `estimate_infidelity` calls on arbitrary states retain measured norms.
Disabling initial normalization makes the caller responsible for supplying
a unit-norm initial state if using the unitary shortcut.

## Accuracy evidence

This intentionally changes the accuracy/cost tradeoff. Finite-boundary
normalization does not establish mathematically exact unit norm. The existing
4x4 D=2 seed-17 identity fixture produces an infidelity of approximately
`-4.638e-4` with default caps `(8, 10)` and assumed target norm one. It now
raises with guidance to increase `normalize_chi` / evaluation accuracy or
explicitly enable target measurement. It does not silently contract the
target, increase caps with an inconsistent fixed norm, or clip this error.

With normalization and evaluation caps 32, the same fixture agrees with
the normalized dense input to absolute tolerance 2e-12 and has zero
infidelity within 1e-12. Explicit target-norm measurement still passes the
earlier automatic-retry regression. That regression is now named to identify
its opt-in measurement policy, and the default low-cap error has its own test.

## Validation

Reused the unchanged environment/upstream audit from the
[original review](2026-10-06-peps-optimizer-default-review.md#upstream-audit).
Inspected `boundary.metrics.peps_infidelity`: its `norm_target is None` branch
owns target norm contraction; supplying a value bypasses that branch.
Classification: **adopt** the existing public known-norm capability, without
upstream shims or dependency changes.

Fresh focused selection (activated Python 3.12 environment, single BLAS/OpenMP
thread): batching, PEPS/global optimizer, boundary input, public API and
package-layout tests. The final result is recorded in the linked handoff.
New real-contraction tests assert absence of `norm_target_result` in both
pre/post checks while retaining candidate norm results, and cover constructor
and per-run overrides, nonunitary evolution and disabled final normalization.

An initial run exposed two cap-precedence regressions from merging stored
options into per-call options; resolved by injecting only the target-norm
policy and keeping existing cap resolution unchanged. The common-cap retry
test now explicitly requests measured target norms, preserving its purpose.
No full suite or GPU testing. Concurrent gradient-solver edits are unchanged.

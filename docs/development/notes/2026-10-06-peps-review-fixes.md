# 2026-10-06 — Fix delegated norm handling after review

Implements the three actionable findings in the
[follow-up review](2026-10-06-peps-followup-review.md). All implementation
changes remain in `pepsy.optimizers.peps.optimizer`; standalone SweepOptimizer
and boundary metric implementations are unchanged.

## Changes

1. The PEPS driver supplies its resolved target norm through the sweep's
   existing `debug_loss_kwargs` option. With normal debugging disabled, this
   routes start/end diagnostics through the boundary-infidelity diagnostic
   with the same norm as the local objective. Default unitary runs no longer
   construct or contract a target norm anywhere in the delegated sweep.
   Explicit diagnostic norm overrides and exact debug metrics retain their
   existing semantics.
2. Delegated constructor normalization now defaults to false, because the
   driver has already normalized the warm start. Explicit
   `sweep_kwargs['renormalize_state']=True` still enables it, with existing
   normalization-cap overrides preserved.
3. Before optimization, an unresolved target norm is measured once through
   public `peps_norm`, independently of optional fidelity checks. This fixes
   nonunitary runs with `normalize_target=False, measure_infidelity=False`
   for both sweep and global cleanup. The resolved evaluation norm cap is
   used (the first entry of a cap pair); overlap-only kwargs are excluded,
   target boundary handles and metric backend policy are preserved, and
   structured timing results are retained. Known norms avoid measurement.
   Skipping optimization as well requires no objective norm.

Batching, exact-target protection, retained-output normalization, and local
`LD_LBFGS` / `maxeval=50` defaults are unchanged. Finite-boundary normalization
still approximates exact unit norm; the explicit negative-metric checks and
the low-cap identity regression remain.

## Validation

The final focused result is recorded in the
[handoff](../../../history/2026-10-06-peps-review-fixes.md).
The test selection covers batching, PEPS/global optimizers, boundary inputs,
public API and package layout. New regressions:

- Instrument the shared double-layer contraction function, below both driver
  and sweep imports, and reject every target-norm contraction in a complete
  default unitary sweep. All four outer/internal overlap checks run.
- Verify exactly three driver normalizations (initial input, warm start,
  retained candidate) and no delegated constructor normalization.
- Verify the explicit constructor-normalization override and its cap.
- Run real sweep/global cleanup on a target with norm four, with fidelity
  checks disabled. Unknown norms are measured exactly once; supplied scaled
  norms are not measured. Both scalar and paired evaluation caps are covered.
  Sweep updates have zero invalid local losses and reproduce the dense
  Schmidt-rank-one optimum to 1e-12.
- Exercise the unknown-norm path with native Torch U1FermionicArray storage,
  checking norm four for the target, unit output norm and preserved storage.

Development failures were resolved explicitly. Five earlier tests assumed
constructor normalization was always enabled; they now request it explicitly
and retain their cap assertions. The first native fixture fit directly in
chi and correctly skipped optimization, so chi was reduced to exercise the
unknown-objective-norm path. That probe exposed a scalar-versus-pair mismatch
in the norm-only helper, which now selects the norm entry and has dedicated
paired-cap coverage. Numerical tolerances were not relaxed.

## Environment and scope

Reused the unchanged environment/upstream audit recorded in the
[original review](2026-10-06-peps-optimizer-default-review.md#upstream-audit).
Inspected installed `peps_norm` and `peps_infidelity` signatures and the
norm-only API's scalar-cap validation. Classification: **adopt** existing
public norm and sweep diagnostic options; no compatibility shim, dependency
upgrade, or installed-library edit.

No full package suite, GPU validation or broad PEPO/cyclic validation. All
changes are uncommitted. Concurrent gradient-solver edits were preserved.

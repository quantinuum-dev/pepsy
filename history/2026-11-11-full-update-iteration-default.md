# 2026-11-11 — Full-update pair iteration default

On `develop` at `1d0a894`, change the `PepsOptimizer` full-update default from
50 to 20 maximum pair-solver iterations. Explicit `max_iterations` continues
to override the default; boundary-MPS sweep counts and loop-cluster controls
are separate. Update the public API, changelog and cluster status ledger.

The sibling Eli runner now defaults to 20 instead of its explicit 10, including
its resolution queue and profiler. Historical plans must explicitly retain
their original caps for reproduction or resume; saved runs are unchanged.

Validation: 9 focused Pepsy tests passed, covering default/override,
reduced-pair reconstruction and cached column updates. The sibling runner's
18 FU preflight/ordering tests and 15 resolution-queue tests passed. Ruff and
diff checks passed in both repositories. The full suite was not run; this
change does not establish production convergence or performance at the new cap.

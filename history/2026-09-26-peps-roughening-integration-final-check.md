# 2026-09-26 — Final roughening sampler integration check

- Scope: user requested correct PepsSampler integration into run_roughening.
- Branch / baseline: Pepsy develop / a13031b; examples main / 301159a.
- Commit status: existing integration remains uncommitted; no staging or publishing.

## Findings

Checked the existing integration against the current sampler implementation.
No further production-code changes were needed. Selected-time scheduling,
private gauge-absorbed snapshots, fresh numerical environments, factored row
caching, auto-hq row planning, reusable Pepsy structural planning, chunking,
coordinate mapping, and importance-weight output are already wired through
the compatibility CLI. See the earlier
[implementation handoff](2026-09-26-peps-roughening-sampling-integration.md).

## New validation

- PEPS, roughening sweep, entrypoint, and runner-layout tests: 61 passed in
  51.43 seconds, using local Pepsy and the shared Python environment on CPU.
- Actual compatibility CLI: NumPy complex128, 2x3, D=4, boundary chi=16,
  dt=.2, depth=2, 8 shots/chunk=3, selected times 0,.15,.4, default planners.
  Output: /tmp/pepsy_examples_runs/peps_sampling_final_integration_20260926.
- Reopened all three NPZ snapshots without pickle: shapes, coordinates, time
  labels, finite log weights, weight formula, normalized sums, planner metadata,
  and complete manifest passed. An initial inspection print requested a
  nonexistent diagnostics key; corrected the probe without changing code.
- Changed-owner Ruff and examples git diff --check passed.

No GPU or large production run was launched, and active jobs were untouched.
Finite-cap draws remain proposals requiring saved importance weights; existing
unweighted MPS plotting/postprocessing does not consume this schema directly.

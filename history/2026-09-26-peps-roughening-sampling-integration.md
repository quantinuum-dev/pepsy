# 2026-09-26 — Integrate selected-time PEPS sampling into roughening

- Scope: user explicitly authorized integration of PepsSampler into
  run_roughening to retain samples at certain times after validation.
- Examples branch / baseline: `main`, `301159a`, ahead one commit at start.
- Pepsy branch / baseline: `develop`, `a13031b`.
- Commit status: uncommitted. Preserved both repositories' extensive existing
  changes; no staging, commits, publication, or production-job changes.

## Implemented

In `pepsy_examples/experiments/mps_magnetization/benchmark`:

- PEPS engine uses public PepsSampler on a private once-absorbed SU snapshot.
  Fresh sampler per time; maps sampler site order into runner mapping order.
- Existing `--save-sample-times` adds exact clipped-step targets; alternatively
  `--save-samples` samples all observation targets. No sampling by default.
- Exposes conditioned ket chi, chunk size, estimated factored cache budget,
  contraction planner, and explicit rho repair. Future cap is chi-boundary.
- Separate peps-importance-v1 NPZ files retain configurations, coordinates,
  scaled q/Psi, log values and stable weights. Summary and manifest record
  snapshot seeds and diagnostics. Boundary observables remain independent.
- Sweep reuse compares sampling controls and checks saved snapshot files.
- Updated runner documentation and outdated no-shots guidance. Package
  numerical implementation was not changed in this integration task.

## Validation

- Initial focused PEPS tests: 24 passed.
- Entire benchmark tests: 402 passed, 6 skipped, 63 existing warnings, 63.10 s.
  Started before final duplicate-time and direct-engine basis guards.
- Final PEPS/sweep/entrypoint/layout selection: 60 passed, 40.70 s; includes
  the final guards, dense NumPy/Torch amplitude and proposal comparisons,
  nontrivial mappings, source/gauge isolation and subsequent evolution,
  seed replay, scheduling, output schema, and missing sample file rejection.
- Actual `run_roughening.py` CLI smoke with default auto-hq: passed. NumPy
  complex128 2×2, D=4, boundary chi=16, dt=.2, depth=2, 8 shots/chunk=3,
  selected t=0,.1,.4. Output:
  `/tmp/pepsy_examples_runs/peps_selected_samples_integration_smoke`.
  Reopened NPZ files without pickle and checked shapes, finite log weights,
  normalized sums, selected times, and complete manifest.
- Changed-file Ruff and diff whitespace checks passed. Repository-wide Ruff
  reports six pre-existing E731/E402 issues in effective/correlations.py,
  tests/test_roughening_eff_correlations.py, tests/test_roughening_eff_theta.py.

## Limits

No GPU integration or long/large production run was launched. Exact sampled
amplitudes remain potentially expensive and are not bounded by the prefix/cache
budgets. Finite-cap proposals require importance weighting; ESS does not certify
support. Existing unweighted MPS/Born sample plots and wall/energy/correlation
postprocessing do not consume the new schema automatically. No entropy added.

Detailed current usage/schema lives in the examples benchmark's
`magnetization/README.md`; implementation evidence is in
`docs/development/notes/peps_sampling_integration.md` in that benchmark.

The known patch-tool sandbox startup failure persists; tracked edits used
asserted exact-match replacements. No approval rejection occurred.

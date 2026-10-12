# 2026-10-08 — PEPS publication review

- Scope: user requested careful commit and push of the accumulated PEPS work
  in Pepsy and the sibling examples repository.
- Pepsy branch/baseline: `develop`, `0ba2a05`, which was already one unpublished
  commit ahead of `origin/develop`.
- Examples baseline: `main`, fast-forwarded from `1a96925` to `563996d` before
  publication; upstream plotting changes did not overlap the PEPS edits.

The reviewed changes add D-squared boundary convergence probes, fresh
confirmation, explicit nonconvergence records, selected-cap normalization,
validated bidirectional boundary reuse, and reduced two-site full-update ALS.
The public row/column and two-site options preserve input gate barriers and
reuse the existing reduced-pair and ALS implementation. See the
[implementation/review evidence](../docs/development/notes/2026-10-08-peps-boundary-reuse-full-update.md)
and [fixed-cap launch correction](2026-10-08-fixed-norm-full-update-launch.md).
The earlier false-convergence fix in `0ba2a05` is included in the pending push.

Validation uses the existing cloudspace environment, local source, hidden
GPUs, two Cotengra workers and one BLAS/OpenMP/Numba thread per test process.
These limits apply only to validation, not production launch settings.
Initial sandbox runs were discarded because Loky/psutil could not resolve
sandbox PIDs; their remaining validation workers were explicitly stopped.

- Full `src tests` Ruff check passed using a tool installed only under `/tmp`.
- Changed Python files parsed successfully; whitespace checks passed.
- Default smoke: 93 passed, one known installed-metadata failure:
  `test_package_version_matches_installed_distribution` sees installed 0.4.0
  against checkout 0.5.0. The shared environment was not reinstalled.
- Focused PEPS optimizer, convergence, boundary numerics, batching,
  performance/safeguard/timing, environment reuse, full-update, gate-order,
  BP reduced-update and public API/layout selection: 438 passed; the only
  failure was the same installed-version metadata mismatch (188 seconds).
- Downstream optimizer/convergence/entrypoint/sweep selection: 134 passed,
  four preexisting bubble-mode compatibility failures. An untouched Git
  snapshot of both repositories reproduced exactly those four failures:
  bubble defaults request a multi-site block with one-site `dmrg`, and the
  legacy `dmrg1` name is removed. These are outside the PEPS commit scope.
- Examples lint: the new convergence test passes from the benchmark working
  directory. Five import/executable-mode findings in the existing engine and
  optimizer test were also reproduced directly from their Git baseline.
- No full-repository numerical-suite pass is claimed.

The user stopped the CUDA:2 production PEPS run after the previous launch
handoff; its results were preserved. Subsequent MPS launches and current
process records live under `/tmp/pepsy_examples_runs/`. No simulation output
or cache is included in these commits.

Validation logs are under `/tmp/pepsy_precommit_bounded_20261008.log`,
`/tmp/pepsy_smoke_precommit_20261008.log`,
`/tmp/examples_precommit_bounded_20261008.log`, and
`/tmp/examples_precommit_baseline_20261008.log`. The publication commit
contains this handoff; earlier journal statements saying uncommitted describe
their original checkpoints in time.

# 2026-10-08 — Fix the three full-update review findings

- Scope: user authorized fixing all findings in the
  [current-commit review](2026-10-08-full-update-current-commit-review.md).
- Branch / baseline: `develop`, `ad8ed05`.
- Status: working-tree implementation, regressions, API/changelog, numerical
  note, and this handoff; nothing staged, committed, or published.
- Preserved concurrent MPS/JAX implementation, tests, documentation and status
  ledger changes. The changelog contains both tasks' separate entries.

## Implemented

- Full-update converts all incoming gates before pair reduction and exact
  strip-target construction, fixing NumPy and mismatched-dtype refinement.
- Single-site updates honor target normalization independently of final
  normalization. Active exact strip targets receive the same scalar scale.
- Smart scheduling updates ready pairs incrementally and visits single-site
  ancestors only for the chosen pair, preserving the previous heuristic.
- Added nine regression cases for backend/dtype conversion on Torch/CuPy,
  normalization defaults and opt-out, active nonunitary refinement targets,
  and scheduling through chained single-site dependencies.

## Validation

Activated existing `py312`, local source, bounded CPU thread counts.

- Initial full-update/refinement/smart-scheduler selection: **97 passed**,
  one warning, 21.58 s; the subsequently added active-target case passed too.
- Final affected PEPS/shared-ALS selection: **486 passed**, 98 warnings,
  no skips, 116.50 s. Includes Torch CPU, CuPy GPU, boundary convergence,
  environment reuse, batching, safeguards, timing, and numerical references.
- Default smoke: **94 passed**, two warnings, 31.02 s.
- Ruff, new handoff/evidence local-link checks, and `git diff --check`: passed.
- Differential scheduling probe: 200 seeded mixed 60-gate queues retain
  exactly the `ad8ed05` execution order.
- Repeated the review's forced-order benchmark: 1,600-gate compilation fell
  from 10.100 s to 3.965 s. Predecessor-copy counts confirm removal of the
  cubic work; quadratic graph construction remains.

Detailed [implementation, benchmark and upstream-audit evidence](../docs/development/notes/2026-10-08-full-update-review-corrections.md).
Temporary logs: `/tmp/pepsy_full_update_fix_regressions.log` and
`/tmp/pepsy_full_update_fix_smoke.log`. Differential probe:
`/tmp/pepsy_scheduler_differential_fix.py`.

## Remaining limits

No full-repository numerical-suite or long-time evolution run. GPU coverage
uses CuPy; no multi-device test was added. No dependency or native-symmetry
changes. The three reviewed issues are addressed; large overlapping queues
can still incur quadratic compiler storage/work.

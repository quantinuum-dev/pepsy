# 2026-09-26 — Reusable PEPS planning and amplitude preflight

- Scope: user requested Pepsy/Cotengra builder use, tag-selected directional row
  reuse like FIT/DMRG, large auto-hq checks, amplitude cost/memory checks,
  automatic cache evaluation, and isolated GPU validation.
- Interpretation: reusable builder for full networks, auto-hq for cached-row
  steps. An optional clarification was offered; implementation proceeded with
  this stated interpretation after allowing response time.
- Pepsy baseline: `develop`, `a13031b`. Examples baseline: `main`, `301159a`.
- Commit status: uncommitted. Preserved prior edits; no commits or publication.

## Implemented

- Default PepsSampler owns `build_optimizer(parallel=False)`, the maintained
  name for build_contraction. Cached rows use auto-hq. Explicit full/row
  optimizer overrides are supported; explicit legacy full optimizers continue
  to feed rows unless overridden.
- Existing X-tag row suffix/prefix and sampled bottom-boundary logic is retained.
  No numerical environment is shared merely because tags or shapes match.
- Optional exact-amplitude largest-intermediate byte and estimated cost limits
  reject work before scaled-leaf allocation/execution, including cached trees.
  Exposed last plan estimates and safe rejection/retry regressions.
- Roughening exposes the planner/limit controls and retains one structural
  optimizer across snapshots while rebuilding numerical samplers.
- Benchmark flags, API docs, runner docs, changelog, and measured policy report.

## Measurements and limitations

See [planner policy report](../docs/development/notes/peps_sampler_planner_policy.md).
9×9 and 10×10 evolved D=4 probes passed with auto-hq and with the new builder
policy. New-policy public two-shot batches took 1.50 s / 3.63 s with process
peaks 659 / 880 MiB. The 10×10 auto-hq plan needed a 16 MiB largest tensor,
showing the prior greedy 4 GiB plan was avoidable. These are short contended
CPU probes, not production-throughput or full-state accuracy certificates.

Automatic factored selection was evaluated, not enabled globally: comparisons
show dependence on planner, shape, and warmup; retain opt-in library behavior.
The runner explicitly selects factored rows. No hard total-memory guarantee.

Isolated GPU validation remains unavailable: the only RTX A5000 is occupied by
the authorized GPU DMRG job (PID 130335 when inspected). No job was stopped,
reconfigured, or queued behind it. No new GPU test was mislabeled isolated.

## Validation

- New-feature suite including policy and guard regressions: 46 passed.
- Runner PEPS/sweep/entrypoint/layout: 60 passed; one added default-policy test
  passed separately. After cross-snapshot structural reuse, all 26 PEPS runner
  tests passed on final code.
- Benchmark smoke: passed; dense amplitude error 1.95e-16.
- Repository smoke: 162 passed, 1 skipped, 1 known version metadata failure:
  installed/runtime Pepsy 0.4.1 versus checkout 0.5.0. No environment edits.
- Existing sampler/shared-result/API/layout suite: 396 passed, 6 skipped,
  1 known installed-version metadata failure, 44 warnings, 338.61 s. CUDA was
  hidden; the six shared sampler CUDA/CuPy checks skipped. Same version failure
  as smoke, with no new sampler failures. No full repository numerical suite.
- Small 4×4 new-policy benchmark comparison passed dense amplitude checks:
  reference 0.621 s versus factored 0.542 s; error <=2.6e-17. Not an isolated
  throughput benchmark or proof of a universal cache selection rule.
- Pepsy Ruff (src/tests/benchmark), changed examples-file Ruff, local report
  links and git diff --check passed. Earlier unrelated examples lint findings
  remain outside this task; no full repository numerical suite was claimed.

The existing patch-tool sandbox startup error required asserted exact-match
replacement scripts for tracked edits. No approval rejection occurred.

# 2026-11-05 — Correct the FU contraction-planner baseline

- Scope: check the user's observation that FU already uses Cotengra through
  `build_optimizer()`.
- Branch / baseline: `develop`, `e94e4b0`.
- Change: correct the prior review's recommendation; numerical code and
  defaults remain unchanged.

## Correction

The [prior handoff](2026-11-05-pepo-fu-opportunities.md) recommended integrating
norm-specific planning after measuring a 12.86x final contraction improvement.
Its probe explicitly forced greedy. The public constructor already creates a
reusable Cotengra optimizer when `contraction_opt=None`, and FU passes it to
the final norm contraction. The earlier result is not an improvement over
the default, and does not justify a new planner/cache integration.

## Follow-up evidence

The [updated evidence note](../docs/development/notes/2026-11-05-pepo-fu-opportunities.md#existing-default-comparison)
records a three-way identical-operand comparison. Existing
`build_optimizer(parallel=False)` and randomized greedy find equal-cost paths:
1.958 billion estimated FLOPs and 4,105,728-element peak intermediates.
Their contraction medians are 0.07454 s and 0.07510 s; forced greedy takes
0.93779 s in the same run. The default matrix differs from greedy by
4.51e-15 after Frobenius normalization.

Default initial planning takes 1.64481 s and a repeated cached lookup takes
0.000304 s. No execution benefit over the default is demonstrated. Default
full-gate profiling, planner startup and actual cache misses remain unmeasured
opportunities, not established implementation defects.

Validation: copied-checkpoint two-gate CUDA probe, alternating fixed-tree
timings, normalized matrix agreement, installed-source call-path review,
local Markdown links and diff whitespace checks. No numerical tests are
required for the documentation-only correction. Production state is untouched.

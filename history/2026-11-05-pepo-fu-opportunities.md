# 2026-11-05 — PEPO full-update performance review

- Scope: push the reviewed cluster batching commit, then investigate analogous
  PEPO `PepsOptimizer` full-update opportunities.
- Branch / baseline: `develop`, `aadd8eb`.
- Commit status: `aadd8eb` pushed to `origin/develop`. This review adds evidence
  only; no maintained numerical implementation/default changed.

## Findings and decisions

The [detailed evidence](../docs/development/notes/2026-11-05-pepo-fu-opportunities.md)
records copied-checkpoint probes and their limits. Disjoint cluster footprints
do not imply FU independence: an exact small-network check finds a 0.00810
relative change in a remote global environment after truncation.

On a copied mature 10x10 D=6 checkpoint, direct-boundary chi56 FU spends 19.1 s
on the cold environment but 0.63--0.72 s on the next three same-row gates.
Those gates reuse all nine transverse cuts. A subphase run isolates about
0.59--0.68 s in the final norm contraction, versus about 0.05 s for the rest
of environment preparation.

An identical-operand randomized-greedy path probe reduces that final kernel
from 0.67103 s to 0.05216 s (12.86x), with one quarter of the largest
intermediate elements and normalized norm-matrix difference 3.89e-15.
Planning takes 0.00170 s. This is a one-graph prototype, not an integrated
feature or production-trajectory speedup. The diagnostic state retains the
original greedy result.

The guarded Torch ALS adapter does not establish a stable drop-in benefit.
The existing ill-conditioned metric uses spectral fallback; local fidelity
agreement alone does not establish physical output equivalence. Keep solver
and conditioning defaults unchanged.

## Validation and limitations

- Exact remote-environment dependency probe, two fixed-problem ALS/gauge
  probes, two four-gate public-driver profiles, and one two-gate path probe
  completed. CUDA timing uses explicit synchronization and private state copies.
- The first standalone environment probe omitted the public driver's strip
  cache and OOM'd. The corrected cached route succeeds; do not classify that
  failure as a default-driver regression.
- Evidence uses direct boundary compression. Default DMRG performance and
  complete circuit trajectories are not established by these checks.
- Numerical code was not edited, so numerical suites were not rerun for these
  documentation-only additions. Validate local Markdown links and diff format.
- Production jobs, frozen sources and archived checkpoints were not modified.

## Suggested next step

Integrate a bounded norm-specific planner/cache, then validate complete
sequential trajectories and physical/reference quantities before selecting a
new default. Keep boundary planning separate. Boundary-front or independent
trajectory batching remains a later possibility; simultaneous same-network
FU gates would require an explicit frozen-environment approximation.

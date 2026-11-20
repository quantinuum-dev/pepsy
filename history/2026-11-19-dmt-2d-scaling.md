# 2026-11-19 — 4×4 and rectangular long-time benchmarks

- Scope: recheck mapped 2D DMT on 4×4 and other shapes, longer trajectories
  and performance. Previous commit/push applied to the preceding work.
- Branch / baseline: `develop`, `04a712435391712b47ad42777bdb4829d9dd7519`.
- Commit status: uncommitted; no production numerical default changed.

## Changes

Added [the scalable benchmark](../examples/dmt_2d_scaling.py) and 15 tests.
The small example's model builder gains a sparse-reference option; its CLI
defaults and dense reference behavior remain unchanged. Sparse Hamiltonians,
statevectors and local panels avoid a global 16-spin density. Use a separately
validated Strang field/commuting-ZZ/field schedule.

Compare routed replay, generic full sweeps and exact rank-two Ising MPO
targets followed by public DMT compression. Analytic multiplication reduces
temporary target rank without a numerical operator cutoff. Different update
orders are explicit; no trajectory-equivalence or universal speedup claim.

## Evidence

[Scientific record and artifacts](../docs/development/notes/2026-11-19-dmt-2d-scaling.md):
all three analytic χ=64 shape runs reach t=20. Endpoint raw plaquette errors
are 0.03650 (2×8), 0.06060 (3×4), 0.04345 (4×4); median step times are
2.54/1.95/4.08 s. A 4×4 χ=96 control and ordinary FIT χ=64 reach t=5.
Routing controls, χ=32 budget rejections, timestep refinement, RSS, per-step
timings, fallback events and a QR/SVD profile are retained. There are
14 rows, 85 readouts and 1,113 timed attempts.

Audited marginal defects stay near roundoff, but positive local panels coexist
with large global Hermiticity loss and local accuracy remains unconverged.
Global PSD is unmeasured (`None`), not assumed. Trace-normalized readout is
separate; the evolved state is not repaired. Supported native SVD fallbacks
occurred; no preservation violation was recorded in the long runs.

## Validation and limits

176 domain tests passed, with one existing Torch rank-policy warning;
after final benchmark additions, 27 focused tests passed. Ruff, skill/catalog,
links and whitespace checks pass. Installed versions remain unchanged;
official upstream audit refreshed, Symmray array-doc unavailable. No package
upgrades, native PEPO, arbitrary mixed-state exact oracle or GPU validation.
Existing package metadata/backend-gradient baseline failures from the preceding
publication were not changed or relabeled as passing.

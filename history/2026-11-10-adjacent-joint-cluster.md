# 2026-11-10 — User-selected adjacent upper/lower cluster fits

- Scope: user requested separate upper/lower versus simultaneous joint fits,
  verification of Cotengra environment contraction and edge/corner selection.
  User explicitly selected adjacent pairing only, with no gate reordering.
- Baseline: `develop` at `add66d9`, with preceding twelve-site and physical-leg
  changes already uncommitted. Nothing staged, committed or published.

## Implementation

Ordinary `gate_loop_cluster` and `PreparedLoopClusterUpdate` now accept
explicit two-site `both` entries with `(upper_gate, lower_gate)`. Optional
`fuse_pepo_legs=True` pairs only adjacent opposite-leg gates on the same
bond, realigning reversed coordinate order. Calls and prepared apply ranges
are hard barriers. Automatic pairing currently requires prepared batch size
one; explicit joint entries can use existing compatible batching.
Separate fits remain the default. Joint fits preserve the exact operator
but can change finite-bond error and runtime by changing truncation points.

The existing full-update adjacent matcher moved into the shared operators
module; its established optimizer wrapper remains available. No duplicate
scheduling algorithm or automatic commuting reorder was introduced. Cluster
callbacks identify the final original entry index, and result metadata retains
original positions. Prepared counters separate physical actions, pair fits
and joint fits. Eli exposes `--[no-]loop-cluster-fuse-pepo-legs`, forwards the
choice to both execution routes and keeps source checkpoints/progress in the
original action coordinates. Ordinary replay now uses one callback-enabled
gate-stream call per source slice rather than wrapping individual gates.

## Cotengra and corner checks

Environment construction already used Cotengra trees and generated array
expressions; no contraction backend replacement was required. Instrumented
checks assert an actual `ContractionTree` and exactly one expression execution
per fit (one for a joint pair, two for separate fits). ALS reuses that tensor.
Every bond on 2×2, 2×5, 5×2, 4×5 and 5×4 open lattices matches an independent
clipped-rectangle oracle. Fixed inactive corner storage stays decoupled;
selected operator amplitudes match an independent untruncated gate network.
The Eli dense small-circuit tests also retain the removed-corner convention.

## Validation

Shared Python 3.12, local sources, one numerical thread; versions and upstream
inspection are unchanged from the preceding handoffs. Adopt the existing
Quimb/Cotengra APIs; no dependency modification or compatibility shim.

- Affected BP/cluster, gate parsing, operator conventions, paired full update,
  public API/layout selection: **454 passed, 3 deselected** in 41.60 s.
- After adding joint CPU/CUDA finite differences and validation checks,
  **20 joint-update tests passed**. These also cover reversed site order,
  explicit/automatic/prepared/reference modes, Cotengra execution, source
  barriers, one-site/same-leg blockers, all-bond geometry and inactive corners.
- Eli loop-cluster workflow: **47 passed** in 18.01 s; **8 adjacent-choice
  cases passed** again after final original-index metadata assertions.
- Full suite fail-fast: **351 passed, 2 skipped, 1 failed** in 50.06 s, at
  the previously recorded native fermionic long-range-hopping readout failure.
  Full log: `/tmp/pepsy-joint-cluster-full-suite-20261110.log`. No full-suite
  success is claimed; the earlier SU warm-start compatibility issue also
  remains outside this change.

No campaign, frozen source, saved result or figure was changed/restarted.
The default for new 10×10 runs remains the rotated twelve-site patch from
the earlier task. No production joint-mode accuracy or speed result is claimed.

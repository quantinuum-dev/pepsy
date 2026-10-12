# 2026-09-30 — Three MPS DMRG modes

This supersedes the same-day [DMRG1 alias implementation](2026-09-30-mps-dmrg1-alias.md).
The user clarified that DMRG1 should be removed and DMRG should perform only
one-site refinement, avoiding two-site FIT updates.

## Implemented behavior

- `MpsOptimizer(mode="dmrg")`: one-site FIT from the initialized guess.
  Explicit `fit_block_size=2` or `3` raises and directs callers to `dmrg2`/`dmrg3`.
  The historical `fit` alias follows the same one-site restriction.
- `dmrg2`: existing two-site warm-up followed by one-site refinement.
- `dmrg3`: existing three-site warm-up, two-site transition, then one-site refinement.
- `dmrg1` raises a migration error at constructor, `set_mode`, `run(mode=...)`
  and shot override entry points. Removed the old latch diagnostic and renamed
  mixed-mode history reasons from `guess_direct_dmrg1` to `guess_direct_dmrg`.
- Target/guess construction can still use SVD and open bond support. One-site
  FIT retains the initialized rank; the existing dense/native preparation and
  exact-target contracts are unchanged.
- Other optimizer classes retain their independently owned mode policies.
  The discussed MPS notebook now labels and executes `dmrg`, `dmrg2`, `dmrg3`;
  old saved outputs remain historical and were not recomputed.

The [same-task upstream audit](2026-09-30-mps-dmrg1-alias.md#upstream-audit)
is reused: environment, dependencies and FIT kernels are unchanged.
Classification: **adopt** the existing one-site kernel with stricter public
mode selection; **defer** unrelated upstream changes. No new upstream shim.

## Validation details

New regressions reject the removed name and multi-site overrides before gate
execution. A separate exact-reference regression replaces FIT's two-/three-site
sweep methods with failing sentinels and exercises batching and fast-path
requests; DMRG stays one-site and reproduces the reference state.

A pre-existing native test expected one-site Symmray FIT to fail. It also
failed against the committed `f01f596` optimizer loaded from a temporary module,
confirming the expectation was stale before this change. The replacement
checks native array retention, the one-site diagnostic, the chi cap and
fidelity 1 to a native MPO reference (absolute tolerance 1e-10). No production
numerical behavior was changed to accommodate that test.

Final counts and limitations are in the
[session handoff](../../../history/2026-09-30-mps-dmrg-modes.md).

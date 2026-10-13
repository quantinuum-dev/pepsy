# 2026-10-13 — Caller-owned native MPS progress

- Scope: align Gaugy MPSEngine progress with native MpsOptimizer gate progress.
- Branch / observed baseline: `develop` at `82ee387`.
- Commit status: working-tree edits only; no staging, commit or publication.
  Concurrent VMC/example edits are preserved.

## Implemented

[`MpsOptimizer.run`](../src/pepsy/optimizers/mps/optimizer.py) accepts a
caller-owned display through the existing `progbar` argument. All native mode
bars send their gate advances and existing diagnostics to it. The private
[`_progress.py`](../src/pepsy/optimizers/mps/_progress.py) adapter prevents a
replay segment from closing the caller's display. Boolean progress retains
the original standalone behavior. No gate, compression, normalization,
canonicalization or gradient policy changed. Classification: **adopt** display
composition through the owning public replay API.

The [API guide](../docs/api/optimizers/mps.md), module map and changelog describe
ownership and the distinction between compression survival and objective cost.
Gaugy aggregates U/V gates into one display per stage and labels sampled cost
separately; see its [handoff](../../gaugy/history/2026-10-13-mps-native-progress.md).

## Fresh validation

Activated the existing shared Python 3.12 environment with one BLAS/OpenMP
thread. **196 passed** across MPS diagnostics/compression modes and public
API/package layout in 15.67 s. Twelve native modes verify identical states,
logical ordering, center metadata and represented scale with/without the
external display, including reuse across two replay calls and caller cleanup.
Warnings cover compatibility aliases, Quimb's SVD defaults and existing SDCR
cutoff coercions.

Ruff over `src tests` and whitespace checks pass. Gaugy's separate scoped
checks cover borrowed displays, no nested bars, complete gate counts, failure
cleanup and unchanged Torch gradients. Full suites and new accelerator runs
are not claimed. Numerical upstream behavior was not changed.

## Styling follow-up

The user identified that Gaugy's aggregate bars still lacked the native mode
colors. `MpsOptimizer.progress_style(mode)` now exposes the owning palette and
ASCII options as an independent dictionary, including aliases and named DMRG
schedules. Native bar construction is unchanged. Gaugy consumes this API rather
than copying private color tables.

Fresh diagnostics/API/layout validation: **83 passed** in 7.20 s, including
twelve comparisons with actual standalone native bar construction and prior
borrowed-display regressions. Ruff over `src tests` and whitespace checks pass.
Prior 196-test compression validation above belongs to the earlier gate-bar
change; it was not repeated for this styling-only follow-up. Existing SVD and
compatibility warnings remain. The follow-up is uncommitted.

## Time/F presentation follow-up

The user requested dark red specifically for direct replay. The native palette
now has a direct-mode override (`#8b0000`), preserving other compressor colors.
Gaugy hides internal resetting counters and shows a time/F target bar instead.
Fresh diagnostics/API/layout checks: **83 passed** in 6.75 s. Full Ruff and
whitespace checks pass; no numerical Pepsy behavior changed or new full-suite
claim is made. Changes remain uncommitted.

# 2026-10-06 — Helix measurement without native shortcut

- Scope: run the C4 archived circuit without native collapse, compare visible
  measurement disentangling True/False, measure seconds per shot and bonds.
- Branch / baseline: Pepsy `develop`, `d41ea9f`.
- Commit status: this handoff is uncommitted; no package/notebook changes,
  commits or pushes. Previous review handoffs remain untracked.

## Method

Used `/tmp/helix_no_shortcut_benchmark.py` in separate sequential processes
under activated genpy. Input is the notebook's archived, noiseless 52-qubit
Stim circuit: 3,818 translated entries, 550 visible measurements and 530
resets. Torch CPU complex128, chi 1024, direct mode, cutoff 1e-12, seed 42,
one direct replay shot, no layout search. Imports, translation, backend
conversion and compilation are excluded from replay timing.

For each no-shortcut process, temporarily replaced native `measure` and
`probabilities` with functions returning None and `fallback_reason` with
`disabled_for_benchmark`. This disables all native collapse, including reset,
and avoids wasting time on unused eligibility scans. No installed or tracked
code was edited. Visible measurement entries explicitly set absorb_basis to
True or False. Resets retain their normal basis-changing MPS algorithm in
both variants. A third fresh process uses the unchanged automatic native path.

Instrumentation checks process memory before entries, observes coefficient
bonds after public entries and internal `_evolve_p` updates, and writes
progress every 100 entries or new public-entry peak. Internal peaks refer to
completed coefficient updates, not temporary MPO/SVD workspace dimensions.
The 120/180-second and 2-GiB guards did not fire. These are instrumented
single-shot observations, not repeated medians or noisy logical-error evidence.

## Results

| Route | Seconds/shot | Peak after public entry | Peak including internal updates | Final coefficient bond |
| --- | ---: | ---: | ---: | ---: |
| No native shortcut, disentangle=True | 114.698347 | 32 | 64 | 32 |
| No native shortcut, disentangle=False | 55.363336 | 128 | 128 | 128 |
| Default native shortcut | 2.659016 | 1 | 1 | 1 |

All runs complete all 3,818 entries, record 550 visible outcomes and finish
with norm one to floating-point precision. Both disabled runs have 1,080 MPS
collapse records and zero native/regional records; the reference has all
1,080 native Stim records and no fallback. No cross-run state-fidelity or
logical-statistics assertion was made. Seed equality alone is not a guarantee
of identical stochastic trajectories across different implementations.

True: first public bond growth at measurements 11/12/13 gives 2/4/8. It stays
at most eight through most replay, generally falling to two, then terminal
readout reaches 16 at measurement 536 and 32 at 543. Internal localizer
updates reach 64. False: bond two at measurement 2; four/eight/16/32 at
measurements 11/12/13/14; terminal readout reaches 64 at 539 and 128 at 547.
Default remains one throughout.

The native shortcut is about 43.1x faster than True and 20.8x faster than
False here. Without native collapse, True uses smaller coefficient bonds
but has additional localizing-Clifford work and takes about 2.1x longer.
Disentangling a projected pivot does not force the entire coefficient MPS to
remain a product. These measurements support retaining the existing shortcut.

## Artifacts / checks

Temporary script and JSON records:

- `/tmp/helix_no_shortcut_benchmark.py`
- `/tmp/helix-no-shortcut-true-chi1024.json`
- `/tmp/helix-no-shortcut-false-chi1024.json`
- `/tmp/helix-no-shortcut-native-chi1024.json`

An initial harness comparison of an array-valued entry head with a string
raised a NumPy ambiguous-truth error before replay. Added the required string
type guard to the temporary harness and reran; it did not change package code.
No numerical suite was needed for unchanged implementation. Whitespace and
Git status checked; only review/benchmark handoffs are untracked.

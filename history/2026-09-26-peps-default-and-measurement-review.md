# 2026-09-26 — PEPS gauge default and measurement capabilities

- Scope: user requested initial-only gauging by default, accepted norm as a
  fidelity proxy, and asked about observables, PepsSampler, and DMRG readout.
- Branch / baseline: Pepsy `develop` / `80f451a`; examples `main` / `301159a`.
- Commit status: unstaged working-tree edits; nothing committed or published.

## Changes and findings

- Changed the example PEPS runner default to `--peps-gauge-every 0`.
  Periodic refresh remains opt-in; initial gauging and gate-local gauge updates
  remain active. Updated existing tests, runner docs, benchmark instructions,
  and the Pepsy changelog. This supersedes the per-step default in the earlier
  [audit](2026-09-26-peps-efficiency-audit.md).
- Current measurements: local Z, mean imbalance, and norm diagnostics. Mean Z
  is derivable; Z second moments and imbalance second moments are not present.
- Tested PepsSampler with absorbed evolved snapshots and Pepsy DMRG observable
  numerator/norm contractions on NumPy/Torch CPU. Both worked. Efficient
  batched DMRG readout and weighted PEPS sampling are not wired into the runner.
- Details, exact probe scope, limitations, and numerical results are in the
  [measurement review](../../pepsy_examples/experiments/mps_magnetization/benchmark/docs/development/notes/peps_simple_update.md#2026-09-26--initial-only-gauge-default-and-measurement-review).

## Fresh validation and limits

- PEPS, roughening sweep, and entrypoint tests: 46 passed in 23.08 s.
- Small 2x3 probes: sampled amplitude error <=1.39e-16; DMRG local Z error
  <=3.34e-16 versus dense contraction. No new CUDA sampler or production
  performance claim. Existing active jobs were left running.
- Ruff for both projects' relevant Python trees and whitespace checks passed.
- apply_patch failed twice due to the existing bubblewrap mountinfo error;
  applied exact checked replacements with the activated interpreter instead.

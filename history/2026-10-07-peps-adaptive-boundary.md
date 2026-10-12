# 2026-10-07 — Adaptive PEPS boundary convergence

- Scope: stop the three PEPS jobs; automatically check norm/overlap convergence
  before sweep fits with bounded chi growth and warning/continuation.
- Branch / baseline: Pepsy develop at 0ba2a05; sibling pepsy_examples main at
  1a96925. The baseline Pepsy reliability commit is locally ahead of origin.
- Commit status: this adaptive work is uncommitted in both working trees;
  nothing new staged or pushed.

## Changes and evidence

See the [numerical note](../docs/development/notes/2026-10-07-peps-adaptive-boundary.md)
and [API policy](../docs/api/optimizers/peps.md#adaptive-boundary-convergence-before-sweeps).
The implementation lives in `optimizers/peps/_boundary_convergence.py` and
the PEPS driver. The example engine exposes matching flags and policy 5.
Defaults are max chi 512, rtol 1e-5, overlap atol 1e-8, doubling. Retain cap
floors but recompute scalar contractions for each new state. No automatic
production restart was requested or performed.

PEPS parents 5923/5924/5926 and children 6299/6300/6301 exited after SIGTERM.
Results are preserved at
`/tmp/pepsy_examples_runs/peps_sweep_comparison_D4_20261007_192044`.
After the bounded diagnostic, GPUs 0–2 are clear; unrelated GPU3 MPS child
89592 remains running. Exact statevector outputs were not changed.

Validation: 21 new adaptive tests, 284 fixed-cap regressions, and 39 example
runner tests passed.
The public API/package selection has one unrelated installed/source version
mismatch (0.4.0 versus 0.5.0), documented in the numerical note. Ruff is
unavailable. Direct and DMRG CUDA complex128 probes passed, retained CUDA
placement, selected chi 8, and normalized output to about 3e-15.

## Remaining limits

No new 5x6 production trajectory has been validated. Pre-fit convergence
does not guarantee accurate moving local environments. At a finite maximum,
nonconvergence is visible and warns; unusable norms and existing local safety
checks are not suppressed. The feature is in `PepsOptimizer` sweep mode;
standalone `SweepOptimizer` and global mode are unchanged. Restart or commit
only on subsequent user instruction.

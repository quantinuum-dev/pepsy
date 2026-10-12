# 2026-09-30 — Boundary PEPS amplitudes and exact-reference experiment

- Scope: user-requested boundary sampling/amplitudes and a beginning 3×3
  D=2/4/6, twelve-angle, dt=0.1 exact-vector comparison, with saved notebook.
- Pepsy branch/baseline: `develop`, `f01f596`; examples: `main`, `3b61b7d`.
- Commit status: working-tree changes only; nothing staged, committed, or pushed.

## Changes

`PepsSampler` supports explicit boundary-MPS amplitude evaluation, using χ′
as the default amplitude cap and preserving phase/exponent. Results identify
approximate weights. The library exact-amplitude default remains compatible;
the downstream roughening frontend defaults to boundary amplitudes.
The examples runner saves schema `peps-importance-v2` and weighted diagonal
observables from the same Z shot batch. A dedicated small validation runner
and `plots/run_exact_peps.ipynb` compare against dense states and report both
raw and weighted averages, true fidelity, and three norm estimates.

See [implementation and validation evidence](../docs/development/notes/2026-09-30-peps-boundary-amplitudes.md).
This supersedes the exact-amplitude limitation in the earlier
[sampling review](2026-09-30-roughening-peps-sampling-review.md).

## Run and validation

Output: examples `experiments/mps_magnetization/benchmark/store/`
`roughening_exact_peps_3x3_dtheta12_dt01_t6_20260930/`.
Launch: `python -m magnetization.runners.roughening_peps_reference --out`
that path, `--samples 4096 --workers 3`, from benchmark with local Pepsy first
on PYTHONPATH and OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1. Execution session 4098;
`run.log`, root manifest, per-case completion files, and `provenance.json` retain
commands, settings, source hashes, and package versions. No GPU used.

Package focused selection: 297 passed, 2 skipped; one pre-existing installed
version metadata mismatch. Smoke: 93 passed. Pepsy Ruff passed. Examples:
86 PEPS/sweep tests plus two observable tests passed; strengthened schema
regression passed. Changed-file Ruff passed, with six unrelated existing
failures in the broader examples lint check. No full-suite claim.

Completed all 36 PEPS cases and 12 references. The new notebook ran all nine
code cells successfully and retains eight figures; executed validation copy
is under `/tmp/pepsy_examples_executed/`. CSV summary, exported figures, and
saved-array validation are in the selected store root. The launch process and
notebook execution both exited successfully.

Mean t=6 fidelities D=2/4/6: 0.007737/0.012300/0.011879. D=6 weighted imbalance
RMSE versus actual PEPS improved from 0.011915 to 0.004878 over angles. Maximum
boundary amplitude relative error was 2.14e-15, but D=6 boundary norm-squared
error reached 9.10% at χ=24. Full numbers and approximation limits are in the
linked note and notebook. The main remaining scientific limitation is SU
accuracy; no claim of D convergence or larger-lattice validation is made.

Preserved unrelated concurrent MPS/tree code, documentation, and notebook
changes. No existing datasets or caches removed.

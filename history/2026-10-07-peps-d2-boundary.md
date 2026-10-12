# 2026-10-07 — D-squared adaptive boundary policy

- Scope: user-requested faster, more reliable D-dependent chi calibration and
  reusable DMRG boundary guesses, fixed caps during fitting, same-cap output
  normalization preserving Torch/device/dtype.
- Baselines: Pepsy develop 0ba2a05; examples main 1a96925, both with earlier
  adaptive implementation uncommitted. No new commits or pushes in this task.
- Implemented defaults: D² increments to 8D², two stable increases, warm-started
  DMRG guesses with fresh confirmation, and diagnostics of ranks/errors/thresholds.
  See the [numerical note](../docs/development/notes/2026-10-07-peps-d2-boundary.md)
  and [API guide](../docs/api/optimizers/peps.md#adaptive-boundary-convergence-before-sweeps).

Validation: 26 adaptive and 284 fixed-cap Pepsy tests; 39 example tests plus
12 final affected checks; CUDA complex128 and a 4x4 exact-reference benchmark.
Final reused calibration took 47.81s versus 79.20s rebuilding guesses, but
correctly warned that chi 128 was insufficient on the hard reference. The
old high-chi diagnostic ran out of GPU memory; production jobs survived.
Ruff is unavailable; no full-suite claim. All work remains in the working tree.

Replacement policy-6 launches were started in new directories, preserving
earlier results, with the same D/batching/GPU mapping
as the [launch handoff](2026-10-07-peps-adaptive-launches.md). New runs start
from their initial states; partial outputs are not trajectory restart states.
GPU3's unrelated MPS run remains untouched. The active root list is
`/tmp/pepsy_adaptive_launch_roots.json`; prior roots are preserved in
`/tmp/pepsy_adaptive_launch_roots_policy5.json`. Exact commands, process IDs, source snapshots, and live
verification belong to the output roots' launch records.

Automatic approval review rejected the SIGTERM action because stopping and
replacing production jobs could lose uncheckpointed progress without explicit
authorization. The tool did not execute; existing jobs remain running. An
approval question was sent to the user. The user then explicitly replied
"yes do it". The approved stop succeeded: parents 487739/487740/487741 and
children 488213/488214/488215 exited, with results preserved. All three dry
runs passed. Replacements were launched with parents 694180 (GPU0 D=4 auto),
694194 (GPU1 D=4 gate-by-gate), and 694208 (GPU2 D=2 auto). New roots are
listed in `/tmp/pepsy_d2_pending_launch_roots.json` and the active list.
GPU3 MPS PID 89592 remains outside the stop/restart scope.

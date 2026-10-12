# 2026-10-07 — User-requested adaptive PEPS comparison

The user subsequently authorized three production launches, superseding the
stopped-job status in the [implementation handoff](2026-10-07-peps-adaptive-boundary.md).
Source remains uncommitted: Pepsy develop baseline 0ba2a05 and examples main
baseline 1a96925, with adaptive changes snapshotted into each output root.

All use 5x6, Torch complex128, dt=0.1, depth=60 (t=6), the previous twelve
delta-theta points, sweep fitting, DMRG boundary compression (`eff`), SciPy
50 iterations per local fit and two round trips per axis. Initial norm and
overlap caps are (128,128), adaptive maximum (512,512), rtol=1e-5,
overlap atol=1e-8. No sampling, entropy, readout, or independent overlap
acceptance checks; adaptive calibration and optimizer contractions remain.
Thread settings are omitted and no inherited thread overrides were found.

| Physical GPU | D | k_2q_batch | Parent PID | Output under /tmp/pepsy_examples_runs/ |
| --- | --- | --- | --- | --- |
| 0 | 4 | auto | 487739 | peps_D4_sweep_dmrg_adaptive_gpu0_20261007_211252 |
| 1 | 4 | 1 | 487740 | peps_D4_sweep_dmrg_adaptive_gpu1_20261007_211318 |
| 2 | 2 | auto | 487741 | peps_D2_sweep_dmrg_adaptive_gpu2_20261007_211323 |

Dry runs passed. All three parents and first-angle children were alive,
allocated on their requested GPUs, and wrote initial checkpoints with the
expected settings. CUDA_VISIBLE_DEVICES masks each child to its physical
GPU, so all child-local device strings read cuda:0. GPU3 MPS remains untouched.
Each root contains launch.sh, launch_record.json, parent.log, child logs,
source patches/untracked sources, and initial_live_verification.json.
`/tmp/pepsy_adaptive_launch_roots.json` lists the three roots.

These are launch checks, not completed-trajectory or accuracy results.
Read live processes and checkpoint timestamps before reporting progress.

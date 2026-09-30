# 2026-09-25 — Launch three 5×5 exact-batch roughening sweeps

- Scope: user-authorized parallel background sweeps at dt=0.05, 0.25, 0.4,
  all through t=10, Torch CPU complex64, 8192 Z shots, 12 threads per job.
- Branch / baseline: Pepsy `develop`, `80f451a`; examples `main`, `301159a`.
- Commit status: no source edits, staging, commits, or publication. This
  handoff is uncommitted; runtime files are under `/tmp/pepsy_examples_runs`.

## Launch and checks

- Matched the completed 4×5 runs: 12 offsets Δθ=3πi/88, i=0…11, J=-1,
  hx=1, hz=0, open diagonal x+y wall, snake mapping, middle-cut entropy on,
  energy and XX off, and no saved full exact distributions.
- Depths are 200, 40, and 25. Requested retained-shot times are 4…10;
  dt=0.4 can retain only 4,6,8,10 from that list.
- All three 12-angle dry runs passed before launch. No other roughening
  jobs were active; the machine reported 38 CPUs and 490 GiB available RAM.
- Started detached sweep parents with separate logs and PID files:
  dt=0.05 PID 3785009; dt=0.25 PID 3785010; dt=0.4 PID 3785011.
- Roots are `exact_batch_5x5_dtheta12_dt005_depth200_samples8192`,
  `exact_batch_5x5_dtheta12_dt025_depth40_samples8192`, and
  `exact_batch_5x5_dtheta12_dt040_depth25_samples8192` under the runtime base.
  Sibling `.launch.sh`, `.dryrun.log`, `.nohup.log`, and `.pid` files record
  each launch. `exact_batch_5x5_t10_samples8192_launches.json` lists all jobs.
- At 19:48:36 America/Denver, all three were running theta_01. Latest saved
  depths were 6/200, 9/40, and 9/25, respectively. Child checkpoints confirmed
  Torch, complex64, exact-batch, and the requested dt values.
- The user asked which other roughening jobs were running. A fresh `/proc`
  check found only these three sweep parents and their three active children;
  no other roughening runner was active. No jobs were stopped.

## Remaining status

The sweeps continue in the background. Completion and later runtime errors
have not been checked. The new outputs have not been added to notebook defaults.

# 2026-09-25 — Launch 5x6 exact-batch roughening with entropy

- Scope: user-authorized background sweep on 5x6, dt=0.4 through t=10,
  12 delta-theta offsets, Torch CPU complex64, 24 threads, 8192 Z shots,
  with the same entropy and physical settings as the previous 5x5 jobs.
- Branch / baseline: Pepsy `develop`, `80f451a`; examples `main`, `301159a`.
- Commit status: no source changes, staging, commits, or publication.
  This handoff is an uncommitted addition; runtime files remain under /tmp.
- Matched the prior launch: J=-1, hx=1, hz=0, open diagonal x+y wall,
  snake mapping, delta-theta=3*pi*i/88 for i=0,...,11, depth 25.
  Middle-cut entropy enabled at the 15|15 partition; sample energy and XX
  correlations disabled; full exact distributions not saved.
- Sampling is 8192 shots per depth. Requested retained-shot times 4,...,10
  match the earlier jobs; only 4,6,8,10 are on the dt=0.4 grid.
- Resource check: 38 CPUs, about 487 GiB RAM available, 2.8 TiB free disk.
  The existing 5x5 dt=0.05 sweep still uses 12 threads; new job requests 24,
  totaling 36 configured compute threads. No existing jobs were changed.
- Twelve-angle dry run passed; `tests/test_entrypoints.py`: 24 passed.
- Detached via nohup and a new process session at 20:59:17 America/Denver.
  Sweep PID 3923668; first child PID 3923805. Root:
  `/tmp/pepsy_examples_runs/exact_batch_5x6_dtheta12_dt040_depth25_samples8192`.
  Sibling `.launch.sh`, `.dryrun.log`, `.nohup.log`, `.pid`, and `.launch.json`
  files contain launch commands, checks, logs, PID, and configuration.
- Live checkpoint metadata confirms lattice, exact-batch, Torch, complex64,
  dt=0.4, torch_threads=24, samples=8192, and middle-cut entropy enabled.
  First angle is running; no completed depth or entropy value was available
  at the startup check. Child log had no traceback. Completion is unverified.
- Notebook integration was not requested in this turn and is unchanged.

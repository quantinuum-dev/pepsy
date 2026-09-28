# 2026-09-28 — Launch 5×6 exact sweep without entropy

- Scope: user approved the displayed command and requested a nohup background launch.
- Pepsy branch: `develop`. No source changes, commits, or publication; this handoff is a working-tree addition.
- Runtime root: `/tmp/pepsy_examples_runs/exact_batch_5x6_dtheta12_dt005_depth200_samples8192_threads36_noentropy`.
  Sibling `.launch.sh`, `.dryrun.log`, `.nohup.log`, and `.pid` retain the command and launch evidence.
- Launched with `nohup setsid bash`, disconnected stdin. Parent PID 688207,
  first child 688319; both verified alive, parent adopted by PID 1.
- Settings: exact-batch, 5×6, dt 0.05, depth 200, final time 10;
  Torch CPU complex64, 36 threads, 8192 Z-basis samples;
  entropy, sampled energy, XX correlations, and saved exact distributions disabled.
  Twelve offsets `3*pi*i/88` for i=0,...,11; J=-1, hx=1, hz=0,
  diagonal x+y wall, snake mapping, retained-shot times 4 through 10.
- New validation: entrypoint tests 24 passed; twelve-angle dry run passed.
  First live checkpoint confirms exact-batch, Torch CPU, complex64,
  36 threads, 8192 samples, disabled entropy/energy, and requested time grid.
- First angle started. Completion and numerical accuracy remain unverified.

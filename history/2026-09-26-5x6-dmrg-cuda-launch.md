# 2026-09-26 — Launch 5×6 CUDA DMRG roughening sweep

- Scope: user-requested detached 12-angle sweep, diagonal wall, dt=0.4,
  t=10, MPS dmrg, Torch CUDA complex128, chi=128, 8192 Z samples,
  entropy disabled, no explicit thread setting.
- Branch / baseline: Pepsy `develop`, `80f451a`.
- Commit status: this handoff is uncommitted; no source changes, staging,
  commits, or publication.
- Runtime root: `/tmp/pepsy_examples_runs/dmrg_5x6_dtheta12_dt040_depth25_chi128_cuda_complex128_samples8192`.
  Sibling `.launch.sh`, `.dryrun.log`, `.nohup.log`, and `.pid` record launch.
- Launched with nohup and setsid; parent PID 4165168, first child 4165284.
  Existing CPU exact sweep was not changed.
- Uses the previous exact sweep's 12 offsets, delta-theta=3*pi*i/88,
  i=0,...,11; J=-1, hx=1, hz=0, open x+y diagonal wall, snake mapping.
  Sample energy and XX disabled. Requested retained-shot times 4,...,10;
  available grid times are 4,6,8,10. Depth 25.
- No `--threads` argument; thread-count environment overrides unset in
  launch script. Checkpoint confirms `torch_threads: null`.
- Validation: Torch 2.6.0+cu124 CUDA complex128 probe succeeded on RTX A5000;
  entrypoint tests: 24 passed; 12-angle dry run passed. Live first-child
  checkpoint confirms requested settings and cuda:0; nvidia-smi lists child.
- At startup, no completed depth yet; completion and later runtime failures
  remain unverified. Notebook integration was not requested or changed.

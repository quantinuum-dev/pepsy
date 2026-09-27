# 2026-09-26 — Launch 5×6 DMRG χ256 at dt=0.05

- Scope: user requested the previous GPU DMRG sweep with dt=0.05 and t=10.
- Branch / baseline: Pepsy `develop`, `a13031b`.
- Commit status: uncommitted handoff only; no source changes or publication.
- Launched detached around 16:32 Denver time; parent PID 130220,
  first child 130335, confirmed live on GPU 0 by nvidia-smi.
- Runtime root: `/tmp/pepsy_examples_runs/dmrg_5x6_dtheta12_dt005_depth200_chi256_cuda_complex128_samples8192`.
  Sibling `.launch.sh`, `.dryrun.log`, `.nohup.log`, and `.pid` retain evidence.
- Copied the preceding dt=0.4 χ256 launch, changing dt to 0.05, depth to
  200, and output path. Same twelve angles, 5×6, diagonal x+y wall, snake
  mapping, J=-1, hx=1, hz=0, Torch CUDA complex128, and 8192 Z shots.
  Entropy, sampled energy, XX, and full exact distributions remain disabled.
  Retained-shot times remain 4 through 10, now all on the time grid.
- Omitted --threads and unset inherited thread-count overrides as before.
- New validation: entrypoint tests 24 passed; twelve-angle dry run passed.
  First live checkpoint confirms dmrg, cuda:0, complex128, 8192 samples,
  torch_threads=null, dt=0.05, depth=200, final_time=10, entropy disabled.
- First angle running at handoff; completion not yet verified. Existing
  exact job was not changed.

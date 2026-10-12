# 2026-09-27 — Repeat 5×6 CUDA MPS DMRG sweep

- Scope: user approved the displayed twelve-angle command and requested a
  detached nohup launch. Pepsy branch: `develop`.
- No source changes, commits, or publication; this handoff is a working-tree addition.
- Runtime root: `/tmp/pepsy_examples_runs/dmrg_5x6_dtheta12_dt005_depth200_chi256_cuda_complex128_samples8192_repeat`.
  Sibling `.launch.sh`, `.dryrun.log`, `.nohup.log`, and `.pid` retain commands
  and launch evidence. Previous outputs are preserved.
- Launched with `nohup setsid bash`, stdin disconnected. Parent PID 513248;
  first child 513344 verified by nvidia-smi on GPU 0.
- Settings: 5×6, MPS dmrg, chi 256, dt 0.05, depth 200, final time 10;
  Torch CUDA complex128, 8192 Z shots, entropy/energy/XX disabled;
  twelve offsets 3*pi*i/88 for i=0,...,11; J=-1, hx=1, hz=0,
  diagonal x+y wall, snake mapping, retained-shot times 4 through 10.
- No explicit thread count; inherited thread overrides unset by the launcher.
- New validation: entrypoint tests 24 passed; twelve-angle dry run passed.
  First live checkpoint confirms dmrg, cuda:0, complex128, 8192 shots,
  dt=0.05, depth=200, final_time=10, entropy and energy disabled,
  torch_threads=null. Parent process alive at verification.
- First angle started; completion and numerical accuracy are not yet verified.

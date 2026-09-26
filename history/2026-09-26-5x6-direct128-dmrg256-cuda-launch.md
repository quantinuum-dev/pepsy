# 2026-09-26 — Concurrent CUDA direct χ128 and DMRG χ256 sweeps

- Scope: user requested two concurrent GPU 0 sweeps matching the completed
  χ128 DMRG run, changing only mode to direct or bond dimension to χ256.
- Branch / baseline: Pepsy `develop`, `80f451a` (same active session).
- Commit status: this handoff is uncommitted; no source edits or publication.

## Launch and validation

- Both use 5×6, dt=0.4, depth 25 (t=10), twelve Δθ=3πi/88 offsets,
  Torch CUDA complex128, 8192 Z samples, J=-1, hx=1, hz=0, open diagonal
  x+y wall, snake mapping, entropy/energy/XX disabled, no full exact distributions.
- Retained-shot times requested at 4…10; available grid times are 4,6,8,10.
- No explicit threads; inherited thread overrides unset. GPU 0 selected with
  CUDA_VISIBLE_DEVICES=0 and --device cuda (CLI rejects cuda:0).
- Fresh entrypoint tests: 24 passed. Both twelve-angle dry runs passed after
  correcting the device spelling. No numerical implementation was changed.
- Launched detached at approximately 12:02 Denver time. Parent PIDs:
  direct χ128 52182; DMRG χ256 52183. Initial children: 52407 and 52369.
- Roots under `/tmp/pepsy_examples_runs/`:
  `direct_5x6_dtheta12_dt040_depth25_chi128_cuda_complex128_samples8192` and
  `dmrg_5x6_dtheta12_dt040_depth25_chi256_cuda_complex128_samples8192`.
  Sibling `.launch.sh`, `.dryrun.log`, `.nohup.log`, and `.pid` files retain
  commands and launch evidence; child logs are inside each root.
- Both parents alive and both children listed by nvidia-smi. First-child
  checkpoints confirm requested modes, cuda:0, complex128, 8192 samples,
  entropy disabled, and torch_threads=null. No completed depths at first check.

## Remaining status

Both sweeps continue in the background; completion is unverified. Existing
CPU exact sweep was not changed. Notebook integration was not requested.

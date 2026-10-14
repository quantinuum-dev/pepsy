# 2026-10-13 — Torch/JAX Symmray VMC compilation audit

- Scope: inspect Pepsy's Torch equivalent of the user-linked Symmray example
  and assess maturity of both backend paths before new algorithm work.
- Branch / baseline: `develop` / `9b2bef2`.
- Commit status: new audit documentation only; nothing staged or committed.
  Earlier uncommitted TDVP improvements were preserved.

## Findings and evidence

- The upstream page demonstrates GPU vmap and a trainable module, not
  `torch.compile` or end-to-end VMC. Pepsy provides additional compiled APIs.
- Exact native fermionic CUDA Inductor amplitudes/gradients and compressed
  boundary inference passed independent checks. Compressed gradients retain
  stabilized original contractions; the tested derivative batch used serial
  execution. Complex Inductor emitted a performance warning; no speedup claim.
- Native fermionic JAX CUDA amplitudes and JIT gradients passed the existing
  validation example. A prior sandbox attempt ran on CPU; it was not counted
  as GPU validation. No dependency or environment edits.
- **19 focused CPU tests passed**, 92.28 s. Scopes, numerical errors and
  remaining gaps are in the
  [audit](../docs/development/notes/2026-10-13-torch-jax-fermionic-maturity.md).
- Both eager and compiled boundary Torch produced NaN log magnitudes for
  zero-amplitude rows outside native parity support. Sampler nonzero masks
  reject these rows. This output-cleanup issue remains unfixed.

## Limits and next step

No new physical trajectory or full repository suite. Prior flat-Z2 compiled
SR/tVMC integration uses bosonic tensors. Establish an integrated native
fermionic CUDA regression and preserve compressed custom VJPs through Torch
compilation before claiming compilation parity or production maturity.

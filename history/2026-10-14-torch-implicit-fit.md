# 2026-10-14 — Torch implicit FIT for MPS replay

- Scope: add a fixed-point Torch VJP for FIT and an opt-in path for MPS DMRG replay.
- Branch / baseline: `develop` / `545c730`; pre-existing JAX FIT work was preserved.
- Commit status: uncommitted working-tree edits; no staging or publication.

## What changed

- Added `pepsy.fitting.make_torch_fit`, an eager custom-autograd wrapper around
  the FIT kernel. Its forward converges local one-site `RL` cycles after an
  optional two-/three-site rank-opening warm-up; backward solves a matrix-free
  fixed-point adjoint with GMRES.
- Added `MpsOptimizer(fit_differentiation="implicit")` for dense Torch
  `dmrg`, `dmrg2`, and `dmrg3` replay. Default replay remains unchanged.
- Documented fixed-rank, first-order, dense-Torch limits and the `k_2q_batch=1`
  requirement. PEPS optimizer integration remains deferred per user direction.

## Validation

- Python 3.12 `py_compile` passed for the new Torch module and edited Python
  modules; `git diff --check` passed.
- No numerical tests were run. The custom complex VJP, GMRES convergence,
  Torch device behavior, and end-to-end gate gradients remain unverified.

## Decisions and limits

- DMRG2/3 block updates are forward-only rank-opening warm-ups. The derivative
  treats their result as the initial point for a one-site fixed point.
- Implicit-solve failures raise and do not silently fall back to MPO replay.
- Native Symmray Torch, batched gate windows, higher derivatives, and
  `torch.compile` are not supported by this path.
- No Torch fixed-point implementation existed in the prior working tree; this
  adds the first Pepsy-owned one.

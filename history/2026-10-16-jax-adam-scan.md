# 2026-10-16 — Chunked JAX Optax updates for cluster benchmarks

- Scope: reduce host/device synchronization in JAX Adam and expose factor
  batching controls for Gaugy's local benchmark.
- Branch / baseline commit: `develop` / current branch was ahead of
  `origin/develop` by one commit; unrelated working-tree edits were present.
- Commit status: uncommitted working-tree edits; nothing staged or published.

## What changed

- Native `jax-adam` now runs 50 Optax updates per JIT-compiled `lax.scan` by
  default. `options["scan_steps"]` chooses a different positive chunk size;
  other Optax solvers retain the previous single-step default.
- The scan keeps optimizer state, best-parameter selection and early-stop
  checks on device. It returns per-step losses and gradient norms, preserving
  the result history and callback cadence while reporting at chunk boundaries.
- Gaugy's `run_benchmark.py` and sweep runner accept and save Adam scan size.
  The local runner also exposes JAX factor-batching mode and its storage budget.

## Why

- The prior Optax path JIT-compiled one update but converted loss and gradient
  norm to Python floats every step. That forced host/device synchronization for
  every Adam iteration.
- p=5's saved JAX report selects ordered scans because its factor storage
  estimate is about 160 MiB compact / 165 MiB uncompact, above the 64 MiB
  budget and 4 MiB automatic threshold. Factor batching remains opt-in; its
  estimate excludes autodiff and compiler workspace.

## How it was checked

- JAX CPU numerical smoke runs with scan sizes 1, 4, and 50 matched parameters,
  per-step histories and final/best losses for 11 Adam steps.
- Additional CPU smokes preserved `patience`, `bad_max`, non-finite gradient,
  best-state restoration and progress-callback behavior.
- Ruff, Python syntax compilation and `git diff --check` passed for changed
  Python files.
- No pytest suite was run. No p=5 GPU timing or factor-batching run was started:
  `nvidia-smi` showed 99% utilization with about 2.1 GiB free, and the active
  GPU processes were not visible in this process namespace. The earlier local
  sweep driver itself was no longer running; its 17 completed records remain.
  The next Adam fit had a partial log but no summary, so it will restart fresh
  when the campaign is resumed.

## Suggested next step

- On a free GPU, compare p=5 Adam with `scan_steps=1` and `scan_steps=50`, then
  try `--jax-factor-batching on --jax-factor-storage-budget-mib 256` while
  recording peak memory and runtime.

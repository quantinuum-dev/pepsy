# 2026-09-28 — Shared random initialization and JAX contraction dispatch

- Scope: continue the user's readability and duplication cleanup.
- Branch / baseline: `develop` / `f410fa0`, preserving the preceding
  [tolerance pass](2026-09-28-fit-tolerance-policy.md) and its earlier changes.
- Commit status: working-tree changes; not staged, committed, or pushed.

## Changes

- MPS and stabilizer-MPS delegate identical dense FIT random draws to
  `fit_random_array` in the existing private random utility. Their method
  hooks, RNG creation, tensor order, and domain-specific guess algorithms
  remain intact.
- NetKet spin and fermion JIT factories share one scaled contraction closure.
  Each adapter retains configuration mapping, fermion phases, output
  formatting, and batching. The existing eager evaluator is unchanged.
- Added ownership explanations to the VMC module map and readability notes.
  No dependency, public signature, default, source module, test, or CI
  selection was added or changed.

## Validation

- Existing focused suites (`test_quimb_compat`, `test_mps_compression_modes`,
  `test_stabilizer_tn`, `test_optimize_mps`, `test_netket_flat_z2`, and
  `test_vmc_api`): **675 passed, 1 skipped, 13 warnings in 27.13s**.
  The skip requires CuPy.
- Temporary baseline probes: **180 sequential random draws** match arrays,
  dtype, and device across NumPy, Torch, and JAX; float16/32/64 and
  complex64/128 templates; three strengths; native Autoray and simulated
  old-Autoray fallback. Sequential draws check generator advancement.
- **64 JAX JIT output/gradient comparisons** match the pre-edit factories:
  dense spin and flat-Z2 fermion PEPS, exact/HOTRG/CTMRG/boundary contraction,
  all three output formats, and gradients of squared amplitudes. Probes use
  reversed site order and CPU, with 32 comparisons each under JAX x64
  enabled and disabled; finite results are required.
- AST comparisons reconstruct both optimizer modules and the NetKet module
  from their pre-edit snapshots, confirming the extracted bodies and all
  other executable code are unchanged.
- Ruff, the two CI mypy targets, the skill catalog, all 31 local Markdown
  links in changed guides/handoffs, and whitespace checks passed.
- Default smoke: `MPLBACKEND=Agg python -m pytest -q` → **89 passed,
  2 compatibility-alias warnings in 18.89s**, exit 0.
- Full combined working-tree suite:
  `MPLBACKEND=Agg python -m pytest -q -ra -o addopts=''` →
  **5,168 passed, 129 skipped, 792 warnings in 511.93s (8m31s)**, exit 0.
  Skips require unavailable CuPy/CUDA/Metal, multiple MPI ranks, or two
  configured XLA host devices. Those configurations remain unvalidated.

## Compatibility and limits

The preceding upstream audit is reused for this continuing task. Installed
versions remain Quimb `1.15.1.dev66+ge927f06e1`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.4.1.dev8+gc45f91457`, JAX `0.8.2`, Torch `2.9.1`, and NetKet `3.22.4`.
Autoray's installed `random.array` and `random.default_rng` signatures were
rechecked for NumPy, Torch, and JAX. Dispatch arguments and optional import
boundaries are unchanged. No runtime speedup or hosted CI result is claimed.

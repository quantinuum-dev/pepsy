# 2026-09-28 — BP, Torch VMC, MPO, and PEPO readability

- Scope: the user explicitly requested the four remaining readability passes.
- Branch / baseline: `develop` / `f410fa0`, with the earlier uncommitted
  [FIT changes](2026-09-28-fit-readability.md) preserved.
- Commit status: working-tree changes; not committed or pushed.

## Changes

- BP shares observable support parsing and separates native-cluster from
  explicit-edge execution after the existing route checks and normalization.
- Torch measurement separates provenance, configuration, and amplitude
  preparation. Both sampling estimators share sweep bookkeeping while callers
  own cumulative counters and profiling/progress state.
- MPO construction separates term validation, prefix sharing, suffix merging,
  and transition/slot emission. Constructor dispatch still uses `cls`.
- Dense PEPO construction separates cluster-family assembly and reporting.
  One allocator and block mapping preserve sector order and lower-cluster
  subtraction. Numerical solvers are unchanged.
- Updated owning API guides, module maps, and the
  [readability review](../docs/development/notes/readability_api_2026_09.md).
  No public signatures/defaults, dependency profiles, tests, or CI selections
  changed. No benchmark directory or documentation builder was added.

## Validation

- BP: `test_bp_open_series`, `test_bp_symmray`, `test_bp_relay` →
  **94 passed**, three warnings, **7.69s**.
- VMC: `test_vmc_api`, `test_vmc_importance`, `test_vmc_transition_plan`,
  `test_vmc_convergence`, `test_vmc_local_energy`, `test_vmc_distributed` →
  **62 passed**, one warning, **14.49s**.
- MPO: `test_mpo_automaton`, `test_mpo_cluster` → **47 passed**, **3.81s**.
- PEPO: `test_cluster_expansion`, `test_pepo_active_storage`,
  `test_pepo_cutoff_policy` → **75 passed**, three warnings, **20.10s**.
- AST comparison against pre-refactor snapshots: all existing signatures
  and decorators match; 340 unaffected function/method bodies match;
  all 17 extracted helper bodies match their source statements. The moved
  support parser matches both original copies.
  Inlining the helpers reconstructs all seven changed entry points exactly,
  including their original executable statement order and argument binding.
- Direct baseline comparisons: 14 MPO cases match exactly for arrays,
  channels, and slots, including two Torch gradient comparisons. Thirteen
  PEPO cases match exactly for active blocks, sector numbering, and reports
  across orders 1–5, C4, complex beta, and rank caps. One unsupported
  C4/complex-beta case raises the same error before and after.
- Eight Torch saved/raw measurement cases match exactly apart from timing
  fields: chain shapes, stored amplitudes, deduplication, empirical weights,
  proposal weights, and multiple observables. Stale Markov samples raise the
  same provenance error. Final call-formatting edits preserve the complete AST.
- Ruff (`src tests`) and the two CI mypy targets passed.
- The new MPO guide example matches its dense reference; the PEPO
  build/materialize example passes. All 21 local Markdown link targets in
  changed guides/handoffs exist; whitespace checks passed.
- Full suite after all four refactors:
  `MPLBACKEND=Agg python -m pytest -q -ra -o addopts=''` →
  **5,168 passed, 129 skipped, 790 warnings in 506.76s (8m26s)**, exit 0.
  This includes the preserved FIT changes. Skips require unavailable
  CuPy/CUDA/Metal, multiple MPI ranks, or two configured XLA host devices;
  those configurations remain unvalidated.
- Default smoke: `MPLBACKEND=Agg python -m pytest -q` →
  **89 passed**, two compatibility-alias warnings, **17.25s**, exit 0.
- Final Ruff, focused mypy, local links, and whitespace checks passed.
  No files are staged; this batch and the preceding FIT pass remain
  uncommitted. No hosted CI or downstream publication result is claimed.

## Limits

This completes the four proposed orchestration passes. Large numerical
kernels elsewhere remain separate review topics. No performance improvement
or exhaustive line-by-line package review is claimed. Local checks do not
establish a hosted CI result.

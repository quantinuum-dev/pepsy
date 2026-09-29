# 2026-09-29 — Tree roughening runtime alignment

- Scope: implement automatic cutoffs, unitary stabilization, unforced CPU
  threads, and disabled optional diagnostics for Tree DMRG roughening; explain
  remaining iteration differences from MPS. The user subsequently authorized
  committing and pushing the accumulated MPS and Tree work in both repositories.
- Baselines: Pepsy `develop` / `25f2737`, examples `main` / `efb7696`.
  The initial review used Pepsy `a233a40`; independent cluster work was committed
  before this handoff. Its implementation is outside this change.
- Publication: this journal accompanies the authorized implementation commit.
  No production job was launched, restarted, or stopped; no data was moved.

## Implemented

- TreeOptimizer already defaults to `cutoff="auto"` and
  `cutoff_mode="auto"`. Roughening forwards those defaults: complex128 resolves
  to cutoff 1e-12, rsum2, and FIT convergence tolerance 1e-9.
- Added TreeOptimizer `stabilize_unitary` at construction and as a scoped
  replay override. Like MPS, standalone Pepsy defaults it off; roughening
  enables it by default. Compression loss enters the norm-survival ledger
  before the canonical centre is scaled back to the incoming norm. The
  physical exponent does not acquire discarded compression loss. Explicit
  non-unitary replay and `track_norm=False` operations retain their scale.
- Stabilization uses detached backend scalars. Accelerator zero checks are
  combined until the replay boundary, avoiding per-update host polling.
  Direct standalone updates check immediately; optional finite checks remain
  opt-in. Copy, shot overrides, failure cleanup, dtype, and gradients are covered.
- TreeOptimizer and TreeSampler default to `threads=None`; roughening forwards
  omitted `--threads` unchanged. This leaves ambient CPU libraries untouched
  and does not control CUDA kernel parallelism.
- `record_history=False` now also suppresses accumulated norm and FIT records.
  Compact `norm_diagnostics(include_history=False)` uses scalar totals and the
  latest event, without traversing retained history. The latest FIT record
  remains available without keeping every update.
- Roughening disables optional spectrum/bond scans, finite/overlap diagnostics,
  per-gate progress, profiling, and accumulated histories by default. Per-depth
  bond maps require `--tree-bond-diagnostics`; profiling/history require timing.
  Convergence calculations, compact cumulative fidelity, requested physical
  observables, and coarse run/depth elapsed times remain.
- Tree resume configuration includes an execution-policy marker. Older runs
  that recorded stabilization without implementing it cannot be silently reused.
  Optional finite/overlap controls now reach TreeOptimizer when explicitly set.
- Updated public docs, changelog, Tree skill references, and regressions.

The same publication includes the earlier MPS changes: adjacent-pair budgets
inherit `n_iter` unless explicitly capped, and roughening now defaults to
`fit_single_pair_fast_path=False`, pair cap None, and `n_iter=8`. Both short
and longer windows allow up to eight sweeps with convergence stopping. This
supersedes the intermediate roughening 5/8 defaults in the earlier journals.
Examples also include the previously reviewed resume-configuration checks,
pre-evolution provenance, sibling import fix, and current CLI controls.

## Validation

Used the existing Python 3.12 environment and local Pepsy sources. CUDA was
hidden and CPU libraries limited only inside validation subprocesses. Those
settings do not alter defaults or any running simulation.

- Tree/domain suite (`test_tree_*.py`, `test_optimize_tree`,
  `test_optimize_tree_stabilizer`, `test_trajectory_noise`, excluding slow and
  benchmark markers): **1405 passed, 35 skipped**, 111.83 s. Log:
  `/tmp/tree-runtime-validation-final-20260929.log`.
- MPS budget/kernel/control/audit and public API/layout selection:
  **335 passed, 1 deselected**, 23.71 s. The deselected version-metadata check
  is explained below. Log: `/tmp/mps-api-validation-final-20260929.log`.
- Complete examples benchmark suite: **475 passed, 6 skipped**, 74.33 s.
  Includes real stabilized/unstabilized finite-chi roughening runs, unforced
  optimizer/sampler threads, disabled diagnostic scans, and resume rejection.
- New Tree stability regressions: **27 passed**, including NumPy, Torch CPU,
  supported native even-parity fermionic states, gradients, non-unitary scale,
  compact ledgers, copying, and zero-state failures.
- A separate 200-update complex64 Torch probe retained norm 0.99999833 with
  event count 200 and no retained history; no GPU performance claim is made.
- Package Ruff, changed examples Python files, skill quick validation,
  12-skill catalog validation, and both repositories' diff checks pass.
- The public package-layout version test has a pre-existing environment
  mismatch: installed distribution 0.4.0 versus source 0.5.0. It failed in the
  initial check and is excluded from the final focused rerun. The shared
  environment was not reinstalled. Broad examples Ruff has the six previously
  reported effective-model findings; those files are outside this change.

The first Tree suite run found an outdated assertion expecting one default
CPU thread and an unsupported odd-parity native FIT fixture. The assertion
now matches the requested default; the new fixture uses supported even-parity
local states. Final checks cover both corrections. Full-package and production
GPU/performance suites were not run for this change.

## Upstream compatibility audit

Reused the same-task official Quimb changelog, Autoray repository, Cotengra
documentation/changelog, and Symmray repository audit. The Symmray array docs
page was unavailable; installed source and the official repository were used.
Installed versions: Quimb 1.15.1.dev66+ge927f06e1, Autoray
0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1,
Symmray 0.4.1.dev7+g83fb22865, Torch 2.6.0+cu124.

**Adopt:** existing Tensor.modify and backend scalar dispatch. **Defer:**
dependency upgrades and low-level cutoff-default changes; Pepsy explicitly
resolves rsum2. No compatibility shim or installed-library edit was needed.

## Remaining differences and recommendation

Tree iteration semantics are unchanged: `fit_n_iter=8` allows eight
inward/outward cycles, or sixteen directional passes; MPS permits eight
directional passes. Tree patience=2 counts two stable comparisons; MPS counts
two samples, or one comparison. Generic Tree uses two block-growth iterations
before one-node refinement, while MPS uses rank-adaptive refinement.

For future parity, define a common directional-pass budget and patience
meaning, then compare rank-growth/refinement schedules on matched small
references. Do not simply halve Tree's budget without deciding how the
warm-up phase should count. Geometry and local problem sizes remain different.

The prior [Tree review](2026-09-29-tree-roughening-dmrg-review.md) records the
12-angle geometry/reference checks. Its optional R/L CLI incompatibility is
unchanged; the default RL path is supported. The earlier
[MPS review](2026-09-29-mps-fit-budget-review.md) also records the historical
explicit-None DMRG2 resume-policy edge case and stale internal MPS skill text;
those are not fixed by the new Tree-only policy marker.

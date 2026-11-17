# 2026-11-16 — Ordinary MPO DMT gate replay and 2D validation

- Scope: the user explicitly requested full `MpoOptimizer` gate-stream replay
  using DMT compression, plus stronger 2D correctness/speed checks and manual
  compression. Repository maintainer/tensor-fitting guidance was used.
- Branch / baseline: `develop`, `59bb6ff0`.
- Status: uncommitted working-tree changes. Preserved unrelated sampling,
  test-harness and historical-note edits. No tutorial or sibling repo changed.

## Implemented

- `mode="dmt"`, `run(dmt_regions=..., dmt_variant=..., dmt_diagnostics=...)`,
  `compress_dmt()` and `get_dmt_reports()`. Reuse exact gate-target construction
  and the existing protected computational-basis SVD. Gate factorization now
  has an explicit zero-cutoff target route rather than Quimb's implicit
  `1e-10` cutoff. Existing matrix payload convention is preserved.
- Transactional candidate commits, in-place identity, atomic replay rollback,
  canonical metadata, dtype promotion and backend metadata restoration.
  DMT invalidates ordinary norm-survival fidelity estimates until `set_mpo`;
  its reports remain separate. Unsupported routes fail explicitly.
- Backend-owned, capability-gated legacy Autoray string namespace correction
  was necessary for NumPy/Torch/JAX replay in one process. Modern namespace
  caches retain identity/behavior; existing device handling is preserved.
- Direct matrix-unit positive-mixture references, dense per-gate partial
  traces, Torch finite-difference checks and a reproducible CPU benchmark.
- [Evidence and limitations](../docs/development/notes/2026-11-16-mpo-dmt-replay.md)
  records upstream checks, per-step versus trajectory errors and performance.

## Validation

- MPO/backend/gate domain: **364 passed, 3 skipped**. Final expanded replay
  file: **35 passed**. Pauli plus earlier replay regressions: **252 passed,
  one pre-existing expected failure**. These selections overlap.
- Final backend/replay rerun: **106 passed, one skipped**. The DMT workflow
  example executes successfully; the namespace capability probe does not
  import Torch when only constructing a lazy namespace.
- API/layout: **54 passed, one known installed-version failure**. Full suite
  stopped at the known BP namespace-dispatch failure after **104 passes and
  7 skips**; no task code executes on that path.
- Complex128 and complex64 benchmark cases passed independent density/trace
  validation. Four-gate 4×4 DMT replay: 2.454 s CPU median, maximum protected
  defect 3.86e-15. No wall-time speedup is claimed from eliminating dtype-copy
  allocations; a matched comparison was effectively unchanged.
- Ruff, Markdown link targets and `git diff --check` passed at handoff.

## Limits

- Per-step protection references the current compressed state. The measured
  3×3 final defect versus the entirely exact circuit was 1.15e-3, despite
  machine-precision preservation at each individual compression.
- Ordinary complex DMT does not guarantee global Hermiticity/positivity.
  Adaptive decomposition is host-controlled; no full-JIT, native PEPO,
  cyclic, native-symmetry, channel-sum or arbitrary-arity DMT route was added.
- Full exact targets and global sweeps remain the primary 2D cost. Native
  GPU execution was not tested. The follow-up below adds workspace reuse for
  adjacent gates; distant gates retain the full-target cost.

## Follow-up: auto cutoffs and faster local updates

- User requested smart automatic cutoffs and faster/stable square-lattice
  replay. Standalone and manual ordinary-MPO DMT now default to auto/auto;
  replay already did. Dtype thresholds are shared, while DMT auto mode is
  relative to the free residual spectrum. Explicit zero remains available.
- Reuse canonical cores, normalized trace environments and region plans for
  adjacent gates after the first full protected sweep. Clone only workspace
  structure for transactionality. Record the actual canonical center and
  tensor isometry hints. Preserve all marginal/trace/rank guards.
- Distant/one-site gates and dtype promotion use exact targets and full
  protected sweeps. This changes the approximation schedule, not the local
  preservation contract; no globally optimal MPO claim is made.
- Matched auto/auto alternating CPU comparisons: **18.1% less time on 3×3**
  and **8.6% less on 4×4** four-edge streams. A complete 12-edge 3×3 layer
  passes independent per-gate target checks. Largest defects over the expanded
  benchmark are `5.98e-15` (complex128) and `4.28e-6` (complex64).
- Expanded MPO/gate/all-Pauli DMT suites: **536 passed, 2 skipped, one existing
  expected failure**. Benchmark and detailed policy/bottleneck findings are in
  the [follow-up evidence](../docs/development/notes/2026-11-16-mpo-dmt-replay.md#follow-up-automatic-cutoffs-and-local-replay).
- Final replay/backend checks after adding local nonunitary/zero-trace rollback:
  **117 passed, one skipped**, overlapping the domain selection. Ruff and diff
  checks pass. API/layout rerun: **54 passed, one existing version mismatch**
  (`0.5.0` installed vs `0.6.0` checkout). Full rerun stops at the same BP
  `test_loopunion_selects_both_plaquettes_and_clips_at_edge[where0-expected0]`
  namespace failure after **104 passed, 7 skipped**; no clean full-suite claim.
- Changes remain uncommitted on `develop`; unrelated sampling edits retained.

## Large-square completion and publication

- User requested 6×6/8×8 stability and speed work, then explicitly requested
  wrapping up, committing and pushing. This implementation commit includes
  the earlier ordinary-MPO replay/auto-cutoff work above; its earlier
  uncommitted statuses describe those intermediate handoffs.
- Balanced log-scaled panels, direct strided traces and per-pair projection
  reuse improve matched compression times by **2.07× on 6×6/χ=96** and
  **2.70× on 8×8/χ=128**. Fixed rank-136 sweeps validate χ=96/128 and reject
  χ=64 explicitly on these inputs.
- Later nonlocal gates use exact forward SWAPs and protected return SWAPs,
  remapping physical regions and restoring any temporary protected budget to
  the final cap. Four-gate interior streams on both lattice sizes passed
  independent reduced-density references at χ=96/128. No full large-square
  layer or completed old/new large-replay timing comparison is claimed.
- The complex64 8×8 replay initially failed the independent trace tolerance.
  Bounded native correction to the actual gate target trace reduced its trace
  defect from **2.78e-5 to 6.29e-7**, with **2.47e-6** local defect. The
  tolerance was retained. Nonunitary trace scale and Torch derivatives pass.
- Final numerical selection (MPO/replay/gates/backends and all five Pauli DMT
  suites plus primitive contraction checks): **634 passed, 3 skipped, one
  pre-existing expected failure**, 130.32 s. Ruff and diff checks pass.
- Final full-suite attempt: **104 passed, 7 skipped**, stopped at the same
  pre-existing BP namespace failure. API/layout: **54 passed**, one known
  installed-version mismatch. These failures remain outside this change.
- [Large-square evidence and reproduction commands](../docs/development/notes/2026-11-16-mpo-dmt-large-squares.md)
  record inputs, protocols, guards, timing limits and final precision checks.
- Publication scope: DMT implementation, numerical tests, benchmark and owning
  docs on `develop`, baseline `59bb6ff0`. Unrelated sampling code, its
  changelog entry, test-harness edits, historical notes and device-local
  `AGENTS.override.md` are excluded. Commit/push outcome is reported in the
  accompanying session response.

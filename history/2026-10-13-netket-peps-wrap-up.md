# 2026-10-13 — PEPS/NetKet code review and simulation stop

- Scope: user requested stopping simulations, reviewing/fixing API and code,
  and committing the PEPS/NetKet work; physical simulations are deferred.
- Branch / baseline: `develop` / `82ee387`.
- Commit status: included in the scoped PEPS/NetKet review commit.
- Commit scope: PEPS/NetKet adapter, shared SR defaults, regression checks,
  TFIM diagnostics, API documentation and dated evidence. Concurrent MPS
  progress edits remain outside this commit. No push requested or performed.
- Earlier audit and short-run drafts are included in this commit; their
  original uncommitted status describes the earlier handoff stage.

## Findings and changes

The simulation stopped at t=0.99, with latest tensors/vector saved at t=0.92.
Final infidelity against physical-initial-state exact evolution was 0.18436;
this does not validate t=5 dynamics. Logs and existing checkpoints were
preserved and partial reports refreshed. No simulations were restarted.
The [retained note](../docs/development/notes/2026-10-13-netket-peps-stopped-run.md)
records setup, scalar results, frozen contraction checks and remaining work.

Review fixed NetKet complex-output SR defaults, sampler thinning/burn-in,
reverse-mode selection for compressed PEPS, named TDVP solvers and dense
allocation checks. Native Quimb tensor scale extraction stabilizes boundary
amplitudes; the example also normalizes its exact dense oracle before
restoring the logarithmic norm. Final review aligns shared `real-imag` SR
mode with NetKet VMC_SR and honors false-valued normalization options.

## Validation and next work

Earlier installed-stack numerical/API/GPU selections are detailed in the
retained note; none establishes full-suite or long-time convergence.
Final review: three SR/scaling checks passed (36 deselected, 12.29 s),
Ruff over src/tests/example passed, whitespace checks passed and 24 local
Markdown links resolved. Physical follow-up requires controlled
dt/samples/regularization/D comparisons before increasing system size.
Snapshots lack full sampler/RNG/integrator restart state.

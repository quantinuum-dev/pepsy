# 2026-10-05 — Fix the four MPS trajectory review findings

- Scope: user-authorized fixes for the four findings in the
  [re-review](2026-10-05-mps-trajectory-rereview.md).
- Branch / baseline: `develop` / `bec773a`.
- Commit status: working-tree changes only; nothing staged, committed or pushed.
  Pre-existing solver, MPO, sampling, documentation and test work is preserved.

Implemented exact/coalesced control preparation without reconstruction
truncation, amplitude-based dense Kraus probabilities, real-Torch generated
gate compatibility, and scoped JAX precision throughout MPS numerical replay.
Reused the existing tree precision context through the backend owner. Added
36 regression cases and updated the owning API guides and changelog.

Validation: 90 regression/API/layout checks pass; 319 non-slow native/control
checks pass; three logical-CPU-device JAX checks pass. Broader selections still
expose existing BP convergence, tree JAX projection, and strict GPU diagnostic
comparison failures. No tolerances were weakened. Optional large-lattice slow
checks were interrupted and excluded from the completed focused rerun;
multi-rank MPI is unavailable. Ruff and diff checks pass.

See the [implementation and dependency audit](../docs/development/notes/2026-10-05-mps-trajectory-corrections.md)
for exact selections, failure names, baseline comparisons and numerical limits.
The four reported defects are fixed; unrelated numerical-test discrepancies
remain recorded rather than being presented as a clean full-suite result.

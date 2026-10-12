# 2026-10-05 — Align TreeStab with TreeOptimizer compression

- Scope: review TreeOptimizer direct/DMRG and align the stabilizer-tree
  coefficient engine and compression controls.
- Branch / baseline: `develop` / `6ab177b`.
- Commit status: finalized in the commit containing this entry; the user
  authorized publication to `origin/develop` after a final review.

TreeStab now exposes the ordinary tree's DMRG/FIT, successive and zipup
algorithms, plus their constructor controls. Numerical updates still use the
existing TreeOptimizer/TreeFIT kernels. Rebuilt engines preserve live settings
and named schedules, and `get_fit_diagnostics()` delegates to the engine.
Public threaded shots initialize Stim matrix conversion before dispatch to
avoid a reproduced cold-start deadlock. The resumed audit removed duplicate
full-TreeMPO gate construction and legacy MPO localizer routing: coefficient
gates now use ordinary compact gate dispatch. Physical caps reconstruct
losslessly before selected-engine compression, fixing the independent
absolute cutoff. Intermediate bond limits and engine diagnostics controls
are forwarded. Exact Clifford gauge moves retain the ordinary tree's
untruncated two-qubit kernel to preserve the represented physical state.

Final validation: combined tree/TreeStab/FIT/compression/trajectory/replay/
public API selection **773 passed**, with JAX x64 enabled. The parity module
contains **133 cases**. An exact zero cap skips unnecessary compression,
avoiding a DM division by zero; every nonzero cap uses the selected engine.
Ruff and whitespace checks passed. No full repository suite,
real accelerator or multi-rank MPI run.

Pre-publication review: repeated the same **773-test** selection with JAX x64
enabled and checked the final diff. The updated skill passed its individual
validator and the twelve-skill catalog/link check. No dependency or sibling
repository changes; device-local instructions excluded from staging.

Integration: `origin/develop` advanced to `bec773a` during final review.
Rebased the alignment commit onto that revision; the only conflict was
`CHANGELOG.md`, resolved by retaining both entries. The numerical alignment
implementation and parity tests are unchanged by the rebase.
The integrated selection, including the upstream zero-weight regression
module, passed **788 tests** with JAX x64 enabled. Ruff passed on the integrated
tree. No tests were skipped in either pre-publication selection.

See the [audit and compatibility evidence](../docs/development/notes/2026-10-05-tree-stabilizer-compression-alignment.md)
and [public API guide](../docs/api/optimizers/tree_stabilizer.md).

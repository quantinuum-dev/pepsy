# Tree direct compression's final branch center — 2026-09-29

The user authorized the final-return optimization identified in the
[routing audit](../../../history/2026-09-29-tree-routing-canonical-audit.md).
It is now implemented in `TreeOptimizer._compress_subtree` for branched
direct/DM compression, including the shared `guess-direct` preparation path.

## Algorithm and invariant

The depth-first edge order is unchanged. Each recursive visit carries whether
it must restore the center to its entry node. An earlier sibling must restore
it; the last sibling must also restore it if an ancestor still has work.
Only the final descent from the hub carries no restoration requirement.

Before every cut, the source tensor remains the canonical center and the
destination's exterior is canonical. Compression advances the center across
that edge. Necessary returns preserve that invariant for later branches.
After the last cut, no further operation needs a return to the hub, so the
recursion reports its terminal node as the state-owned center. Existing TTN
compression and canonicalization routines maintain tensor `left_inds` proofs.

For a given completed update, the removed steps are a lossless change of
gauge. The state and its canonical-center norm are preserved up to roundoff.
The saved work is one return move per edge between the final tensor and hub;
there is no universal wall-clock speedup claim. The earlier 24-case probe
saved three moves in each sampled geometry.

Later path selection depends on the incoming center. Finite-bond trajectories
therefore need not match the previous endpoint policy. Tests check exact
multi-gate action when untruncated and valid canonical metadata and local
norms after finite-bond updates. They do not demand identical entire truncated
trajectories under different sweep directions.

The local compression infidelity remains the existing canonical-norm metric
with log accumulation. There is no added target, overlap contraction, norm
diagnostic, timing call, or backend-specific numerical kernel in replay.
Path compression and successive SRC/SDC/SDCR rounding keep their existing
algorithms.

## Upstream decision and validation

**Adopt:** reuse the existing backend-neutral TTN compression and QR wrappers;
only change traversal control and the final center assignment. No new upstream
API, dependency change, or compatibility shim is needed. The
[same-session audit](2026-09-29-tree-norm-backends.md#upstream-audit) of the
unchanged environment applies, including its installed versions, inspected
Quimb/Autoray capabilities and Symmray documentation fallback.

New regression coverage compares the entire compression/return trace against
an explicit return-after-every-child reference, checking dense states, local
norm metrics, and canonical proofs. It covers binary/ternary trees, direct/DM,
lossy and lossless updates, NumPy/Torch, and native graded replay. Separate
sequences exercise subsequent supports, DMRG direct guesses, and device
preservation. Torch state gradients through both lossy and lossless branch
updates also agree with the reference sweep. Fresh results are recorded in the
[implementation handoff](../../../history/2026-09-29-tree-final-branch-center.md).

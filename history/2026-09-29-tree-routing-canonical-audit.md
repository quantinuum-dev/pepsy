# 2026-09-29 — Compact tree routing and canonical metadata audit

- Scope: check SubTreeMPO integration, `left_inds` reuse, canonical movement,
  and unnecessary decomposition work in TreeOptimizer direct/DMRG replay.
- Branch / baseline: `develop`, `b343ced`, with existing working-tree repairs.
- Commit status: only this handoff added; no implementation changes, staging,
  commits, or interference with concurrent edits.
- User clarification: retain the cheap local canonical-norm compression metric
  and its log accumulation. The branching-direct difference from exact target
  overlap in the [previous audit](2026-09-29-tree-local-fidelity-audit.md) is
  acceptable under that contract. It is not a request to build diagnostic
  targets or add exact-overlap contractions.

## Confirmed implementation behavior

- Ordinary gate replay lowers to compact `SubTreeMPO` and
  `apply_sub_mpotree`. Operators store only the active connected subtree;
  exterior identity tensors are implicit. Repeated unchanged gate payloads
  reuse their factorization. Tests cover this dispatch across all named modes,
  compact/full equivalence, reordered physical labels, a physical root, scale,
  and native graded action.
- The gauge belongs to `TreeTensorNetwork.canonical_region`, with its singleton
  exposed as `orthogonality_center`. Optimizer `center` delegates to that state.
  Tree replay has no separate MPS-style `info_c`. Optional TreeMPO `info_c`
  snapshots are not the optimizer's source of gauge truth.
- Tensor `left_inds` supplies local isometry proofs. Dense and eligible native
  graded tensors skip proven redundant QR. Native charge alignment guards keep
  the graded fallback when the proof is unavailable. Raw unmanaged data edits
  require invalidation/synchronization, which clears stale proofs.
- Known-center movement touches only the geodesic. A known region is peeled
  only as needed toward a new region or connecting path; full preparation is
  reserved for unknown gauge. Contained/overlapping region tests establish
  locality, including unchanged exterior arrays and operation counts.
- Direct routing completes the exact operator action before truncation,
  preserves routed Q factors' `left_inds`, and uses one-sided reduced
  compression when the destination is proven isometric. Path updates compress
  each edge once and retain the terminal center without a return walk.
- DMRG retains operator/state layers and cached environments. Block preparation
  stops at the first entry into the block, and does no QR if the current center
  or canonical region is already contained. Installation updates only local
  factors and preserves exterior isometry proofs.
- One-site unitary replay preserves center and isometry metadata without QR.

## Measured checks

- Routing/canonical/native/FIT selection: **287 passed, 9 skipped**, 1 warning,
  53.22 s. Files: `test_tree_path_execution.py`,
  `test_tree_canonical_regions.py`, `test_tree_operator_representation.py`,
  `test_tree_compression_hook.py`, `test_tree_fit_messages.py`,
  `test_tree_fit_priorities.py`, `test_tree_device_fastpaths.py`,
  `test_tree_native.py`, and `test_tree_state.py`.
- Compact operator selection: `test_tree_submpo.py`: **53 passed**, 6.25 s.
- Fresh tests used the existing Python 3.12 environment, CUDA hidden, and
  single-threaded BLAS/OpenMP. Native symmetry, NumPy, and available Torch/JAX
  CPU cases were included. No full-package or GPU run in this review.
- A nine-case instrumentation probe covered direct, DMRG2, DMRG3 and one-site,
  two-site path, and three-site branching supports on an eight-qubit tree.
  Full canonicalization and dense state/operator readout were forbidden during
  replay. All cases passed numerical canonical and isometry-metadata checks.
- Direct active operators contained 1, 7, or 9 of the state's 15 nodes.
  Routed installation performed **zero additional QR decompositions**.
  All direct compression calls used `reduced="left"`; path/branch updates had
  6/8 edge calls, respectively. One-site unitaries had zero QR/edge calls.
- Across DMRG probes, contained blocks performed **zero QR decompositions**.
  Block-entry movement remained local; branch traversals still entail more
  center travel than path sweeps.

## Remaining optimization candidate — final branch return

`_compress_subtree` returns by QR after every branch, including the final walk
back to the selected hub after the last cut. This honors the current final-hub
policy but is not a minimum-movement requirement of the completed state.

A temporary monkeypatch omitted only those final return moves and recorded the
last compression destination as the center. Across **24** normalized cases
(NumPy/Torch CPU complex128; two arities; three seeds; chi=2/3; support `(0,2,7)`),
it saved **three QR moves** per update. Maximum dense amplitude difference was
**2.424e-16**; maximum norm difference was **5.552e-16**. Numerical canonical and
metadata checks passed. An initial probe used unnormalized random states with
large norms; rerunning on normalized inputs made the absolute comparison
tolerance meaningful without changing it.

This is an experiment, **not an implemented optimization** or a wall-clock
speedup claim. Ending at a different center can change later path orientation
and finite-bond compression trajectories. Any production change needs an
explicit endpoint policy, multi-update validation, and native coverage. There
is no basis here to claim every operation globally optimal, and no confirmed
canonical correctness regression was found.

## Evidence and source map

- Logs: `/tmp/pepsy-tree-routing-canonical-audit-tests.log`,
  `/tmp/pepsy-tree-compact-audit-tests.log`, `/tmp/pepsy-tree-routing-audit.log`,
  `/tmp/pepsy-tree-final-return-probe.log`.
- Scripts: `/tmp/pepsy_tree_routing_audit.py`,
  `/tmp/pepsy_tree_final_return_probe.py`. No experimental code was installed
  into the repository. `git diff --check` passed.
- [Application planning](../src/pepsy/optimizers/tree/_application.py)
- [Replay and direct compression](../src/pepsy/optimizers/tree/optimizer.py)
- [Canonical regions and local proofs](../src/pepsy/optimizers/tree/ttn.py)
- [FIT block preparation and installation](../src/pepsy/fitting/tree.py)

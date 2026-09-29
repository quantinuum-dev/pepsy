# 2026-09-29 — Tree local norms, fidelity, and accumulation audit

- Scope: verify TreeOptimizer's local norm, local fidelity, log accumulation,
  and direct/DMRG modes following the MPS audit and Kraus repair.
- Branch / baseline: `develop`, `b343ced`, including the existing uncommitted
  [tree norm/backend repairs](2026-09-29-tree-norm-backend-fixes.md).
- Commit status: this audit adds only this handoff. No implementation edits,
  staging, commits, or changes to concurrent work.

## Findings

1. With a known canonical center, `_norm_components_backend` contracts just
   that tensor; native fermionic states use the graded center contraction.
   Unknown-center states fall back to the full network. The ledger keeps
   tensor norms and stored exponents separate. Center norms matched dense
   state norms within **1.555e-15** in the 540-case probe.
2. `_diagnostics.norm_event` computes local log fidelity from norm logarithms
   (and exponent differences), adds it to `_norm_log_survival`, and reports
   fidelity with `exp` and infidelity with `-expm1`. Loss is recorded before
   optional working-norm restoration. `norm()` / `state_norm` are actual
   represented norms; the diagnostic field `local_norm` is instead the square
   root of the dimensionless local fidelity ratio.
3. DMRG's terminal projection reproduced explicit normalized target overlaps
   in **432 comparisons**, maximum discrepancy **4.885e-15**. This covers
   `dmrg`, `dmrg1`, `dmrg2`, and `dmrg3` on NumPy/Torch CPU complex128, two tree
   arities, chi=1/2/3, two- and three-site gates, and stabilization on/off.
   It verifies the per-update projection identity, not global optimality of
   finite-sweep FIT or final-circuit fidelity.
4. Direct mode needs a distinction: two-qubit path updates matched explicit
   overlaps, but the retained-norm ratio is **not universally the true
   per-gate fidelity for branching updates**. The branched compression path
   descends edges and returns by QR before visiting another branch. Its
   sequence of orthogonal cuts need not act as one orthogonal projection of
   the original full target. The ledger's documented compression-proxy
   interpretation is necessary here.
5. The cumulative product is not the exact final-state fidelity. With chi=1,
   direct replay of a rotation by 0.4 in the |00>,|11> subspace followed by its
   inverse reports product **0.7197034144** (infidelity **0.2802965856**), while
   its final normalized overlap with the untruncated result is **1**.

## Confirmed direct-mode example

- Six qubits, balanced binary plan, random canonical state with D=3 and seed=1,
  complex128, chi=3, cutoff=0, Haar-style QR unitary on `(0, 2, 5)`.
- Reported local fidelity: **0.9521803199771247**.
- Explicit normalized target overlap: **0.9518767535824942**.
- Difference: **3.0356639463e-4**, reproduced with NumPy and Torch and with
  stabilization disabled as well as enabled. The state remained canonical.
- Increasing direct chi to 64 gives the explicit dense target within
  **6.275e-16** maximum amplitude error, independently checking gate routing.
- The unnormalized residual `<output|target> - ||output||^2` is approximately
  `-1.5179620530e-4 - 4.1590991888e-5j`, so the projection identity needed to
  infer true gate fidelity from output norm alone does not hold in this case.
- A follow-up NumPy direct probe had zero discrepancies above 1e-10 in 36
  two-site updates and six discrepancies in 18 three-site branching updates.
- This is distinct from the previously recorded Torch FIT trajectory
  sensitivity: it is a single-update direct-compression metric discrepancy.

## Validation and reproduction

- Fresh CPU suites: `test_optimize_tree.py`, `test_tree_unitary_stability.py`,
  `test_tree_fit_messages.py`, `test_tree_fit_priorities.py`, and
  `test_tree_gpu_backend.py`: **400 passed, 64 skipped**, 6 warnings, 90.74 s.
  These include the earlier exponent/backend regressions and local-norm
  checks; they do not require every branching direct ratio to equal overlap.
- Temporary probe: `/tmp/pepsy_tree_norm_fidelity_audit.py`, log
  `/tmp/pepsy-tree-norm-fidelity-audit.log`: 540 comparisons, with canonical
  validation after every update and independently applied dense gates.
- Follow-up: `/tmp/pepsy_tree_direct_fidelity_scope.py`, log
  `/tmp/pepsy-tree-direct-fidelity-scope.log`; concrete gate/input/output/target
  arrays saved at `/tmp/pepsy-tree-direct-fidelity-reproducer.npz`.
- To reconstruct the specific gate without the temporary arrays, start
  `rng = np.random.default_rng(1)` and iterate chi over `(1, 2, 3)`. For each
  chi, draw complex square matrices of dimensions `(4, 4, 8)`, forming each as
  `rng.normal(size=(d,d)) + 1j*rng.normal(size=(d,d))`, and take the Q factor
  of `np.linalg.qr`. The ninth Q is the example gate. Construct the state with
  `TreeTensorNetwork.rand(plan, D=3, seed=1)`, cast to complex128, and physically
  normalize with `TreeOptimizer.normalize()` before application.
- `git diff --check` passed. No new full-package or GPU execution. Fresh
  overlap probes are dense NumPy/Torch; native symmetry fidelity equivalence
  is not established by them.

## Follow-up

If exact per-gate fidelity is required for branching direct updates, investigate
an explicit target-overlap readout or a compression schedule preserving the
required projection identity. No algorithm or diagnostic contract was changed
as part of this check.

## Source map

- [Norm event and log arithmetic](../src/pepsy/optimizers/tree/_diagnostics.py)
- [Canonical norms and branching compression](../src/pepsy/optimizers/tree/optimizer.py)
- [TreeFIT local projection and diagnostics](../src/pepsy/fitting/tree.py)
- [Tree replay documentation](../docs/api/optimizers/tree_replay.md)

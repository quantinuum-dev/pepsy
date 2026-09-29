# 2026-09-29 — Tree DMRG step-by-step review

- Scope: user requested another careful review of TreeOptimizer DMRG and
  whether its steps are correct and optimal for roughening.
- Baselines: Pepsy `develop` / `f1b9924`, examples `main` / `1021497`.
- Status: review only, with this uncommitted handoff added. No implementation,
  settings, production processes, or saved production data changed. Concurrent
  operator/PEPO edits in Pepsy and existing plotting notebook edits in examples
  are outside this review and were preserved.

## Conclusion

No correctness defect was found in the reviewed default dense DMRG path.
The local least-squares projection and two-node SVD behaved correctly in the
new checks. This does not establish global optimality at finite chi, production
convergence, or best GPU performance. Two concrete performance opportunities
remain; neither requires changing the physical evolution or cutoff policy.

## Findings

1. **P3 — Scalar host reads remain in the default GPU path.**
   `TreeOptimizer.apply_sub_mpotree` obtains the target norm through
   `TreeFIT._center_norm_stripped` before each multi-node fit
   (`optimizer.py:5955`). TreeFIT reads another terminal centre norm each
   completed iteration (`tree.py:2184`); `_scalar_value` converts it to NumPy
   and a Python number (`tree.py:339`, `tree.py:1528`). These paths can
   synchronize a CUDA stream even when timing, overlap diagnostics, and finite
   checks are disabled. Adaptive convergence needs a decision, but the
   pre-fit read and repeated one-site gate unitarity checks are separate
   optimization candidates. One-site certification uses `allclose` and a host
   bool for each application (`optimizer.py:5803`). Any reuse must preserve
   the current protection against callers mutating a supplied gate.

2. **P3 — The unchanged-exterior identity shortcut excludes Torch.**
   `TreeFIT._identity_environment` explicitly requires NumPy arrays
   (`tree.py:1049`). Torch uses the correct full directed-message contraction.
   The matched frontend probe recorded 384 identity shortcuts for NumPy and
   zero for Torch. A backend-independent proof based on unchanged tensor
   ownership/gauge metadata is a candidate improvement; merely adding
   `torch.equal` would introduce another synchronization. No production GPU
   speedup was measured or inferred from these counts.

3. **Iteration convergence is not a global optimum guarantee.**
   Roughening's eight iterations each contain two passes. Generic DMRG on
   multi-node paths uses two two-node growth iterations and then one-node
   refinement, not a continuously rank-adaptive growth phase
   (`optimizer.py:3301`). Patience counts two stable norm comparisons and
   resets after the block-size change. Of 16 finite-chi random stress fits,
   six reached the eight-iteration budget without declaring tolerance
   convergence. Another sixteen one-node iterations reduced squared residual
   by at most 1.45e-9 in this particular sample. This is not evidence that eight
   is sufficient for every production window, nor evidence of a serious
   convergence problem in the tested sample.

Recommendation: preserve the current numerical defaults while profiling a
representative GPU case; prioritize duplicate host decisions and a safe Torch
exterior proof. Change iteration or growth policies only with matched accuracy
comparisons. The previously documented optional R/L CLI incompatibility and
stale thread-count wording in the performance skill remain outside the default
RL path; actual optimizer/sampler defaults are None.

## Steps checked

| Step | Reviewed behavior |
| --- | --- |
| Physics and layout | All 12 sweep children resolve the same diagonal x+y wall, snake logical labels, and alternating-xy tree geometry. |
| Target construction | An exact layered operator/state target is distinct from the disposable guess; target bonds are private. |
| Initialization | Default guess-src is capped independently; strength zero adds no extra random perturbation, while SRC itself remains seeded randomized compression. |
| Canonicalization | Only the exterior of the local block is prepared; a centre already inside the block needs no redundant interior QR. |
| Local solve | One-node projection and two-node capped SVD solve the local least-squares problem for the fixed canonical exterior. |
| Environments | Cached directed messages and invalidation agree with independently contracted complete branches. |
| Traversal | Endpoint order is frozen before the guess; inward/outward sweeps reverse both window order and final centres. Branched regions use depth-first traversal. |
| Truncation | chi, cutoff 1e-12 and rsum2 reach local splits in complex128. |
| Stopping | Norm-change tolerance 1e-9, minimum two iterations, patience two; phase transitions reset stopping history. |
| Stabilization | Incoming norm is restored after recording loss; saved physical states retain unit norm in the frontend reference check. |
| Runtime controls | No implicit CPU thread cap; retained histories, profiling, finite/overlap scans, and bond scans are disabled. |

## New validation

Existing Python 3.12 environment, local sources, CPU-only subprocesses with
local OMP/MKL/OpenBLAS limits. These limits are test setup, not runner defaults.
Numerical dependency versions match the preceding implementation audit;
installed Pepsy metadata now reports 0.5.0, so the earlier 0.4.0 metadata
mismatch is no longer observed. This review did not reinstall anything.

- Focused FIT, messages, traversal, parity, norm/stabilization and optimizer
  selection: **159 passed, 191 deselected**, 41.26 s. Log:
  `/tmp/tree-dmrg-review-tests-20260929.log`.
- **16 finite-chi dense fits**, NumPy/Torch CPU, path/branched supports,
  both pass orders, and four seeds: **1528 individual local updates** checked
  against the exact dense target. Maximum increase in squared residual was
  1.11e-16. Cached versus complete-branch-reference states differed by at most
  1.38e-15. Twelve additional partially identical fused-target cases differed
  by at most 1.32e-11 after iterative sweeps, with 50 identity shortcut uses.
  Script `/tmp/audit_tree_dmrg_numerics_20260929.py`; detailed output
  `/tmp/tree-dmrg-numerics-review-20260929.json`.
- **12 angle identities** checked for the explicit 5x6, chi512, depth200,
  dt0.05, sample8192 comparison command without launching it.
- Actual frontend exact comparison on **3x2, chi64, two steps**, all 12 angles,
  NumPy and Torch CPU: **72 states**, minimum normalized fidelity
  0.9999999999987095 and maximum unit-norm error 1.34e-15. All **336 multi-node
  FIT calls** used `(2,2,1,1,1)` and stopped after five iterations. These are
  small, effectively untruncated reference tests. They do not validate 5x6
  finite-chi long-time accuracy. Script:
  `/tmp/audit_tree_roughening_final_20260929.py`; saved summary:
  `/tmp/tree_dmrg_review_20260929_y6hcn2zl/summary.json`.

The earlier [runtime alignment handoff](2026-09-29-tree-runtime-alignment.md)
records the broader 1405-test Tree suite and 475-test examples suite. Those
were not rerun wholesale for this read-only review. No production GPU timings
or large-lattice convergence claims are made.

# 2026-09-29 — Review of the latest MPS and tree optimizer commits

- Scope: review the latest Pepsy commits, especially MpsOptimizer and
  TreeOptimizer; no implementation fixes were requested.
- Branch / baseline: `develop` / `b343ced`; the initial working tree was clean.
- Reviewed implementation changes: `f1b9924` and `b343ced`. The intervening
  square-cluster commit `4e7d08e` is outside this optimizer review.
- Commit status: this review handoff is uncommitted. No implementation files
  changed, and nothing was staged, committed, or published.

## Confirmed findings

1. **P1 — Torch DMRG fails when its incoming norm cannot use the ledger
   shortcut.** In `TreeOptimizer.apply_sub_mpotree`, the fallback target norm
   added by `b343ced` calls `ar.do("maximum", torch_scalar, 0.0)`. Installed
   Autoray dispatches to `torch.maximum`, which requires a Tensor second
   argument. A four-site Torch tree and a CZ on `(0, 3)` raise
   `TypeError: maximum(): argument 'other' (position 2) must be Tensor, not float`
   with `track_infidelity=False` and default stabilization. A nonzero state
   exponent (even `1.0`) also triggers this with default tracking enabled.
   Both cases pass against the preceding `4e7d08e` source in the same
   environment. The disabled-tracking failure also reproduces with complex64
   in `dmrg`, `dmrg1`, `dmrg2`, `dmrg3`, and `mix`.
   Use backend-compatible clipping or a same-backend zero;
   cover disabled tracking and nonzero exponents separately.

2. **P2 — Unitary stabilization rejects valid extracted-scale states.** The
   stabilization added by `f1b9924` computes its ratio from represented host
   norms. With finite unit-norm tensor data and `tn.exponent=400` or `-400`,
   those norms become infinity or zero. A unitary CZ then raises
   `FloatingPointError` in direct mode with `stabilize_unitary=True`, despite
   valid working tensors. The same updates succeed with stabilization off.
   The problem is also present in `4e7d08e`; it is separate from the latest
   Torch regression. Compute the restoration ratio from stripped norms and
   exponent differences, avoiding materialization of the common scale.

3. **P3 — Internal MPS skills retain the previous DMRG2 default.**
   `.github/skills/mps-optimizer/SKILL.md` and
   `.github/skills/tensor-fitting/references/fit-architecture.md` still say
   adjacent DMRG2 windows automatically take one update. Current implementation,
   public API documentation, and changelog correctly require the explicit
   `fit_single_pair_fast_path=True` shortcut and otherwise inherit `n_iter`.
   This mismatch was already noted in the earlier
   [FIT-budget review](2026-09-29-mps-fit-budget-review.md).

The Torch exterior-identity proof checks shared non-differentiable arrays,
matching leg order, and canonical structure. No additional correctness defect
was identified in that proof or in the MPS budget forwarding during this review.

## Validation

- Existing `py312` development environment, CPU only; CUDA hidden and numerical
  CPU thread limits set only in validation subprocesses. No environment changes.
- Public API and package layout: **53 passed**; repository Ruff: **passed**.
- Broad MPS/tree/trajectory domain selection (`test_optimize_mps.py`,
  `test_mps_*.py`, `test_optimize_tree.py`, `test_tree_*.py`, and
  `test_trajectory_noise.py`, with default smoke filtering disabled and
  `not slow and not benchmark`): **2269 passed, 78 skipped, 33 deselected**
  in 322.46 s. Skips cover GPU/Metal availability and JAX x64 requirements.
  This suite does not cover the two failing configurations demonstrated above.
- `git diff --check` and the new journal's local-link/whitespace checks pass.
- Extra MPS checks: all four DMRG modes accept adjacent cap `1` with both
  fixed and automatic stopping, perform one iteration, and produce the expected
  normalized Bell state.
- Additional cached-versus-full-exterior circuit probes used six-site trees,
  four mixed-support unitary gates, complex64/complex128, chi 1/3, and
  DMRG2/DMRG3. Seven of eight final vectors agreed within `9.2e-16`. The
  complex128/chi3/DMRG3 case differed by `2.06e-4` in maximum amplitude
  (mutual infidelity `5.05e-7`). A follow-up compared every one of its **144
  local effective tensors** against fresh full contractions on the same live
  fitted state: maximum difference **9.24e-16**, with canonical checks passing.
  Thus the probe found sensitivity of the finite-iteration trajectory to tiny
  contraction differences, rather than evidence of an incorrect local identity
  proof. The precise source of that amplification was not established; no
  universal finite-chi equivalence claim is made. Scripts:
  `/tmp/pepsy_tree_review_differential.py` and
  `/tmp/pepsy_tree_review_local.py`.
- Reproducer: `/tmp/pepsy_optimizer_review_probe.py`. Previous source extracted
  to `/tmp/pepsy-review-parent-4e7d08e`; both versions used the same interpreter
  and dependencies. Domain log: `/tmp/pepsy-review-20260929-tests.log`.

## Upstream audit and limits

Installed: Pepsy 0.5.0, Quimb 1.15.1.dev66+ge927f06e1, Autoray
0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1,
Symmray 0.4.1.dev7+g83fb22865, Torch 2.6.0+cu124, JAX 0.10.2.
Inspected installed Tensor.split, TensorNetwork.copy and TreeFIT.run_gate
signatures, plus Autoray like-array identity creation (Torch dtype preserved).
Consulted the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The Symmray abelian
array documentation URL failed to load.

**Adopt:** backend-compatible scalar bounds for the target norm.
**Defer:** dependency changes and implementation fixes pending a repair task.
No compatibility shim or installed-library edit was made. Full-package and
GPU validation are not claimed. Finite-chi convergence is not a proof of a
global variational optimum.

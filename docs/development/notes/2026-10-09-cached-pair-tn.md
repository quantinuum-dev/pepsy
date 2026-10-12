# 2026-10-09 — Cached full-pair TN objectives and fixed-chi defaults

## Implemented

This supersedes the dense full-pair implementation in the earlier
[pair L-BFGS note](2026-10-09-pair-lbfgs.md). It follows the
[Quimb/scaling audit](2026-10-09-quimb-pair-autodiff-audit.md), reusing that
audit's installed versions and upstream findings.

- Two-site full update defaults to reduced ALS, `gauge=False`, one scalar
  boundary cap `2*D**2`, and cached DMRG boundary MPS (`fit_mode="dmrg"`).
  Normalization/evaluation inherit this cap unless explicitly overridden.
  Boundary-convergence probes and automatic evaluation retries default off.
  Constructor aliases and `run(update_style="two-site")` resolve consistently.
  Row/column defaults and explicit convergence/diagnostic options remain available.
- The FU engine completes the gate queue without independent global pre/post
  fidelity checks or outer candidate acceptance by default. It records local
  pair fidelity and `accumulated_local_fidelity` (the product of the local
  estimates), alongside accumulated local infidelity. The product is not an
  exact whole-circuit fidelity. Explicit diagnostic/acceptance options remain
  available, and row/column/global defaults retain their existing checks.
- The local SVD initializes the pair once. If its numerical support fits the
  requested PEPS cap, accept it directly with local fidelity one and no FU
  environment or solve. Machine precision determines rank independently of
  ALS/L-BFGS tolerance. Requested normalization is preserved. `skip_exact=False`
  exposes the full solver for exact warm starts. This is the local-SVD variant
  of initialization; persistent SU bond weights are not introduced.
- Reduced tensors alone form a dense pair norm matrix, with the existing
  Hermitian/PSD treatment. Their optional analytic-gradient L-BFGS remains.
- Full-tensor L-BFGS retains the exterior as a TN, snapshots the fixed arrays,
  and keeps the original two tensors plus gate as the target. Cotengra builds
  scalar norm and overlap expressions and folds constant-only contractions
  once per local solve. The target norm is measured once. Both site arrays
  vary through the same `GradientOptimizer` infrastructure used by sweep fits.
- Torch autodiff supplies complex gradients; CuPy inputs contract through
  Torch on the same GPU and are restored to CuPy. SciPy parameter/gradient
  vectors still travel through host memory. Differentiating through a complete
  update and native symmetry arrays remain outside this path's contract.
- The objective is normalized squared residual. Full mode performs scalar
  validity checks but no PSD projection or environment gauge. Invalid finite
  boundary estimates are rejected; the warm start is retained if the returned
  candidate is worse. Reduced PSD and full raw-TN metrics need not coincide.
- Existing version-checked transverse and strip caches handle reuse across
  gates. New gates rebuild local objective constants, preventing stale gate
  or environment values. A shared boundary-retuning fix avoids repadding
  canonicalized boundaries already at their geometric cut capacity at an
  unchanged requested cap, which had invalidated otherwise reusable DMRG cuts.
  Lower-rank compressed guesses with room to grow still expand; explicit cap
  growth/shrink remains supported.

## Validation

Tests compare scalar values and complex gradients with exact physical-state
contractions, forbid expression rebuilding during repeated evaluations, check
constant snapshots and gate changes, compare cached/fresh multi-gate results,
and prohibit dense pair norm construction/PSD projection in a bulk D=4 fit.
Default-policy tests verify real DMRG cache hits and prevent convergence probes,
including when two-site mode is chosen only at run time. D=1,3,4 distinguish
the new `2*D**2` rule from the former linear cap.

Final suite results are recorded in the linked
[session handoff](../../../history/2026-10-09-pair-lbfgs.md).

## Cost limits

Avoiding explicit full pair norm storage removes the mandatory D**12 matrix
and D**18 dense eigendecomposition. Scalar contraction costs still depend on
D, boundary chi, geometry and path, and autodiff retains intermediates. The
prior topology-specific scaling audit remains useful; it is not a universal
complexity guarantee. No claim that L-BFGS is always faster or more accurate
than reduced ALS is supported by these correctness tests.

# 2026-10-10 — Optional one-site L-BFGS in cached PEPS sweeps

- Scope: use L-BFGS instead of CG for each tensor inside the existing
  alternating row/column fits.
- Branch / baseline: `develop`, `b3e7985`; preserve the earlier uncommitted
  [ALS review](2026-10-10-peps-als-review.md).
- Commit status: working-tree changes only; nothing staged, committed or pushed.

## Implemented

`optimizer="als"` accepts `linear_solver="dense-lbfgs"` for explicit Hermitian
N, or `"lbfgs"` for matrix-free actions. The default remains `"dense-cg"`.
The new [_als_lbfgs.py](../src/pepsy/optimizers/sweep/_als_lbfgs.py) minimizes
the local quadratic with its analytic real/imaginary-coordinate gradient,
using SciPy L-BFGS-B. Existing directional cursors and matrix-free environment
halves are reused. No dense N is built for `"lbfgs"`.

Public GradientOptimizer was inspected: its host-problem interface expects
autodiff/finite-difference objectives rather than an analytic value/gradient
callback. This focused kernel uses public SciPy `minimize(jac=True)` and the
existing backend converter, without adding a general solver framework or
duplicating L-BFGS internals. Contractions stay on NumPy/Torch/CuPy; only
site-sized parameter/gradient vectors use host storage. Non-NumPy callers
receive a warning once per owner. Dtype/device/exterior arrays are preserved.

Controls: `lbfgs_maxiter=100`, `lbfgs_history=10`, `lbfgs_maxls=20`, and
`lbfgs_rtol=None` (1e-6 double, 1e-4 single). Budget/line-search termination
returns the best finite evaluated quadratic candidate with an honest residual
and convergence flag. Original-overlap acceptance can retain an improving
unconverged iterate. There is no automatic fallback or spectral projection.
Driver summaries include local L-BFGS solve/evaluation counts.

## Validation

Independent finite differences validate real and complex analytic gradients;
small SPD systems are compared with independent direct solutions. Tests cover
both axes/methods, NumPy/Torch CPU/Torch CUDA/CuPy, dtype/device and frozen
exteriors, site-only host transfers, exact statevector fidelity, finite budgets,
controls, network exponents, and PepsOptimizer gate targets with enlarged bonds.
Single-precision exponent-invariance checks use a 1e-5 fidelity budget: float32
L-BFGS line searches can stop at iterates differing by a few 1e-6 under large
overall exponent shifts. Reference statevectors remove global exponents to
avoid overflowing float32; normalized fidelity is scale invariant.

Final targeted regression: 464 passed, 66 warnings in 66.32 seconds, including
120 ALS tests across the three ALS test modules. The run also covered sweep
safeguards/performance, PEPS optimizer batching, optimize_peps, public API, and
package layout. Ruff (`src tests`), `git diff --check`, and related documentation
link checks passed. The full repository suite was not run.

## Compatibility and limits

Reused the same-task upstream audit and inspected installed SciPy 1.17.1
`minimize` signature plus current official L-BFGS-B documentation. SciPy is
imported lazily; no dependencies or installed libraries changed. Classification:
adopt public SciPy and existing backend conversion. No full-suite or trajectory
benchmark was run, and no speedup over CG is claimed. Matrix-free evaluation
retains O(D^10) at chi=D^2; explicit N still incurs O(D^12) assembly. See the
[updated cost note](../docs/development/notes/2026-10-10-sweep-als-costs.md).

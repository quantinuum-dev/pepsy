# 2026-10-10 — One-site sweep ALS: locality, linear solvers, and costs

This audit concerns `PepsOptimizer(mode="sweep", optimizer="als")`, not reduced
two-site FU or the notebook's separate exact whole-PEPS experiment. See the
[implementation](../../../src/pepsy/optimizers/sweep/_als.py) and
[public controls](../../api/optimizers/sweep.md#als-within-a-slice).

## Final solver policy

The requested default is **an explicit local norm matrix with an iterative
solve**, not matrix-free ALS and not direct factorization:

1. Fix a row/column and its outside boundary MPS.
2. Assemble the active tensor's norm and overlap environments using the
   updated prefix and cached suffix.
3. Form `N_H = (N + N.H) / 2`, then solve `N_H A = b` using CG.
4. Accept only a converged, finite candidate that does not worsen the original
   boundary-estimated normalized overlap; otherwise retain the previous tensor.
5. Move to the next tensor. Rebuild directional caches on reversal, then
   finish the requested inner trips before advancing the outer strip.

`linear_solver="dense-cg"` is the default. Matrix-free `"cg"`, direct `"dense"`,
and spectral positive-support `"pinv"` are explicit alternatives. There is no
default SVD, eigendecomposition, direct factorization, shift, or fallback.
CG has a diagonal preconditioner, a 200-iteration default cap, and an explicit
true-residual check. Optional relative shifts and guarded direct fallback
are documented. Hermitianization alone does not restore positive definiteness;
nonpositive curvature/unconverged solves are rejected. This follows the
[CG operator requirements](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html),
although execution uses native Autoray arrays rather than SciPy CG.

## Arithmetic counts

Assume a bulk square-lattice site, uniform PEPS bond D, fixed physical
dimension d, realized boundary rank chi, and target ranks O(D). Let L be the
strip length, r the inner round-trip count, and k the CG iteration count.
These counts exclude outer boundary construction and contraction-path search.

The tensor has d D^4 entries. The norm acts only on its four virtual legs,
so it is an n-by-n matrix with n=D^4 and d independent physical RHS columns.

| Operation | Work | With chi=D^2 |
| --- | --- | --- |
| Cached prefix/suffix advance | chi^3 D^4 + d chi^2 D^6 | D^10 |
| Explicit norm construction | chi^3 D^4 + chi^2 D^8 | D^12 |
| Hermitianization | D^8 | D^8 |
| k CG iterations with explicit N | k d D^8 | k D^8 |
| Direct factorization or eigensolve | D^12 | D^12 |
| k matrix-free CG iterations, including setup | k (chi^3 D^4 + d chi^2 D^6) | k D^10 |

Thus the default site's cost is O(D^12 + k D^8), and r round trips of a strip
cost O(r L (D^12 + k D^8)). The D^12 term is **matrix construction**, not the
iterative solve. No inverse is explicitly computed. Edges/corners have fewer
virtual legs and smaller systems. Caching makes traversal linear in L; it
does not remove the final contraction needed to assemble each new matrix.

For the construction count, the site hole is a ring of four tensors, each
with two chi legs and two open D legs. Contracting two adjacent pairs costs
2 chi^3 D^4. Each pair has shape D^4-by-chi^2 after grouping indices. Joining
the pairs into N costs chi^2 D^8. This is D^12 when chi=D^2, even though the
output itself has only D^8 entries.

The matrix-free implementation caches those two halves and absorbs a trial
tensor before joining them. It never materializes their full product.
Hermitian actions average N x and N.H x. Temporary halves/intermediates can
still have O(chi^2 D^4) entries: O(D^8) memory at chi=D^2. Explicit N also
takes O(D^8) storage. Directional caches add O(L chi^2 D^2), excluding global
boundary stores, PEPS arrays, and backend workspace. At D=4, explicit N is
256-by-256 (1 MiB for complex128).

## Whole-column L-BFGS comparison

The existing Torch sweep objective contracts scalar norms and overlaps and
uses reverse-mode autodiff. It does not build N. With suitable contraction
paths and fixed boundaries, a column objective plus gradient costs
O(L [chi^3 D^4 + d chi^2 D^6]), hence O(L D^10). L-BFGS iterations and
line-search evaluations multiply this cost. Its vector/history algebra is
lower order for a fixed history length. The D^12 explicit-ALS construction
is not shared by this scalar-objective method.

Current `SweepOptimizer._params_require_finite_differences` routes non-Torch
parameter backends, including NumPy/CuPy, to finite differences. Those paths
multiply scalar evaluation work by the parameter count and do not have the
autodiff gradient cost above. This audit does not change gradient routing.

Outer boundary construction depends on compression policy. See the
[separate boundary audit](2026-10-09-fu-boundary-costs.md); its one-site DMRG,
two-site, and direct-compression costs must not be conflated with a fixed
strip objective. The standard scalar boundary contraction count also appears
in [Lubasch et al., section III.1](https://arxiv.org/html/1405.3259).

## Evidence and limitations

Dimension-only Cotengra searches on the explicit four-tensor ring at D=2, 4,
8, 16, 32 with chi=D^2 reproduced `2 chi^3 D^4 + chi^2 D^8` multiplication
units. At D=4 this is 18,874,368 units. The RHS and prefix advance each gave
`2 chi^3 D^4 + 2 d chi^2 D^6` (6,291,456 units for d=2). A scalar site norm
added d D^4. Greedy paths can be worse; the iterative ALS scalar/cursor kernels
use optimal small-network paths, while matrix-free actions use a fixed path
that absorbs the trial tensor before joining environment halves.

Focused regressions compare every interior update in both orientations and
both boundary engines against independent dense least squares, including
freshly changed neighbors. They check frozen exterior/target/boundary arrays,
native CPU/GPU arrays, rank deficiency, exponent scaling, fallback guards,
and that the default calls no factorization or eigendecomposition.

A NumPy complex128, one-thread local-kernel timing on a 3-by-3 random PEPS at
D=4, chi=16 gave approximately 11.4 ms for explicit construction plus direct
factorization. Matrix-free CG took approximately 1.4 seconds at 500 iterations
and did not meet its residual tolerance. This is a deliberately limited
conditioning example, not a 4-by-4 trajectory or an ALS/L-BFGS speed ranking.
It illustrates why a lower power per iteration does not guarantee fewer
seconds. Temporary reproductions were saved under `/tmp/pepsy_als_*benchmark*`.
The attempted 4-by-4 sweep probe failed the existing local-environment validity
gate before reaching its requested interior solve, so it supplied no timing.
No long-time notebook data were changed.

## Follow-up: one-site L-BFGS alternative

`linear_solver="dense-lbfgs"` uses explicit N and `"lbfgs"` uses the cached
matrix-free action. Both minimize the same local quadratic with exact
real/imaginary-coordinate gradient `2 (N_H A - b)` and retain one-site
forward/backward traversal. The default remains explicit-matrix CG.
For q objective/gradient calls the costs are O(D^12 + q D^8) and O(q D^10),
respectively, at chi=D^2. This differs from whole-column joint L-BFGS.

The implementation reuses public
[SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
with `jac=True` and the existing backend converter. It avoids finite-difference
gradients on non-Torch states. Site parameter/gradient vectors cross the host
boundary; environments and contractions do not. A finite budget can return an
unconverged candidate, with true residual/convergence status recorded and a
separate original-overlap acceptance check. No speed ranking is claimed.

# Equality-constrained ALS for PEPO DMT

This investigation compares feasible-seed refinement with alternating
equality-constrained least squares and SLSQP for a fixed single-layer
environment. It does not establish a general protected-SVD solution for a
loopy PEPO. See the [4×5 TFIM experiment](2026-11-17-pepo-tfim-4x5.md) for
the model, regional maps and independent state-vector reference.

## Method

For a fixed feasible rank-$\chi$ pair $A=XY$ and frozen linear map $C$, seek
to reduce $\|A-M\|_F$ subject to $C\operatorname{vec}(A-M)=0$. Use the SVD
of $Y$ to write its row space as an orthonormal $Q$; then $A=ZQ$, and for fixed
$Q$ minimizing the pair error is equivalent to minimizing

$$
\|Z-MQ^\dagger\|_F^2,\qquad
{\color{blue}C\operatorname{vec}(ZQ)=C\operatorname{vec}(M)}.
$$

The half-step projects $MQ^\dagger$ onto this affine constraint using least
squares. The next half-step mirrors the procedure for the other factor. Each
candidate must be finite, satisfy every scaled regional residual, and strictly
improve the local pair error. A final check against the original regional maps
can reject the complete refinement and retain the feasible seed. Each fixed
factor subproblem is a constrained least-squares projection; the alternating
rank-constrained problem is nonconvex and has no global-optimum guarantee.
SVD is used to choose the factor basis; it is not a general protected SVD.

## Same-source lossy split

After one 4×5 TFIM timestep, use the same PEPO source and exact post-gate
target for all three methods; chi=4, boundary chi=8, one protected L.
Absolute local errors are independently contracted against the untruncated
target. Pair errors are in the current tensor gauge.

| Method | Pair error | Protected L error | Worst of 48 L errors | Time |
| --- | ---: | ---: | ---: | ---: |
| Feasibility seed | 1.62e-4 | 3.18e-10 | 7.31e-5 | 0.039 s |
| ALS, up to 4 sweeps | 9.48e-5 | 8.15e-16 | 2.42e-5 | 0.040 s |
| SLSQP | 5.30e-5 | 9.33e-11 | 8.34e-6 | 0.082 s |

This is one case, not a general ranking. The pair error is gauge-dependent;
the regional errors are independent physical contractions. Run:

```bash
OPENBLAS_NUM_THREADS=1 python examples/pepo_dmt_ising_4x5.py --compare-refinements
```

On a complex synthetic constrained matrix, ALS reduced the local error from
17.64 to 3.94 with rank at most two and constraint residual 5.3e-15. A lossy
2×3 PEPO regression separately checks reduced-density preservation and local
error improvement.

## Two-step trajectory

On the same 4×5 example, ALS completed two steps in 13.3 s with 49 accepted
factor updates. The maximum per-gate computed-map residual was 6.2e-9. The
selected and worst L errors against the exact gate trajectory were
9.4e-5 and 0.0261. Feasibility-only replay took 8.8 s and yielded 5.4e-5 and
0.0136 in that run. Thus **lowering each local pair objective did not improve
these accumulated physical errors**. Dense SLSQP had not completed the second
step when interrupted after 83 s; no full-trajectory comparison is claimed.

Finite boundary maps, one protected L, explicit map memory, physicality and
small sample size all limit these results. A useful next step is to investigate
gauge-aware or physical regional objectives and matrix-free constraints, while
checking selected and unselected densities and whole-trajectory accuracy.

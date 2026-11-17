# Weighted regional objectives for PEPO DMT

This prototype keeps requested local densities as hard constraints and uses
other local reduced densities as a weighted objective, with single-layer PEPO
boundary maps. In the current pair coordinates it searches for

$$
\min_{\operatorname{rank}A\leq\chi}\;
\sum_{R\in\mathcal S}w_R\|\mathcal T_R(A-M)\|_F^2,
\qquad {\color{blue}\mathcal T_W(A-M)=0\quad\forall W\in\mathcal P}.
$$

The selection of geometrically local information is related to
[lattice DMT](https://arxiv.org/pdf/2610.40776), but that paper uses a
vectorized MPO/MPS and a protected-subspace truncation scheme. This weighted
PEPO optimization is a separate proposal, not an implementation of its rule.

## Solver

The optimizer first finds a feasible pair as before. ALS then computes
factor-space corrections in the numerical nullspace of the protected maps,
using a thin SVD and an iterative least-squares solve for weighted regions.
SLSQP can also optimize the weighted objective. Each accepted candidate must
pass all original hard-region residual checks and decrease the weighted cost.
No double-layer PEPO norm is constructed.

An initial single-soft-site probe exposed near-null directions that produced a
large pair error while nearly eliminating the soft objective. The implementation
therefore defaults to a configurable pair-error growth limit of two relative to
the feasible seed, plus backtracking for ALS. This is a stability safeguard,
not a mathematical guarantee or part of the displayed unconstrained objective;
it can be disabled explicitly. Cost fields use the PEPO mantissa and expose
the logarithmic exponent separately.

## 4×5 TFIM test

The model, state-vector reference and hard central L are described in the
[TFIM note](2026-11-17-pepo-tfim-4x5.md). The weighted objective adds two
unprotected nearest-neighbor pairs: `((0,0),(0,1))` with weight 1 and
`((2,2),(2,3))` with weight 2. They are only two selected regions, not all
local densities. The PEPO uses chi=4, boundary chi=8 and direct contraction.

For the **same first lossy split** from the same source, independent physical
regional contractions gave:

| Method | Weighted cost | Protected L error | Worst of 48 L errors |
| --- | ---: | ---: | ---: |
| Feasible seed | 4.24e-9 | 3.18e-10 | 7.31e-5 |
| Weighted ALS | 2.22e-19 | 3.18e-10 | 2.09e-5 |
| Weighted SLSQP | 2.61e-10 | 4.09e-10 | 2.42e-5 |

After two full Trotter steps, weighted ALS had a selected-L error of 1.27e-4
and worst-L error of 0.0446 against the state-vector gate reference. The
unweighted pair-error ALS previously gave 9.4e-5 and 0.0261; feasibility-only
gave 5.4e-5 and 0.0136. These are short single runs, not a general ranking.
The weighted run's worst local Hermiticity defect was 0.0494, and the minimum
eigenvalue of a local panel's Hermitian part was -0.00632. Weighted local
closeness and hard local preservation do not ensure positivity or accurate
unselected regions and do not guarantee better long-time dynamics.

Run from the Pepsy root with the configured environment active:

```bash
OPENBLAS_NUM_THREADS=1 python examples/pepo_dmt_ising_4x5.py --weighted-regions --refine-method als --compare-refinements
OPENBLAS_NUM_THREADS=1 python examples/pepo_dmt_ising_4x5.py --steps 2 --weighted-regions --refine-method als
```

The map-memory guard remains a practical bottleneck for many selected regions.
All results use NumPy complex128, one lattice and an approximate frozen boundary;
neither a global optimum nor large-scale performance is claimed.

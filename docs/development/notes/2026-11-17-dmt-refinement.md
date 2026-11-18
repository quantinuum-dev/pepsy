# Optional fixed-border DMT refinement and shared region protection

Implemented as optional accuracy/selection controls; the fast defaults remain
unchanged. See the [public controls](../../api/operators/mpo_dmt.md#optional-improved-bond-truncation).

## Bond objective and acceptance

Rotate the canonical matrix into the protected bases, with ranks $p,q$:

$$
M=\begin{pmatrix}{\color{blue}P}&{\color{blue}B}\\{\color{blue}C}&D_0\end{pmatrix},
\qquad A(X)=\begin{pmatrix}P&B\\C&{\color{purple}X}\end{pmatrix},\qquad
\min_{\operatorname{rank}A\le\chi}\|X-D_0\|_F^2.
$$

The default paper construction reserves $p+q$ factor channels and truncates
the free connected residual. `truncation="optimized"` first keeps that
feasible candidate. A compatible corner obeys $PP^+B=B$ and $CP^+P=C$.
Then, with $r_P=\operatorname{rank}P$,

$$
{\color{purple}X_*}=CP^+B+\operatorname{SVD}_{\chi-r_P}(D_0-CP^+B).
$$

This is the full one-cut optimum at zero cutoff in exact arithmetic. The
native implementation avoids explicit full protected complements. In the
original coordinates its completion baseline is
$MQ_R(Q_L^\dagger MQ_R)^+Q_L^\dagger M$. Singular values below
$\sqrt{D\epsilon}$ times the pivot maximum are not inverted. The original
trace constraints must hold after reconstruction; incompatible or unstable
candidates are rejected. No arbitrary pseudoinverse completion is accepted.

For incompatible borders, alternate rank-$\chi$ factors $A=LR$. With
$E=M-LR$, $G=RQ_R$, $T=I-GG^+$, and $H=Q_L^\dagger L$,

$$
\Delta L=(I-Q_LQ_L^\dagger)E(TR)^+T,\qquad
\Delta R=(I-H^+H)[L(I-H^+H)]^+E(I-Q_RQ_R^\dagger).
$$

Recompute $E$ after each accepted half-step. These are constrained
least-squares updates: $Q_L^\dagger\Delta L=0$, $\Delta L RQ_R=0$,
$Q_L^\dagger L\Delta R=0$, $\Delta RQ_R=0$. Numerical pseudoinverses use
dtype-scaled thresholds, including the parent-factor scale for nominally
zero projected systems. QR stabilizes the gauge. Rank-deficient seeds can
use zero left columns with free residual right rows to activate unused rank.
There are no random restarts or finite constraint penalties.

Every acceptance checks finite error, factor width, the original protected
maps, and nonincreasing Frobenius error. QR allows an $8\epsilon\|M\|_F$
roundoff allowance and must still pass feasibility. The limit defaults to
eight sweeps; zero tests only the SVD seeds. Numerical guards and the full
physical-region checks remain active without reports. No global optimum
certificate is claimed for alternating refinement, nor is improved error
against the initial target guaranteed across different complete sweeps.

## 2D selection as a bipartite cover

For every region $W$, expand operators on $W\cap L$ and $W\cap R$ into
trace-aligned orthonormal physical strings. Connect every pair of nonidentity
strings occurring in that region. A protected left vertex enforces
$f_l^\dagger(A-M)=0$; a protected right vertex enforces
$(A-M)f_r=0$. Consequently each edge's expectation is fixed if either
endpoint is selected. Identity/trace directions are mandatory on both sides,
so all strings with an identity endpoint, and regions wholly on one side,
are fixed as well.

`region_selection="min_cover"` constructs a maximum bipartite matching with
SciPy, then extracts a minimum vertex cover by alternating paths. This
minimizes the total physical-string count for these sufficient one-sided
constraints. It does not optimize state-dependent ranks, arbitrary two-sided
constraints, geometry inventories, or contraction time. Enumeration remains
exponential in region size. Deterministic region ordering fixes tie handling.
Routed gates rebuild covers for their permuted region inventories.

| Plaquette square | Smaller-half strings, summed over cuts | Minimum-cover strings | Peak, both rules |
| --- | ---: | ---: | ---: |
| 3×3 | 148 | 142 | 29 |
| 6×6 | 1795 | 1735 | 65 |
| 8×8 | 4683 | 4557 | 89 |

Counts include trace on both sides of every cut. These are structural
measurements, not timing or required-bond predictions.

## Upstream audit

Reviewed [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray). The linked Symmray
`abelian_arrays.html` page was unavailable; dense-only capability rejection
is retained. Classification: **adopt** existing public `array_split`, Autoray
dispatch and SciPy matching; **defer** native symmetry, loopy PEPO and global
JIT compilation. No dependency upgrades, new shims or installed-library edits.

Installed: Quimb 1.15.0, Autoray 0.8.4, Cotengra
0.8.3.dev8+g8954240f2, Cotengrust 0.2.1, Symmray
0.4.1.dev15+g0374aaa3c, NumPy 2.5.2, SciPy 1.18.0, Torch 2.9.1,
JAX 0.8.2. Inspected `array_split(x, method='auto', absorb='auto',
max_bond=None, cutoff=1e-10, cutoff_mode='rsum2', renorm=None, info=None,
**kwargs)` and `maximum_bipartite_matching(graph, perm_type='row')`.
Native SVD/QR/solve dispatch is present for NumPy, Torch and JAX. All new
linear algebra stays native; graph planning and adaptive decisions use host
data. No fully differentiable/JIT guarantee follows from native dispatch.
Torch uses native `linalg.pinv(A, *, atol=None, rtol=None, hermitian=False,
out=None)` for the constrained solves: differentiating an explicitly factored
complex pseudoinverse failed on its exact null directions in a combined
backend test. The structured pinv derivative passes the protected-observable
gradient check. No in-place array updates are used in the refiner.

## Large-square measurements

Fixed rank-136 positive product mixtures, seed 321, complex128, all physical
plaquettes, default snake mapping, auto cutoffs, two refinement sweeps,
single-thread CPU. Each row is one compression timing with diagnostics;
oracle contractions are outside timing. No statistical speedup is inferred.

| Square / cap | Truncation / selection | Seconds | Relative global HS error | Largest plaquette defect |
| --- | --- | ---: | ---: | ---: |
| 6×6 / 96 | paper / smallest | 2.134 | 0.7732 | 7.12e-15 |
| 6×6 / 96 | paper / min_cover | 2.117 | 0.7659 | 5.15e-15 |
| 6×6 / 96 | optimized / min_cover | 2.974 | 0.5457 | 7.93e-15 |
| 8×8 / 128 | paper / smallest | 7.461 | 0.7226 | 7.89e-15 |
| 8×8 / 128 | paper / min_cover | 7.244 | 0.6985 | 6.07e-15 |
| 8×8 / 128 | optimized / min_cover | 9.857 | 0.3013 | 7.70e-15 |

All outputs obey their requested caps. Optimized splits accepted ALS on 25
cuts of the 6×6 sweep and 51 cuts of the 8×8 sweep. Four and six cuts,
respectively, used Schur seeds without accepted ALS improvement; six fitting
edge splits took the exact path. The lower full-sweep global errors here
are observations, not a guarantee for other targets or region inventories.
Relative global errors use independent product-mixture overlap contractions
and the final canonical center norm; they do not densify the full system.

Reproduce the three methods for each square:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python examples/dmt_refinement_benchmark.py --square 6 8 --refinement-sweeps 2
```

The maintained example was rerun on 6×6 after fixing the complex-gradient
path: global errors and defects reproduce, with single-run times 2.199,
2.119 and 3.023 seconds. The NumPy mathematics was unchanged by that fix.

## Validation

See the [session record](../../../history/2026-11-17-dmt-refinement.md) for
final commands and results. Tests compare a complex rotated compatible
corner to an independent Schur solution, reject incompatible pseudoinverses,
exercise rectangular/singular cases and verify numerical rank and both
constraints. A small cover is checked against exhaustive enumeration.
Physical tests contract noncontiguous densities after individual splits,
full sweeps and gate replay in both storage bases. The simple diagonal
counterexample improves squared error from $0.10$ to $0.01$ at rank two.

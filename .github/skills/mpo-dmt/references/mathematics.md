# DMT constraints, dimensions, and the scope of the SVD optimum

Use this reference to derive the protected split or assess an optimality claim.
Blue marks protected data; purple marks the compressed candidate or objective.
The input is the current post-gate target, not the pre-gate state or the exact
untruncated multi-gate trajectory.

## 1. The full one-bond objective

At a canonical cut, write

$$
\rho=\sum_{\alpha,\beta=1}^{4\chi}M_{\alpha\beta}L_\alpha\otimes R_\beta,
\quad \langle L_\alpha,L_{\alpha'}\rangle_{\rm HS}=\delta_{\alpha\alpha'},
\quad \langle R_\beta,R_{\beta'}\rangle_{\rm HS}=\delta_{\beta\beta'}.
$$

The $4\chi$ support is the maximal adjacent-qubit example. Exact gate
contraction reshapes a two-site MPO tensor into a matrix of size
$(4\chi_L)\times(4\chi_R)$, so the resulting rank is at most
$\min(4\chi_L,4\chi_R)$. Use actual support when smaller. Six sites admit at
most $4^3=64$ orthonormal directions at the middle cut; a literal $4\chi$
support therefore requires $\chi\le16$ there.

Define the small compressed center and its embeddings:

$$
M'\in\mathbb C^{\chi\times\chi},\quad
U_c,V_c\in\mathbb C^{(4\chi)\times\chi},\quad
U_c^\dagger U_c=V_c^\dagger V_c=I_\chi,
\qquad {\color{purple}A=U_cM'V_c^\dagger}.
$$

Both $A$ and $M$ are $4\chi\times4\chi$ in the original virtual bases.
The small $M'$ belongs to new bases
$L_i'=\sum_\alpha(U_c)_{\alpha i}L_\alpha$ and
$R_j'=\sum_\beta(V_c^\dagger)_{j\beta}R_\beta$.
Thus $\rho_c=\sum_{ij}M'_{ij}L_i'\otimes R_j'
=\sum_{\alpha\beta}A_{\alpha\beta}L_\alpha\otimes R_\beta$.
Unused zero directions can be retained or removed.

$$
\Delta M=A-M,\qquad
\boxed{\|\rho_c-\rho\|_{\rm HS}^2=\|A-M\|_F^2.}
$$

Never write $\|M'-M\|_F$: the shapes and virtual coordinates differ.
Every rank-at-most-$\chi$ matrix $A$ admits the displayed factorization,
completing orthonormal columns and padding zero singular values if needed.
Consequently the two programs are equivalent:

$$
\boxed{
\begin{aligned}
\min_{U_c,M',V_c}\;&\|U_cM'V_c^\dagger-M\|_F^2\\
\text{s.t. }&U_c^\dagger U_c=V_c^\dagger V_c=I_\chi,\\
&{\color{blue}F_L(U_cM'V_c^\dagger-M)=0},\\
&{\color{blue}(U_cM'V_c^\dagger-M)F_R^T=0}
\end{aligned}}
$$

$$
\boxed{\min_{\operatorname{rank}A\le\chi}\|A-M\|_F^2
\quad\text{s.t.}\quad
{\color{blue}F_L(A-M)=0},\quad {\color{blue}(A-M)F_R^T=0}.}
$$

These are fixed-outer-basis problems. Their regional constraints can be
stronger than preserving only a requested small-window inventory. A feasible
point need not minimize the objective. Positive semidefiniteness and global
Hermiticity are not additional constraints of these programs.

## 2. Why the two trace constraints preserve local densities

Use six sites $012|345$, so $L_\alpha$ acts on $012$ and $R_\beta$ on $345$.
For matrix units $E_a=|s\rangle\langle t|$, define

$$
\operatorname{Tr}_{01}L_\alpha=\sum_{a=1}^{4}(F_L)_{a\alpha}E_a^{(2)},
\qquad
\operatorname{Tr}_{45}R_\beta=\sum_{a=1}^{4}(F_R)_{a\beta}E_a^{(3)}.
$$

Each map is $4\times4\chi$. As a tensor, $F_L$ has **three legs**:
ket $s_2$ and bra $t_2$, each of size two, and virtual $\alpha$ of size
$4\chi$. Fusing $(s_2,t_2)$ makes the four-dimensional matrix row index.
$F_R$ has the analogous legs on site $3$.

Substitute these definitions into the partial traces of
$\Delta\rho=\sum_{\alpha\beta}\Delta M_{\alpha\beta}L_\alpha\otimes R_\beta$:

$$
\Delta\rho_{2345}
=\sum_{a\beta}\left[\sum_\alpha(F_L)_{a\alpha}\Delta M_{\alpha\beta}\right]
E_a^{(2)}\otimes R_\beta
=\sum_{a\beta}(F_L\Delta M)_{a\beta}E_a^{(2)}\otimes R_\beta,
$$

$$
\Delta\rho_{0123}
=\sum_{\alpha a}\left[\sum_\beta\Delta M_{\alpha\beta}(F_R)_{a\beta}\right]
L_\alpha\otimes E_a^{(3)}
=\sum_{\alpha a}(\Delta M F_R^T)_{\alpha a}L_\alpha\otimes E_a^{(3)}.
$$

Orthonormality proves both directions of each equivalence:

$$
F_L\Delta M=0\iff\Delta\rho_{2345}=0,\qquad
\Delta M F_R^T=0\iff\Delta\rho_{0123}=0.
$$

The right coefficient is $(F_R)_{a\beta}$, not its conjugate. This explains
the transpose. In an explanation or schematic, show that contraction before
introducing virtual projectors.

| Protected three-site window | Containing preserved region | Further trace |
| --- | --- | --- |
| $012$ | $0123$ | site $3$ |
| $123$ | $0123$ | site $0$ |
| $234$ | $2345$ | site $5$ |
| $345$ | $2345$ | site $2$ |

For a general chain cut $b|b+1$, every window $\{j,j+1,j+2\}$ lies in
$[0,b+1]$ if $j<b$, or $[b,N-1]$ otherwise. This cover is specific to that
1D inventory. Mapped 2D regions need their own covering/protection construction.

## 3. Reduce the full constraints to a free-block completion problem

Choose orthonormal columns $Q_L$ spanning $\operatorname{range}F_L^\dagger$
and $Q_R$ spanning $\operatorname{range}F_R^T$. Put
$p=\operatorname{rank}F_L$, $q=\operatorname{rank}F_R$ and complete to unitaries
$W_L=[Q_L,Q_L^\perp]$, $W_R=[Q_R,Q_R^\perp]$.
These $W$'s are square basis rotations, distinct from the thin embeddings $U_c,V_c$.

$$
\overline M=W_L^\dagger MW_R=
\begin{pmatrix}
{\color{blue}M_{11}}&{\color{blue}M_{12}}\\
{\color{blue}M_{21}}&M_{22}
\end{pmatrix},\qquad
\overline A=
\begin{pmatrix}
{\color{blue}M_{11}}&{\color{blue}M_{12}}\\
{\color{blue}M_{21}}&X
\end{pmatrix}.
$$

The linear constraints are exactly equality of this blue L-shape. Unitary
invariance of the norm gives an equivalent full problem:

$$
\min_X\|X-M_{22}\|_F^2
\quad\text{s.t.}\quad
\operatorname{rank}\begin{pmatrix}M_{11}&M_{12}\\M_{21}&X\end{pmatrix}\le\chi.
$$

This is a rank-constrained matrix-completion problem. Plain SVD of $M$ can
change protected entries; plain SVD of $M_{22}$ does not generally enforce
the total-rank bound. Feasibility checks alone certify neither optimum.

### A solvable mathematical special case

If $p=q$ and $M_{11}$ is invertible, define
$S=M_{21}M_{11}^{-1}M_{12}$. Invertible block elimination gives

$$
\operatorname{rank}\begin{pmatrix}M_{11}&M_{12}\\M_{21}&X\end{pmatrix}
=p+\operatorname{rank}(X-S).
$$

For $\chi\ge p$, the exact optimum of this fixed-basis program is therefore

$$
X_* = S+\operatorname{SVD}_{\chi-p}(M_{22}-S).
$$

The optional `truncation="optimized"` route tests this construction in a
generalized form. For $P=M_{11}$, the necessary compatibility checks are

$$
{\color{blue}PP^+M_{12}=M_{12}},\qquad
{\color{blue}M_{21}P^+P=M_{21}}.
$$

When these hold, replace $p$ by $r_P=\operatorname{rank}P$ and
$S$ by $M_{21}P^+M_{12}$ in the formula. Rectangular/singular corners can
therefore qualify, but a pseudoinverse alone does not establish compatibility.
The implementation rejects unresolved pivots and verifies the original trace
maps before accepting a candidate. At nonzero cutoff it may retain fewer
residual directions than the full one-cut optimum.

For other borders, Pepsy optionally refines factors $A=LR$, where $L$ has at
most $\chi$ columns. With one factor fixed, updates solve a linear
least-squares problem inside the null space of both protected maps. For
example, with $G=RQ_R$, $T=I-GG^+$ and $E=M-LR$,

$$
{\color{purple}\Delta L}
=(I-Q_LQ_L^\dagger)E(TR)^+T,
\qquad Q_L^\dagger\Delta L=0,\quad\Delta L G=0.
$$

Mirror the update for $R$, balance the factor gauge by QR, and accept only
feasible nonincreasing-error candidates, allowing floating-point roundoff.
The best paper/Schur seed remains available. This is a local alternating
solver, not a proof of the full nonconvex optimum or a global MPO fitter.
See the [implementation record](../../../../docs/development/notes/2026-11-17-dmt-refinement.md).

## 4. What the paper and density DMT actually do

[White et al., Sec. III B–C](https://arxiv.org/pdf/1707.01506#page=6)
freeze the protected rows/columns, subtract a disconnected baseline, and
SVD-truncate the remaining free connected block. Their trace-aligned
coordinates give $B_{\alpha\beta}=M_{\alpha0}M_{0\beta}/M_{00}$.
Equations (23)–(25) describe the connected-block update; Eq. (27) refactors
the resulting matrix. Appendix B supplies the conservative $r+8$ bond bound
for four protected directions per side.

In the conventions above, let $\tau_{st}=\delta_{st}$,
$t_L=\tau^TF_L$, $t_R=\tau^TF_R$, and $z=t_LMt_R^T\ne0$. Define

$$
B=\frac{(Mt_R^T)(t_LM)}{z},\qquad
C=(Q_L^\perp)^\dagger(M-B)Q_R^\perp.
$$

$B$ represents $\rho_L\otimes\rho_R/z$; rank one means **bond rank**, not
rank one of the physical density matrix. Let
$\mathcal E(Z)=Q_L^\perp Z(Q_R^\perp)^\dagger$ and $K=M-\mathcal E(C)$.
The chosen family is

$$
{\color{purple}A(Z)=K+\mathcal E(Z)},\qquad
\operatorname{rank}Z\le r,\qquad r=\chi-p-q\ge0.
$$

It preserves the L-shape. To justify the rank bound, use adapted trace
coordinates $t_LW_L=(a,0)$ and $W_R^\dagger t_R^T=(b_0,0)^T$.
Then $B_{22}=M_{21}b_0aM_{12}/z$, and with $H=b_0aM_{12}/z$,

$$
\overline K=
\begin{pmatrix}M_{11}&M_{12}\\M_{21}&M_{21}H\end{pmatrix}
=\begin{pmatrix}I_p\\0\end{pmatrix}(M_{11}\ M_{12})
+\begin{pmatrix}0\\M_{21}\end{pmatrix}(I_q\ H).
$$

Thus $\operatorname{rank}K\le p+q$ and
$\operatorname{rank}A(Z)\le p+q+r\le\chi$. An arbitrary value of block $22$
would not justify this factorization.

Orthonormal embeddings give

$$
\boxed{\min_{\operatorname{rank}Z\le r}\|A(Z)-M\|_F^2
=\min_{\operatorname{rank}Z\le r}\|Z-C\|_F^2.}
$$

Eckart–Young solves this **restricted** objective by $C_r=\operatorname{SVD}_r(C)$:

$$
A_{\rm DMT}=K+\mathcal E(C_r),\qquad
\|\rho_c-\rho\|_{\rm HS}^2=\sum_{i>r}s_i(C)^2.
$$

The family is a subset of all feasible total-rank matrices. This is why its
optimum does not establish the full boxed optimum in section 1.
For `variant="operator"`, use $B=0$ and the raw free block instead.
That avoids the nonzero-trace denominator but changes the chosen residual.

### Counterexample to an unrestricted optimum claim

Keep the first row/column of $M=\operatorname{diag}(1,0.3,0.1)$ fixed,
so $p=q=1$, and set $\chi=2$. Conservative DMT uses $r=0$:

| Feasible embedded result | Total rank | Squared error |
| --- | --- | --- |
| $A_{\rm DMT}=\operatorname{diag}(1,0,0)$ | $1$ | $0.10$ |
| $A_*=\operatorname{diag}(1,0.3,0)$ | $2$ | $0.01$ |

Both obey the constraints and rank cap. The latter follows the solvable
corner case with $p=1$ and one available Schur-residual direction. This is
an optimality-scope example, not a failure of the preservation guarantee.

## 5. Basis, sweeps, and implementation interpretation

With normalized Pauli operators $\widehat P_\mu=P_\mu/\sqrt2$, set
$T_{\mu,(s,t)}=\operatorname{Tr}(\widehat P_\mu^\dagger E_{st})$.
Then $T$ is unitary, $\tau_P=\sqrt2e_I$, and $\tau_P^TT=\tau_E^T$.
The maps transform as $F_{L,P}=TF_{L,E}$ and $F_{R,P}=TF_{R,E}$.
Their protected virtual ranges, trace rows, and constraint sets agree.
The same virtual gauge and protected bases can therefore use the same
$M,B,C$ and chosen $C_r$. Degenerate singular cutoffs can admit multiple
equally optimal residual approximations; independent sweeps need not pick
identical tensors. Raw Pauli coordinates use $2e_I$ and a $2^N$ factor for
squared coefficient norms versus physical Hilbert–Schmidt norms.

For a sweep $\omega^{(0)}=\rho,\ldots,\omega^{(J)}=\rho_c$, local preservation
holds by induction. Each tail norm refers to its own input, and the safe
final bound is

$$
\|\rho_c-\rho\|_{\rm HS}
\le\sum_{j=0}^{J-1}\|\omega^{(j+1)}-\omega^{(j)}\|_{\rm HS}.
$$

The default [shared kernel](../../../../src/pepsy/operators/_pauli_dmt.py)
uses an exact support SVD, identifies protected ranges, computes the density
baseline when requested, projects the residual, truncates with the conservative
budget and cutoff policy, and refactors thin factors. These are different
roles for SVD/QR. The optional optimized route adds the guarded completion
and local alternating solver above. The
[ordinary MPO contract](../../../../docs/api/operators/mpo_dmt.md)
states both methods' optimality and finite-precision limits.
Read current callers before mapping these formulas to code variables.

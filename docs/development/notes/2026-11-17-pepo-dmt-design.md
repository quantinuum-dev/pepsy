# Proposed protected truncation for native PEPOs

Status: **experimental prototype** for NumPy PEPOs, with exact and direct or
variational boundary-MPS maps. Matrix-free scalable environments and broader
backend support remain design work. The new
[`PepoOptimizer`](../../api/optimizers/pepo.md) supplies gate replay with
local SVD, simple update, boundary-MPS full update, and a guarded regional
feasibility/refinement solver using single-layer maps. The existing
[`PepsOptimizer`](../../api/optimizers/peps.md#native-pepo-full-update) accepts
native PEPOs and has reduced two-site full updates, ALS/L-BFGS, and a
simple-update warm start. [`gate_loop_cluster`](../../api/operators/gates.md#local-six-site-cluster-updates)
also supplies finite-cluster SU/BP-closed PEPO updates. Neither promises DMT
local-density preservation today.

## Candidate first implementation

1. Start with a dense open PEPO, one adjacent physical pair, and an explicit
   inventory of protected physical regions. Form the exact gated pair before
   output truncation, keeping both operator legs for $U\rho U^\dagger$.
2. Build a cheap SU or finite-cluster environment. Its norm metric weights
   truncation error, while separate **single-layer trace contractions** map
   the gated pair to the requested reduced operators. The norm environment
   alone is not a partial-trace environment.
3. Fit a rank-$\chi$ pair to the untruncated target. For fixed exterior
   tensors, each selected reduced operator is a linear function
   $\mathcal T_W$ of the active pair matrix $M$. For a positive-semidefinite
   approximate norm metric $N$, a candidate objective is

   $$
   \min_{\operatorname{rank}A\le\chi}
   {\color{purple}\|A-M\|_N^2}
   \quad\text{s.t.}\quad
   {\color{blue}\mathcal T_W(A-M)=0}\quad\forall W.
   $$

   On a PEPO with loops, cutting the active bond does not generally separate
   the network. The maps $\mathcal T_W$ therefore need not factor into the
   left/right trace maps used by MPO DMT; an SU/BP norm environment alone
   does not make the MPO proof exact.
4. If the metric factors as left/right positive matrices on their supported
   subspaces, whiten those subspaces and use a protected SVD **only when**
   the required trace maps also factor consistently. A zero eigenvalue needs
   explicit support handling; blindly inverting SU weights is unstable.
   Otherwise use a constrained local solve or report that the protected SVD
   construction is inapplicable. Finite penalties give approximate rather
   than exact constraints.
5. Report the requested cap, protected numerical ranks, solver and environment,
   trace defect, local-density defect measured with the chosen environment,
   and optional independent checks. A separate explicit relaxed mode can
   trade local errors for feasibility at small bond dimension.

## Exactness and performance

An SU/BP-closed result can preserve constraints *in that approximate
environment*. This does not establish exact global PEPO reduced densities.
Validate against exact contractions on small PEPOs and compare larger cases
with converged boundary contractions. BP fixed-point convergence alone is not
a global accuracy certificate; see the [BP/SU relationship](https://arxiv.org/abs/2008.04433)
and [neighborhood updates](https://arxiv.org/abs/2107.06635).

The first prototype should cache contraction **plans**, refresh numerical
environments after the state changes, and avoid global-density checks on
every gate. Report optional periodic independent checks and separately time
target formation, environment construction, protected split, and checks.
No speedup, positivity, long-time accuracy, automatic JIT, or GPU support is
implied by the design.

Suggested validation: dense 2×2 and 2×3 references for trace and all selected
one-/two-/three-site densities; complex transpose tests; actual lossy gates;
bond cap and failure transactionality; exact-tree/separable-environment
cases; SU/BP versus independently converged boundary checks; backend/dtype
and gradient checks. Compare full accuracy trajectories across $\chi$ and
environment size, not only a local split against its immediate target.

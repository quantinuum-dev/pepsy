# PEPS tVMC implementation and stability review — 2026-10-13

**Implemented:** Pepsy has PEPO startup and fixed-bond Torch/NetKet TDVP.
It has not reproduced the paper's scalable algorithm or long-time accuracy.
This review checks current source against
[Wu and Nys, PRX Quantum 7, 033035 (2026)](https://journals.aps.org/prxquantum/pdf/10.1103/tggc-8fjx).

## Implementation comparison

The paper removes analytic gauge directions with QR in direct TDVP, or uses
uncentered minSR, and exploits physical-slice derivative sparsity for storage.
Its main target is low-energy local quenches; the Ising global quench uses
full-update startup before fixed-bond tVMC. See Secs. III and IV F.

| Mechanism | Pepsy implementation | Remaining gap |
| --- | --- | --- |
| Single-layer sampled amplitudes | Both backends support boundary contraction and exact small-system references | Trajectory accuracy needs independent contraction checks |
| Direct gauge removal | Torch constructs dense bond/scaling generators and uses rank-revealing SVD | No paper-style constraint removal/QR; no native Symmray gauge helper; no explicit fixed-particle-number rescaling generator |
| Uncentered minSR, Eq. (17) | Torch holomorphic solver reuses uncentered derivatives and energies | NetKet TDVP adapter uses upstream QGT solves; VMC_SR sample-space optimization is a separate driver |
| Compressed complex coordinates | Both backends expose real/imaginary coordinates | Holomorphic minSR cannot be substituted for this real-coordinate equation |
| Physical-slice derivative storage | Full Jacobians are currently stored | No small-o implementation; test derivative sparsity under the chosen compression before adopting it |
| Sampling | NetKet Metropolis and native Torch local/exchange/BP proposal routes | No paper-scale sampling/mixing validation |
| Product-state startup | TFIM examples use fitted PEPO evolution before TDVP | Startup time and retained ranks require convergence checks |

Owning source: [NetKet bridge](../../../src/pepsy/vmc/netket.py),
[Torch TDVP](../../../src/pepsy/vmc/torch/tdvp.py),
[Torch SR](../../../src/pepsy/vmc/torch/sr.py), and
[public API](../../api/vmc.md#fixed-bond-real-time-peps-evolution).

## Stability and code changes

The paper's Eq. (26) uses a **squared**, unregularized normal-equation residual.
It does not measure all errors in the physical trajectory.
For centered Jacobian $C$, centered local energies $e$, and normalized weights $W$:

$$
{\color{blue}\epsilon_{\mathrm{solve}}}
=\frac{\|S\dot\theta+iF\|^2}{\|F\|^2},
\qquad
{\color{purple}r_{\mathrm{tangent}}}
=\frac{\|W^{1/2}(C\dot\theta+ie)\|}{\|W^{1/2}e\|}.
$$

For real coordinates, replace the first equation's metric and right-hand side
by $\operatorname{Re}S$ and $\operatorname{Im}F$; retain the complex tangent
mismatch in the second. A velocity can solve the normal equation exactly
while missing a component orthogonal to the sampled tangent span.

This session adds the squared solve residual, complex tangent residual and
its absolute numerator/denominator to Torch diagnostics without extra
contractions. Both drivers reject holomorphic compressed PEPS. NetKet rejects
invalid static driver controls before importing optional dependencies. Known forward-mode
QGTs remain rejected when wrapped in `functools.partial` too.

The [short-run evidence](2026-10-13-netket-peps-convergence.md) already
separates algebraic and tangent residuals. The
[stopped run](2026-10-13-netket-peps-stopped-run.md) retained infidelity 0.18436
at t=0.99 for D=3. Frozen contractions/Jacobians agreed with exact references
at one snapshot; this does not isolate the trajectory's failure mechanism.
No new physical trajectory was run for this review.

## Coding priorities for stability

**Proposed, not implemented by this review:**

1. Expose equivalent accepted-state NetKet solve and tangent diagnostics.
   Native driver internals hold RK-stage quantities, so copying the most
   recent force/QGT can incorrectly label a stage diagnostic as an endpoint.
2. Prototype sample-space dynamics for explicit real coordinates. Validate
   against small full-sum TDVP, including genuinely truncated amplitudes;
   the holomorphic uncentered formula cannot be copied unchanged.
3. Remove known null modes before Cholesky. The sample-count versus parameter-
   count dispatch is a cost heuristic, not a rank certificate. Duplicate
   configurations, restricted support and symmetry add dependencies.
   A successful shifted Cholesky solve alone does not establish conditioning
   of the unshifted equation.
4. Implement physical-slice derivative storage after testing its sparsity
   under the actual compression. PEPS-local derivatives offer a memory path
   that generic stored NetKet Jacobians do not exploit.

Check samples/seeds, regularization, time step, boundary cap and D separately.
In-sample minSR residuals can be tiny through interpolation, so independent
configurations remain necessary. A sample Gram matrix is positive semidefinite
even with approximate differentiable amplitudes; that does not remove its null
space or certify accuracy of the amplitudes. Born-only sampling also misses
directions supported entirely on zero-amplitude configurations.
Gauge removal cannot repair insufficient fixed D or missing Born support;
these are separate from redundant parameter directions. The paper identifies
product-state initialization and support collapse as open issues in Sec. VI.

## Environment and validation ownership

Installed API inspection: NetKet 3.22.4, JAX 0.11.2, Flax 0.12.10;
the remaining tensor/backend versions match the
[earlier audit](2026-10-13-netket-symmray-audit.md). Inspected native TDVP,
iteration and QGT signatures. Decisions: **adopt** explicit residual
conventions and compressed-chart validation; **prototype** scalable dynamics
solvers/storage; **defer** production application claims. No dependencies
were changed. Final checks and working-tree status belong to the
[session handoff](../../../history/2026-10-13-tvmc-paper-review.md).

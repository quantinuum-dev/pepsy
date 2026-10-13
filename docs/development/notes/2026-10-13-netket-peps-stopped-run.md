# 2026-10-13 — Stopped 4×4 PEPS TDVP run and API review

**Measured:** the user requested stopping simulations and committing the code
review. The requested t=5 trajectory stopped after 194 accepted TDVP endpoints
at t=0.99. This is partial evidence, not a completed or converged t=5 result.

## Setup and retained results

Open TFIM `H=-sum ZZ-sum X`, product-plus initial state, two full bond-2 PEPO
startup steps of 0.01, fit caps `(18,72)`, then native NetKet midpoint TDVP.
Fixed D=3, single-layer boundary-MPS chi=24, cutoff zero, dt=0.005,
1024 total samples per stage, 64 chains, burn-in 16, sampler sweep size 16,
chunk size 256, native Cholesky and diagonal shift 1e-6. Exact sparse-matrix
evolution starts from the physical initial state, including startup time.
The [upstream audit](2026-10-13-netket-symmray-audit.md) records dependency
versions and the user's locally patched NetKet 3.22.4/JAX 0.11.2 stack.

| Quantity at t=0.99 | PEPS | Exact |
| --- | --- | --- |
| X magnetization per site | 0.3578158242 | 0.3653405550 |
| Average nearest-neighbor ZZ | 0.3966765780 | 0.4231062967 |
| Global X parity | 0.9593586031 | 1.0000000000 |

Final infidelity is 0.184359576; phase-aligned vector distance is 0.440163305.
Total energy is -15.2452910585, with drift +0.7547089415 from initial -16.
Raw squared norm is 1.5898028897e-10; acceptance is 0.3634948730.
The final accepted step took 3.964 s, including 3.656 s in TDVP. Its elapsed
trajectory time was 784.83 s, including initialization and diagnostics.

The user subsequently requested deletion of generated simulation outputs.
The temporary run directory, trajectories, figures and tensor/vector snapshots
were removed. Before deletion, the latest saved snapshot was step 180, t=0.92;
later accepted endpoints had scalar records only. Snapshots omitted RNG,
sampler and integrator state. The interrupt was requested by the user and was
not a numerical failure. This note retains the measured summary only.

## Frozen contraction and compilation evidence

At saved t=0.72, boundary chi=24 and chi=48 amplitudes agreed with exact
contraction to relative 1.84e-15; HOTRG/CTMRG chi=24 agreed to 3.77e-15.
Boundary-versus-exact Jacobians on four Born-distributed configurations and
2304 real coordinates agreed to relative 1.40e-15. This frozen check does
not establish trajectory-wide convergence. Increasing boundary chi alone
is not supported as the remedy for the observed late-time error.

A separate earlier profile recorded compilation on the first TDVP step
and zero compile messages on steps two and three. Fixed shapes reused JIT
kernels. Raising chunk size from 32 to 256 reduced measured TDVP time from
about 13 s to about 3.6 s; these are scoped local measurements, not a portable
performance guarantee. No new simulation was started after the stop request.

## Code changes and validation

- Boundary inputs use native Quimb full-tensor scale extraction before
  selecting configurations; logarithmic amplitudes remain finite even when
  the complete raw amplitude is below floating-point range. Dense, flat-Z2
  and native fermionic paths retain phases and ordering. This does not use
  `balance_bonds_()` or densify Symmray tensors.
- Native SR detects output mode by default. Shared real/imaginary mode maps
  to the NetKet VMC_SR complex-output Jacobian. Compressed custom-VJP paths
  select reverse Jacobians and reject unsupported forward-mode requests.
- Builders honor thinning; call-time burn-in is applied and restored even
  with the released NetKet zero-default bug. TDVP exposes native dense/JAX CG
  solvers, solver kwargs and a configurable dense-metric allocation limit.
- The [TFIM example](../../../examples/tfim_netket_tdvp.py) supports absolute
  final time, separate exact initial/handoff references, state snapshots,
  observables, timing and a scale-safe exact normalized-vector oracle.

Installed-stack checks before stopping: 32 TDVP checks passed, then 16
selected eager/scaling/example checks, 114 VMC/native-symmetry/public-API
checks, two native fermionic scaling checks and 14 selected GPU
derivative/scaling checks passed. Selections overlap and are not a full-suite
claim. Final review checks are recorded in the
[wrap-up journal](../../../history/2026-10-13-netket-peps-wrap-up.md).

**Deferred:** controlled timestep, sample-count, seed, regularization and D
comparisons at later times; gauge/tangent conditioning; complete restart
checkpoints and paper-level scalable dynamics. The earlier
[short convergence probe](2026-10-13-netket-peps-convergence.md) covers a
different interval and reference origin.

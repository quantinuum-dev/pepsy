# NetKet PEPS TDVP — bounded validation, 2026-10-13

Implemented in [the NetKet bridge](../../../src/pepsy/vmc/netket.py), with a
[runnable example](../../../examples/tfim_netket_tdvp.py) and
[numerical regressions](../../../tests/test_netket_tdvp.py).

## Executed trajectories

Open TFIM `H=-sum ZZ-sum X`, initial product-plus, two full bond-2 PEPO
startup steps of `0.01`, fixed fitting caps `(18,72)`, then two midpoint
tVMC steps of `0.001`. PEPS grows to `D=3` at the second startup step.
Single-layer boundary `chi=9`, cutoff zero, 16 chains, 16 discarded samples
per chain, chunk 32, real/imaginary coordinates, native Cholesky with
diagonal shift `1e-6`. The exact oracle evolves the state **at handoff**;
it does not include startup errors.

| 4×4 sample count | Final time | Final exact infidelity | Energy drift from handoff |
| --- | --- | --- | --- |
| 128 | 0.022 | 3.1631e-9 | -5.2000e-6 |
| 256 | 0.022 | 1.7501e-11 | -4.0459e-6 |

Both completed with finite parameters and fixed `D=3`/`chi=9`.
This sample comparison is short development evidence, not statistical or
long-time convergence. Native NetKet 3.22.4, JAX/JAXlib 0.8.3, Flax 0.12.2,
Optax 0.2.6, Equinox 0.13.2, CPU. Dependencies were installed only into
isolated `/tmp` overlays using the session interpreter. Shared packages and
dependency declarations were unchanged. The installed CUDA plugin does not
match this CPU JAX probe; it was not used.

Earlier baseline: NetKet 3.21.0 with the same compatible JAX reproduced
the 128-sample result. A 2×2/D=2 startup plus three sampled midpoint steps
ended at `t=0.023`, infidelity `1.1311e-12`, drift `-6.4740e-7`.

## Corrections found by independent checks

- Native TDVP RHS calls leave the shared variational parameters at an
  intermediate Runge--Kutta stage. The adapter copies the accepted public
  `integrator.y` back before endpoint handoff and clears cached samples.
- A strict floating-point `t<t_end` loop can add a full extra step at a
  requested endpoint a few ulps away. Explicit final stops plus the
  adapter's roundoff check prevent this.
- Generic random 2×2/D=2 PEPS has insufficient tangent rank for exact
  Schrödinger evolution. Its initial projected derivative residual was
  about 0.164. Exact full-sum comparisons use D=3; sampled D=2 still checks
  bounded variational dynamics.
- Flat Z2 boundary cap one creates a zero-width sector upstream. Native
  coordinate-gradient checks use cap two, retaining both equal sectors;
  dense checks use cap one. Both checks require actual truncation and
  compare reverse derivatives to finite differences.
- NetKet 3.22.x fails to import with the shared JAX 0.11.2 annotation types;
  its existing declared JAX bound is relevant. JAX 0.10.2 with NetKet 3.21
  also fails in removed sharding operations. No installed library was patched.

## Remaining scope

No paper-scale reproduction, optimized gauge projection, small-o storage,
native Z2 time-evolution integration, dense-to-Z2 conversion or compiled
Torch compression derivatives was added. No convergence certificate for
later entanglement, longer evolution, sample mixing or bond dimension.

# PEPS/NetKet tVMC convergence probe — 2026-10-13

**Measured:** seven 4×4/D=3 trajectories completed without non-finite
parameters or endpoint errors. Each evolves for 0.2 tVMC time units after
the same PEPO startup, ending at t=0.22. This extends the previous 0.002
tVMC interval by a factor of 100. All 240 accepted endpoints were compared
with exact sparse-matrix evolution from the fitted handoff state.

The user subsequently requested deletion of generated simulation outputs.
The full scalar JSON, parameter/reference snapshots, logs and exported figures
were removed from the checkout and temporary run directory. This note retains
the measured summary only.

## Fixed setup and comparisons

Open TFIM `H=-sum ZZ-sum X`, initial product-plus, two full bond-2 PEPO
steps of 0.01. Fit caps `(18,72)`, fixed D=3, single-layer cutoff zero,
16 chains, burn-in 16, chunk 32, midpoint integration, native Cholesky
and diagonal shift 1e-6. Each trajectory keeps its own boundary cap fixed.
Seeds are 44 unless explicitly listed. The reference starts at handoff;
startup/Trotter/compression errors are not included in the tVMC infidelity.

| dt | Samples | chi | Seed | Final infidelity | Maximum absolute energy drift |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 256 | 9 | 44 | 5.2464e-6 | 7.0987e-3 |
| 0.005 | 256 | 9 | 44 | 1.0586e-6 | 1.5388e-3 |
| 0.0025 | 256 | 9 | 44 | 6.9524e-7 | 9.4703e-4 |
| 0.01 | 512 | 9 | 44 | 4.5296e-6 | 7.2922e-3 |
| 0.005 | 512 | 9 | 44 | 3.7609e-7 | 1.5504e-3 |
| 0.01 | 256 | 6 | 44 | 5.2464e-6 | 7.0987e-3 |
| 0.01 | 256 | 9 | 45 | 5.6360e-6 | 6.9663e-3 |

Energy drift is the total energy difference from handoff; divide by 16 for
the per-site value. Maximum drift is used because final drift can be small
through cancellation. Run times in the JSON include overlapping work and
are not controlled throughput comparisons.

The exact endpoint observables are `Mx=0.7576085073`, average nearest-neighbor
`Czz=0.1615934666`, and global X parity one. With dt=0.0025/samples=256,
the final magnetization/correlation errors are `2.4708e-5` and `-5.5932e-5`.
With dt=0.005/samples=512, they are `1.9620e-4` and `-8.4699e-5`.
Pauli reconstruction `E=-16 Mx-24 Czz` agrees with the sparse-operator
energy to better than 3e-14 at every endpoint.

## What limits this benchmark

- **Timestep:** halving dt at 256 samples reduces maximum energy drift
  about 4.6-fold and final infidelity about fivefold. The finer step further
  improves magnetization and correlation. Late infidelity does not follow
  a clean second-order convergence law, so this is not a timestep certificate.
- **Sampling:** doubling samples at dt=0.01 gives a modest fidelity change;
  at dt=0.005 it reduces final infidelity about 2.8-fold. A second seed at
  dt=0.01 gives comparable errors. This is limited seed evidence, not a
  statistically converged uncertainty estimate for the trajectory.
- **Boundary cap:** chi 6/9 trajectories agree to numerical precision in
  this test. At saved t=0.02/0.12/0.22 snapshots, 128 Born-distributed
  reference configurations per snapshot give chi 6-versus-18 amplitude
  relative errors below 2e-14; chi 9/18 outputs agree exactly. Boundary
  truncation is not the observed bottleneck in this geometry/interval.
- **Linear solve:** a separate endpoint check with 2048 samples has
  regularized relative residual `2.4539e-15`, and unregularized relative
  residual `1.6926e-6` (squared `2.8650e-12`). These use the real-coordinate
  equation. Small algebraic residual does not certify physical TDVP accuracy.
- **Finite-D projection:** the same sampled endpoint tangent equation has
  relative wavefunction residual `1.2867e-3`. This suggests a projection
  contribution may remain after timestep refinement; a higher-D and
  better-sampled comparison is needed to establish its size.

The tangent diagnostic is the sampled norm of `O_centered v+i E_centered`
relative to `E_centered`, computed from the same samples, forces and QGT.
It differs from the algebraic residual of the QGT linear system.

## Sampling, normalization and snapshots

A fresh frozen-endpoint measurement used 2048 samples, 16 chains, burn-in
32 and sweep size 16, with NetKet FFT autocorrelation enabled. Acceptance
is 0.7653, local-energy Rhat 1.0027, tau 0.9140, maximum chain tau 1.5756.
The energy estimate is `-16.1158+0.0938i`, native reported standard error
0.1194; exact energy of the frozen PEPS is -16.0063. This is one endpoint
mixing check, not a guarantee for every stage or observable.

Raw state norm changes as a projective gauge scale; final squared norms
are approximately 5.7e-4. Fidelity and observables use normalized vectors.
Normalize the reconstructed PEPS before treating its norm as explicitly one
in another workflow. Snapshots retain the raw complex tensor parameters;
they do not contain a complete sampler/RNG/integrator restart state.
Checkpoint reconstruction was verified against saved normalized dense states.

## Reproduce and scope

Use [the example](../../../examples/tfim_netket_tdvp.py) with the optional
NetKet dependencies and the recorded compatible environment:

```bash
python examples/tfim_netket_tdvp.py --lx 4 --ly 4 --d 3 \
  --samples 512 --chi 9 --dt .005 --steps 40 --solver cholesky \
  --reference --checkpoint-dir /tmp/peps-tvmc-validation
```

CPU, native NetKet 3.22.4, JAX/JAXlib 0.8.3, Flax 0.12.2, Optax 0.2.6,
Equinox 0.13.2, and the previously audited Quimb/Symmray development versions.
Used the existing session interpreter with isolated `/tmp` dependency overlays;
shared packages were unchanged. The mismatched installed CUDA plugin was
not used. No paper-scale, GPU trajectory, native-Z2 trajectory, D convergence,
regularization convergence or long-time mixing claim follows from this probe.

New observable/snapshot regressions: **2 passed**, 11.94 s. Ruff, local links
and diff checks passed. Existing dynamics/solver algorithms were unchanged.

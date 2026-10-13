# 2026-10-13 — Longer PEPS/NetKet tVMC validation

- Scope: user authorized the proposed longer 4×4/D=3 exact-reference
  benchmark and controlled fixed-cap, timestep and sampling comparisons.
- Branch / baseline: `develop` / `82ee387`; the PEPS/tVMC integration is
  published in ancestor `8032d49`.
- Status: new example/diagnostic changes are uncommitted; no publication.

## Implemented

- Extended the [TFIM example](../examples/tfim_netket_tdvp.py) with normalized
  X magnetization, nearest-neighbor ZZ correlation, global X parity and
  raw norm reporting, plus tensor/reference snapshots.
- Snapshots contain real/imaginary parameters and metadata, not full sampler,
  RNG or integrator state. They are for frozen-state checks, not exact restarts.
- Added independent Pauli-matrix and snapshot parameter checks in
  [the numerical test file](../tests/test_netket_tdvp.py); updated the API guide.
- Package dynamics and solver algorithms were unchanged.

## Measurements

Same H=-sum ZZ-sum X and two PEPO startup steps of 0.01 as the earlier
handoff. Extended tVMC duration from 0.002 to 0.2, ending at t=0.22.
Fixed D=3, cutoff zero, 16 chains, burn-in 16, chunks 32, midpoint,
native Cholesky with diagonal shift 1e-6. Individual trajectories retain
their chosen chi. Compared chi 6/9, samples 256/512, dt 0.01/0.005/0.0025
and seeds 44/45. All seven trajectories completed, with 240 accepted endpoints.
[Retained note and full scalar records](../docs/development/notes/2026-10-13-netket-peps-convergence.md)
record the comparisons. Baseline final infidelity 5.2464e-6; dt=0.005/512
samples gives 3.7609e-7; dt=0.0025/256 samples gives 6.9524e-7. Timestep
reduction improves the observables/peak energy drift; late fidelity also
depends on sampling and possibly finite-D projection. No error floor is
certified. Exported PNG/PDF figures were visually checked. All probes finished.

Frozen baseline snapshots at t=0.02/0.12/0.22: 128 Born-distributed reference
configurations per snapshot. Chi 6 amplitudes agree with chi 18 to relative
2e-14; chi 9/18 outputs agree exactly in these samples. Parameter snapshots
reconstruct the saved normalized dense states.

A separate frozen endpoint check used 2048 samples, 16 chains, burn-in 32,
sweep size 16 and NetKet FFT autocorrelation. Acceptance 0.7653, local-energy
Rhat 1.0027, tau 0.9140. These are endpoint diagnostics, not a trajectory-wide
mixing certificate. Regularized QGT residual 2.4539e-15; unregularized
relative residual 1.6926e-6, squared 2.8650e-12. The sampled tangent residual
is 1.2867e-3 relative, suggesting finite-D projection deserves follow-up;
it does not isolate a globally converged D=3 error floor.

## Validation and environment

- New observable/snapshot checks: **2 passed**, 11.94 s.
- Pauli energy reconstruction E=-16 Mx-24 Czz matches the sparse-operator
  energy to roundoff at the observed endpoints.
- Ruff and diff checks passed before final handoff.
- Existing compatible `/tmp` dependency overlays, CPU, native NetKet 3.22.4,
  JAX 0.8.3, Flax 0.12.2, Optax 0.2.6, Equinox 0.13.2. No shared package changes.
- Runtime logs and parameter snapshots under `/tmp/pepsy-netket-long-20261013`.
- No new full-suite claim or paper-scale reproduction.

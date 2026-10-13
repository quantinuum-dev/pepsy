# 2026-10-13 — Reference caps versus practical evolution and tVMC readiness

- Scope: explain the earlier `(81,324)` budgets, check smaller caps, and
  assess optimization/time evolution against Wu and Nys (2026).
- Branch / baseline: `develop` / `43dfc4a`.
- Commit status: uncommitted working-tree changes; no staging or publication.

## Findings

- Reference caps came from a four-site fused-boundary middle-cut rank bound:
  norm open dimension `3*3=9`, overlap open dimension `3*2*3=18`; two sites
  per side give `81` and `324`. They remove truncation for the tested
  representation. They were not measured minimum practical caps or a general
  law for larger lattices. The
  [validation note](../docs/development/notes/2026-10-13-pepo-4x4-d3.md)
  now states that distinction and links the new evidence.
- Fresh 4×4 product-plus startup, `D` cap 3, `dt=0.01`, `J=h=1`, three
  steps to `t=0.03`: fixed `(18,72)` and `(36,144)` gave identical final dense
  vectors (relative difference zero in a direct rerun). Output grew to `D=3`
  at step two; final circuit infidelity `4.6163e-11`, exact squared norm
  `0.9999999999999981`. Reported step-three loss agrees within `1e-14`.
  The sweep did not improve the already accurate warm start at steps two/three.
- On four configurations of the previous random 4×4 `D=3` state, single-layer
  amplitude caps `3,6,9,12` gave relative errors `0.330`, `0.0757`, `1.9e-15`,
  `1.9e-15`. A four-site single-layer chain has the analogous rank bound nine.
  This cap belongs to tVMC amplitudes, separately from multi-layer PEPO metrics.
- [Paper Sec. IV F](https://journals.aps.org/prxquantum/pdf/10.1103/tggc-8fjx)
  uses `D=5`, FU boundary MPO 125, single-layer tVMC boundary MPS 20, an 11×11
  critical-field quench, and FU startup to `t=0.4`. The short `h=1` probe above
  does not reproduce that trajectory or establish later cap adequacy.

## Added validation and runnable workflow

- [Flat-Z2 tests](../tests/test_vmc_tdvp_compile.py) now check that five native
  real-coordinate SR updates lower exact energy and that the parity-preserving
  2×2 midpoint tVMC step agrees with dense evolution (fidelity above `1-1e-10`).
- [Example](../examples/tvmc_flat_z2.py) accepts `--optimize-steps`,
  `--learning-rate`, `--sr-diag-shift`, then the existing real-time step count.
  Every optimization step draws fresh samples; emitted optimization energies
  precede their parameter update. Three sampled optimization updates followed
  by two midpoint steps completed, retaining native flat-Z2 `D=4`.
- Updated TDVP/compile selection: **25 passed**, 35.50 s. Focused new energy/
  dense-dynamics checks: **2 passed**, 7 deselected, 4.16 s. These are scoped
  results; the earlier 211 related checks/94 smoke results remain separate.
- Ruff, local links and diff checks passed. All bounded probes completed.

## Readiness assessment

- Implemented and validated for small systems: native PEPS VMC/SR ground-state
  updates, Torch Euler/midpoint TDVP, dense PEPO startup, and an independent
  native flat-Z2 workflow. JAX GPU amplitudes/gradients are validated; a full
  JAX TDVP integration has not been validated in this environment.
- Paper-level maturity is **not established**: sustained energy/observable
  trajectories, sampling/boundary/time-step/bond convergence, the small-o
  memory path and the paper's optimized gauge treatment remain missing from
  the validation/implementation. Current direct gauge projection uses dense
  rank-revealing SVD; native symmetry analytic gauge projection is unsupported.
- Compressed Torch export remains inference-only, with stabilized derivatives
  on the original contraction path. No automated dense-to-native-Z2 PEPO
  handoff, production trajectory, shared dependency change or full-suite run.

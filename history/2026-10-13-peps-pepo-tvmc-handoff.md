# 2026-10-13 — PEPO startup and fixed-bond PEPS tVMC

- Scope: user requested a short PEPO/PEPS evolution followed by fixed-bond
  PEPS tVMC, referencing Wu and Nys, PRX Quantum 7, 033035 (2026).
- Branch / baseline: `develop` / `43dfc4a`.
- Commit status: working-tree edits; nothing staged, committed or published.

## Implementation

- [Native TDVP](../src/pepsy/vmc/torch/tdvp.py) composes the existing amplitude,
  Metropolis, connection, local-estimator, Jacobian, and SR solver kernels.
  Supports Euler/midpoint, complex holomorphic parameters, stage resampling or
  refreshed importance probabilities, rollback after stage failure, and
  unchanged parameter shapes. Repeated configurations share one Jacobian row
  with their weights aggregated, preserving sampled covariances.
- Direct TDVP projects out analytic bond-gauge and projective scaling vectors.
  Rank-revealing SVD handles dependent generators; it is a small-system route,
  rather than the paper's optimized constraint-removal/QR implementation.
- minSR reuses uncentered `O` and local energies (paper Eq. 17). This avoids
  constructing parameter-space gauge vectors. Cholesky/pseudoinverse dispatch
  is reused; diagnostics report the unregularized centered TDVP residual.
- Binary lazy single-site Metropolis moves permit TFIM sampling across
  magnetization sectors; existing exchange rules are preserved.
- [Example](../examples/tfim_pepo.py) adds optional tVMC after PEPO startup,
  including a separate single-layer boundary cap and correct native
  spin-1/2-to-Pauli Hamiltonian conversion.

## Evidence and limitations

- [Paper, Sec. IV F](https://journals.aps.org/prxquantum/pdf/10.1103/tggc-8fjx)
  itself starts the Ising dynamics with full update to `t=0.4`, then switches
  to fixed-`D=5` tVMC. This supports the requested strategy; its numerical
  production benchmarks have not been reproduced here.
- `tests/test_vmc_tdvp.py`: **15 passed**. Covers real-time sign,
  direct/minSR dense handoff (fidelity above `1-1e-9`), second-order midpoint
  convergence, analytic null directions, duplicate-row weight equivalence,
  unchanged bonds/tags, lazy spin flips, input rejection and rollback.
- Related VMC API/local-energy/importance, PEPO and public API/layout selection:
  **139 passed, 4 deselected**. The initial unfiltered selection had the same
  139 passes and four NetKet failures: `pytest.importorskip('netket')` accepts
  a namespace directory left in the shared dependency search path, but it has
  no `hilbert` API or installed distribution. Those existing NetKet-dependent
  checks were excluded explicitly for the native rerun; NetKet is unvalidated.
- Tiny `2x2` boundary-amplitude CLI: two PEPO startup steps and two sampled
  midpoint tVMC steps completed to `t=0.22`, retaining `D=2`, using uncentered
  minSR. This is a functional smoke run, not a sampling-accuracy certificate.
- Ruff and `git diff --check` passed; the full repository suite was not run.
- No NetKet installation or shared-environment changes. No restarted long
  benchmarks. No small-o sparse-Jacobian implementation, distributed TDVP,
  time-dependent Hamiltonian API, real-coordinate driver, or native Symmray
  analytic gauge projection. MinSR can still require a pseudoinverse for
  dependent sample rows. Long-time/high-bond accuracy remains unverified.
- Starting bonds and elapsed startup time are chosen explicitly. Short
  startup does not certify adequate tensor ranks or Monte Carlo support;
  inspect residual, sampling, boundary-cap and time-step convergence.

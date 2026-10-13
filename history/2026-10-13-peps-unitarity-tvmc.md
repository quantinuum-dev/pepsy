# 2026-10-13 — Unitary norm reuse and PEPS tVMC assessment

- Scope: avoid redundant unitary PEPO target norms; explain sweep reuse and
  assess a NetKet-style tVMC route through Pepsy's existing VMC interfaces.
- Branch / baseline: `develop` / `43dfc4a`; working-tree edits only.
- Nothing staged, committed, pushed, installed, or launched as a persistent job.

`run_pepo(..., unitary=True)` now trusts the caller's unitarity assertion and
measures the input PEPS norm once per attempt, reusing it as the target norm.
This replaces a four-layer target self-overlap with a two-layer input norm.
An explicit `infidelity_kwargs={"norm_target": 1.0}` skips that measurement
for an input known to have exact unit norm. Finite-chi normalization alone
does not establish this assumption. Arbitrary PEPOs retain the default full
target-norm measurement. The TFIM example now uses the unitary hint.

Tests compare measured/supplied input norms, including a norm-two input,
against dense fidelity and assert that no full target norm is requested.
The existing three-step ALS/column-L-BFGS/global trajectory tests now exercise
the unitary hint. Final selection: **24 passed**; log:
`/tmp/pepsy-pepo-unitary-final.log`. Ruff and `git diff --check` passed.

## tVMC findings

- Local code has PEPS amplitudes, sampling, local energies, log derivatives,
  holomorphic and real/imaginary coordinates, and SR/minSR solvers. The Torch
  driver's current update is energy minimization; a real-time TDVP driver
  and ODE loop are not implemented.
- Pepsy's `build_ising_vmc` returns a packed PEPS NetKet `MCState`, providing
  the integration point for NetKet's existing experimental TDVP driver.
  NetKet is absent from the isolated validation environment, so no execution
  or integration-success claim was made.
- For holomorphic parameters, the real-time equation is `S theta_dot=-i F`,
  with centered log-derivative covariance S and energy force F. Real/imaginary
  coordinates need their corresponding projected force before solving;
  multiplying an already-real ground-state SR update by i is insufficient.
- Standard Born-sampled log-derivative tVMC is problematic at the exact
  zero-state start because the wavefunction has extensive zero support.
  The projected PEPO-fitting route remains the existing validated startup
  path; any tVMC integration needs a small-system reference/support check.
- Preserve Hamiltonian conventions: our circuit uses `-J ZZ-h X`, while
  NetKet's Ising builder uses `+J ZZ-h X`; matching the J=1 circuit needs
  NetKet J=-1. TDVP should be checked against continuous-time evolution,
  distinct from the earlier exact Trotter-circuit reference.

Sources: [NetKet TDVP](https://netket.readthedocs.io/en/stable/api/_generated/experimental/dynamics/netket.experimental.TDVP.html),
[NetKet Ising convention](https://netket.readthedocs.io/en/stable/api/operator.html),
and [Sinibaldi et al., tVMC bias at wavefunction zeros](https://arxiv.org/abs/2305.14294).

Moving an unchanged exact boundary representation preserves contractions.
Tensor updates invalidate environments containing those tensors; only
unchanged exterior cuts can be reused. Finite-chi compression can change
norm/fidelity estimates even while the physical state is unchanged.

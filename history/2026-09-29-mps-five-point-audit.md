# 2026-09-29 — MPS norm, fidelity, overhead, and trajectory audit

- Scope: check the user's five claims about canonical norms, local fidelity,
  log accumulation, direct/DMRG replay, diagnostic overhead, and trajectories.
- Branch / baseline: `develop`, `b343ced`.
- Commit status: no implementation edits, staging, or commits in this audit.
  This handoff is new; prior tree repairs and concurrent cluster edits remain.

## Findings

1. Open canonical MPS norms use the tracked center tensor; stored exponent is
   separate. Direct orthogonal truncation and terminal DMRG projection obey
   `<approx|target> = ||approx||^2` before optional stabilization. Thus the
   squared retained/target norm ratio is the actual normalized fidelity for
   that projection against the target generated from the current input state.
   Canonicalization alone would not prove this for arbitrary compressors.
   Some public/general ledger descriptions call every local value a proxy;
   that language is too broad for these projection paths. FIT kernel comments
   explicitly state the projection identity.
2. `_norm._accumulate_norm_survival` sums logarithms and reports fidelity with
   `exp` and infidelity with `-expm1`. The accumulated product is not generally
   the final overlap with an untruncated circuit. With chi=1, a rotation of
   angle 0.4 in the |00>,|11> subspace followed by its inverse yields local
   fidelities about 0.848353 each and product about 0.719703, while the final
   normalized fidelity is 1 in direct, generic DMRG, DMRG2, and DMRG3.
3. No direct/DMRG numerical defect was found in this selection. This is scoped
   evidence, not exhaustive correctness or a guarantee of variational optimality.
4. Default replay skips optional timing clocks/device barriers, full finite
   scans, quality checks, and explicit FIT-target overlap contractions.
   Required norm/event bookkeeping and convergence checks still cost work.
   Autoray retains ordinary accelerator unitary norm arithmetic on device;
   readout/progress, adaptive decisions, and branch sampling can synchronize.
5. Ordinary trajectory integration batches gates, forwards solver options,
   computes Kraus probabilities before collapse, and normalizes selected
   branches with the represented exponent cleared. A confirmed option-routing
   bug remains: `noise._run_trajectory_entries` sets `non_unitary=True` for
   Kraus branches but leaves `stabilize_unitary=True` in forwarded options.
   The MPS run-option resolver correctly rejects that combination.
   Reproduced with amplitude damping 0.3 on |1> for direct and DMRG2, both
   independent and coalesced two-shot replay. Default stabilization-off replay
   passes. A future fix should disable unitary restoration only for the Kraus
   branch and preserve it for ordinary unitary segments; no fix made here.

## Validation

- Fresh CPU selection: normalization, FIT performance, adjacent-window budgets,
  GPU-backend module with CUDA hidden, dynamic controls, optimizer, and trajectory
  tests: **417 passed, 17 skipped**, 68.20 s.
- Additional existing audit/FIT kernel selection for shot-option forwarding,
  fidelity/norm, timing, and sweeps: **31 passed, 108 deselected**, 4.37 s.
- Temporary numerical probe: **120** complex128 comparisons across direct,
  DMRG, DMRG1/2/3; four seeds, chi=1/2/3, adjacent and non-adjacent gates on six
  sites. Maximum local fidelity versus explicit dense overlap discrepancy:
  **2.998e-15**. Maximum center versus dense norm discrepancy: **9.992e-16**.
- The four stabilization-plus-Kraus probes all raised the same expected
  validation error, confirming the routing bug outside existing test coverage.
- No fresh full-package or GPU execution in this audit. These checks do not
  validate unrelated concurrent cluster changes or all native symmetry paths.
- Temporary evidence: `/tmp/pepsy_mps_fidelity_audit.py`,
  `/tmp/pepsy-mps-fidelity-audit.log`,
  `/tmp/pepsy-mps-five-point-audit-tests.log`,
  `/tmp/pepsy-mps-five-point-extra.log`.

## Source map

- [Norm calculations and ledger](../src/pepsy/optimizers/mps/_norm.py)
- [FIT projection kernels](../src/pepsy/fitting/local.py)
- [MPS replay and option validation](../src/pepsy/optimizers/mps/optimizer.py)
- [Trajectory replay and branch normalization](../src/pepsy/optimizers/noise.py)

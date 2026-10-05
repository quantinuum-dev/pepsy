# 2026-10-05 — Pauli eigenstate certificates and noisy reference checks

- Scope: expand exact native measurement eligibility, test noisy mixed circuits,
  and report collapse routing, as authorized after the initial native path.
- Branch / baseline: `develop`, `7a01b05`.
- Commit status: uncommitted; preserves the preceding
  [native measurement](2026-10-05-stabilizer-tableau-measurement.md) and
  [trajectory parity](2026-10-05-stabilizer-trajectory-parity.md) changes.

## Implementation

- Exactly separated coefficient sites can now be any of the six X/Y/Z
  eigenstates, with arbitrary finite nonzero scalar amplitudes. Backend
  predicates check exact zero/equality relations and transfer only Boolean
  flags. Local Clifford preparation is absorbed into the physical basis;
  coefficient reconstruction preserves scalar phase and avoids cancellation
  residues from applying numerical inverse Clifford matrices.
- Regional updates still prove identity action on every uncertified coefficient
  site. Trainable, traced, block and cyclic arrays retain conservative fallback.
  General entangled stabilizer-region detection remains deferred.
- Norm records expose fallback reasons. Simulator routing summaries derive
  from committed norm events, respecting copies and rollback. Shot-result
  summaries multiply retained leaf histories by shot multiplicity and report
  missing histories. They are logical per-shot counts, not unique executed
  prefixes or importance-weighted estimates.
- New diagnostics tests exposed coalesced replay dropping explicit `False`
  basis flags. Replay now distinguishes omitted and explicit flags, including
  measure-reset and axis aliases. Fixed-frame sampler branches also record
  their intentional MPS route.
- Kept the general sampler-localizer compression test on a genuinely magic
  coefficient input; added a separate exact Pauli-product sampling test that
  expects native collapse and no localizer compression.

## Numerical checks

- Exact eigenstate tests cover NumPy/Torch CPU complex128, X/Y/Z observables,
  nontrivial physical tableaux, permuted layouts, scalar phases/scales, and
  regional collapse beside entangled magic. Nearly matching eigenstates and
  a representable imaginary perturbation of 1e-20 are rejected.
- Twelve noisy dense-reference cases cover independent/coalesced/auto replay
  on NumPy and Torch: Clifford+T, amplitude damping, Pauli noise, measure-reset,
  feedback and X-basis reset. Every retained conditional state, recorded Born
  probability and likelihood ratio is compared to direct dense evolution.
  Uniform proposals ensure a positive damping branch at gamma=1e-30 is retained.
- STN, Stim stream, trajectory parity/noise/importance, sampling, public API
  and package suite: **762 passed, 6 skipped** before the final eight additional
  control-flag cases. Log: `/tmp/stn-pauli-domains-final.log`.
- Final expanded selection, including those control cases and available
  STN GPU/backend, MPI unit/fingerprint checks: **854 passed, 28 skipped**.
  Log: `/tmp/stn-pauli-final.log`. Skipped hardware paths are not validated;
  MPI dispatch tests here do not establish real multi-rank execution.
- Ruff and `git diff --check` passed. The full-suite attempt with Agg and
  `--maxfail=2` again stopped at **1017 passed, 12 skipped, 2 failed** in the
  same baseline-reproduced Hamiltonian tests:
  `test_ham_builder_converts_generic_mpo_to_configured_backend` and
  `test_ham_builder_automaton_preserves_shared_structure_on_backend`
  (`SVD_real requires a real Torch tensor`). Remaining tests did not run.
  Log: `/tmp/stn-pauli-full.log`.

## Compatibility audit

The preceding unchanged-environment audit is reused. Installed versions were
rechecked: Stim 1.16.0, Quimb 1.15.1.dev66+ge927f06e1, Autoray
0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f and Symmray
0.4.1.dev8+gc45f91457. Adopt installed Stim local Clifford/tableau composition
and Autoray namespace predicates; no dependency or compression-policy changes.
See the upstream sources and signatures in the
[preceding audit](../docs/development/notes/2026-10-05-stabilizer-trajectory-parity.md#upstream-audit).

No original noisy Helix builders or experimental readout were recovered by
this task; synthetic dense-reference validation is separate from that experiment.

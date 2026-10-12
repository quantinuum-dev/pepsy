# 2026-10-05 — Certified native STN measurement

- Scope: implement native Stim collapse for Clifford trajectories and safe
  measurement support regions in mixed Clifford/non-Clifford trajectories.
- Branch / baseline: `develop`, `7a01b05`.
- Commit status: working-tree edits, alongside the preceding uncommitted
  [trajectory parity changes](2026-10-05-stabilizer-trajectory-parity.md).

## Changes and decisions

- Private `_tableau_measurement` certifies live coefficient arrays using exact
  one-hot values. Global computational product states use a physical tableau.
  Regional collapse requires dimension-one boundary bonds on every mapped
  support site and verifies identity action on the entire complement before
  committing the basis update. There is no approximate magic detection.
- Omitted measurement basis flags enable automatic certified routing; explicit
  `False` retains the fixed-basis projector. MPS collapse remains the fallback.
  Shared coalesced branching and reset purity checks use the same certified
  probabilities. Compilation does not sample or mutate the quantum state.
- Pepsy owns outcome RNG, visible measurement records, feedback, normalization,
  layout, and norm diagnostics. `NormEventRecord.measurement_backend` identifies
  `stim`, `stim_region`, or `mps`. Identity measurement remains a scale-preserving
  no-op. Forced impossible outcomes are rejected before state mutation.
- Tiny nonzero amplitudes, trainable/traced arrays and block arrays are not
  certified. A numerically represented stabilizer state may remain on the MPS
  path; a basis-updating collapse can restore an exact computational certificate.
- Added independent dense/Stim references on NumPy and Torch CPU complex128,
  mixed magic support, coefficient bit rebasing, feedback/reset, finite-chi
  loss avoidance, explicit fixed-basis behavior, and atomic rejection tests.

## Validation

- STN, Stim stream, new native measurement, and trajectory parity suites:
  **475 passed, 3 skipped**, before the final four regional/re-entry cases.
- Final native measurement module: **22 passed**, including the four added
  NumPy/Torch regional and magic-removal re-entry cases.
- Shared noise, importance sampling, STN compression, sampling, GPU/backend,
  MPI unit/fingerprint, and public API/package suites: **415 passed, 26 skipped**.
- Archived Helix 52-qubit circuit: two shots on each of NumPy and Torch CPU
  complex128, 550 visible measurements per retained shot, bond dimension 1,
  all collapse records native `stim`. This is the recovered noiseless preview;
  it does not validate unavailable original noise/builders or QEC readout.
- Ruff and whitespace checks passed. Saved notebook outputs were not edited.
- Full-suite attempt with `MPLBACKEND=Agg`, `--maxfail=2`: **1017 passed,
  12 skipped, 2 failed** in the same Hamiltonian tests previously reproduced
  on the clean `7a01b05` baseline: `test_ham_builder_converts_generic_mpo_to_configured_backend`
  and `test_ham_builder_automaton_preserves_shared_structure_on_backend`
  (`SVD_real requires a real Torch tensor`). This is not a full-suite pass;
  remaining tests were not executed. Log: `/tmp/stn-native-full.log`.

## Compatibility evidence

The unchanged-environment upstream audit from the preceding parity task was
reused. Installed Stim 1.16.0's `peek_observable_expectation`, forced
`postselect_observable`, tableau composition and Pauli images were inspected
against the [official API source](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md).
Decision: adopt existing Stim APIs, with conservative exact runtime capability
checks; defer general stabilizer-region recognition and arbitrary magic cleanup.

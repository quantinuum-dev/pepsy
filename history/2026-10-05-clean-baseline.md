# 2026-10-05 — Restore test isolation and review STN changes

- Scope: user's priority 1 — resolve the two baseline failures, complete the
  full suite, and commit the reviewed stabilizer/trajectory work.
- Branch / starting baseline: `develop`, `7a01b05`.
- Commit status: included with the reviewed stabilizer/trajectory changes in
  the local priority-1 commit (see Git history). No push requested or performed.

## Root cause and correction

The two Hamiltonian failures were test-order contamination, not a missing
complex SVD implementation. `test_register_torch_linalg_leaves_quimb_drivers_untouched_without_opt_in`
installed a process-global stabilized real-only Torch SVD policy without
restoring it. Later complex Hamiltonian MPO delinearization correctly rejected
that incompatible policy. A second leak came from the real Symmray PEPS energy
validation test, whose optimizer also installs a real-only global policy.
A full-collection transition trace found a third installation in
`test_u1u1_fermionic_boundary_qr_preserves_untruncated_norm`. Local cleanup of
the first two tests was therefore insufficient. The final correction is an
autouse test fixture that snapshots and restores the process-global Torch
linear algebra policy after every test. It uses public registration/reset
APIs and does not import optional libraries in dependency-light runs. The
temporary per-test corrections were removed. Production dtype checks,
Hamiltonian construction, and SVD/QR policy behavior are unchanged.

## Evidence

- Hamiltonian module in isolation: **93 passed**.
- Minimal reproducer before the fix: the leaking registration test followed
  by the two Hamiltonian tests yielded **1 passed, 2 failed**.
- Backend and Hamiltonian modules together after cleanup: **162 passed,
  1 skipped**. Log: `/tmp/torch-policy-leak-fixed.log`.
- Policy transition tracing in a temporary pytest hook identified the second
  leak specifically after `test_peps_energy_optimizer_validation_keeps_symmray_backend`.
- Backend, energy, zero-gauge simple-update and Hamiltonian modules together
  after both corrections: **186 passed, 1 skipped**.
  Log: `/tmp/torch-policy-both-leaks-fixed.log`.
- Ruff and whitespace checks passed.
- Initial full run, started before the second correction, was interrupted
  after **1969 passed, 52 skipped, 2 Hamiltonian failures**. It is not a full
  result. Log: `/tmp/pepsy-full-clean-baseline.log`.
- The next full run with the two local corrections was also interrupted:
  **1097 passed, 12 skipped, 2 Hamiltonian failures**.
  Log: `/tmp/pepsy-full-baseline-final.log`.
- Full collection with execution restricted to the prefix before Hamiltonian
  tests reproduced **987 passed, 12 skipped, 2 failed**, identifying the third
  real-only policy installation. Log: `/tmp/torch-policy-full-prefix-audit.log`.
- With the shared isolation fixture, backend, energy, fermionic-boundary and
  Hamiltonian modules passed together: **172 passed, 1 skipped**.
  Log: `/tmp/torch-policy-isolation.log`.
- Complete run with the isolation fixture: **7165 passed, 525 skipped,
  5 failed**, 719.34 seconds. Neither Hamiltonian case failed.
  Log: `/tmp/pepsy-full-isolated-final.log`.

## Additional failures found by the complete run

- Four 4x4 sampler tests requested truncated `amplitude_mode="boundary"`
  while comparing amplitudes and importance weights with the exact dense
  oracle. This is incompatible with the current documented boundary-amplitude
  contract. They now request `amplitude_mode="exact"`; proposal caps, seeds,
  oracle, and tolerances are unchanged. The dense-cache test also compares
  against a reference with row caching explicitly disabled. Dedicated boundary
  amplitude tests remain in the full suite. This is test maintenance, not a
  production sampling or truncation change.
- The planner's dynamic-width test supplied `("cap", (0,), "left")`, omitting
  the required state vector. It now supplies `(1., 0.)` before the absorption
  direction and retains its original candidate/warning assertions.
- Corrected 4x4 module: **10 passed**, 90.74 seconds.
  Log: `/tmp/peps-4x4-exact-fixed.log`. Planner module: **5 passed**.
- Ruff and `git diff --check` passed again after these corrections.
- Final complete rerun, without early stopping: **7170 passed, 525 skipped,
  zero failures**, 646.12 seconds. Log: `/tmp/pepsy-full-reviewed-final.log`.
  Includes the new native tableau, noisy dense-reference and MPS/STN
  trajectory checks as well as existing boundary-amplitude tests.
- The 525 skips are excluded coverage, including unavailable optional
  dependencies/hardware. Available MPI unit checks do not establish real
  multi-rank execution, and unavailable GPU paths are not hardware validation.
- Reviewed implementation, tests and API documentation; final Ruff and
  whitespace checks passed. Device-local `AGENTS.override.md` is excluded.

This completed full run supersedes the incomplete validation status in the
preceding stabilizer journals. Their earlier results remain historical evidence.

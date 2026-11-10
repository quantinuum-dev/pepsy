# 2026-11-09 — Extended Pauli damping checks and commit

- Scope: the user requested more tests and a commit of the Pauli/DAOE work.
  Geometry and 2D remain deferred.
- Branch / baseline commit: `develop`, `ad523b6`.
- Publication: this record accompanies the local Pauli/DAOE commit on
  `develop`; identify it with `git log -- history/2026-11-09-pauli-damping-validation-commit.md`.
  No push requested or performed.

## What changed

Committed the accumulated native Pauli correctness fixes, exact DAOE filter,
reusable `PauliDampingMPO`, public exports, API documentation and focused
tests described in the [review](2026-11-09-pauli-mpo-daoe.md) and
[MPO handoff](2026-11-09-pauli-damping-mpo.md).

Added dense-reference eight-period evolution/filter/compression trajectories
at zero, weak and stronger damping; spin-correlation conservation and norm
contraction checks; real/complex Torch finite-difference gradient checks;
and JAX application/export placement on a second logical CPU device.

Existing JAX sampling changes, their changelog entry and records, the noisy
optimizer journal, the earlier read-only lattice-DMT audit and device-local
`AGENTS.override.md` are excluded. Local research links in the Pauli records
were adjusted so the commit does not depend on the excluded DMT audit.

## Validation

- Four focused Pauli/damping modules: **128 passed, 1 skipped**. The skipped
  test requires a second logical CPU device and passes in the dedicated run:
  `XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu`
  with `test_jax_filter_preserves_a_nondefault_device`: **1 passed**.
- `test_mpo.py`, `test_mpo_automaton.py`, `test_mpo_trotter.py`,
  `test_mpo_alignment.py`, `test_import_boundaries.py`, `test_public_api.py`
  and `test_package_layout.py`: **235 passed, 18 failed, 8 skipped**.
  The same selection with baseline `HEAD`'s entire `pauli_mpo` module restored
  in memory reproduces the same failures and counts; no tracked files or
  shared dependencies are changed for that comparison.
- Seventeen broader failures match the
  [preceding audit](../docs/development/notes/2026-11-09-pauli-mpo-daoe.md#validation-on-this-checkout).
  The eighteenth is `test_package_version_matches_installed_distribution`:
  installed metadata 0.5.0 versus source 0.6.0. Seven skips require an absent
  upstream native Trotter scheduler; one requires unavailable CuPy.
- Ruff over `src tests`, local Markdown targets and staged diff checks pass.
  Full suite, GPU/CuPy and transport-scale convergence were not run.

The previous implementation-time handoffs describe their then-uncommitted
state; this record documents their inclusion in the requested commit.

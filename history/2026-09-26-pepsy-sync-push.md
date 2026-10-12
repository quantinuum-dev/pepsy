# 2026-09-26 — Commit and synchronize Pepsy develop

- Scope: user explicitly requested committing Pepsy, pulling remote changes,
  synchronizing, and pushing.
- Branch / starting baseline: `develop` / `80f451a`, initially ahead of the
  stale remote-tracking ref by eight commits.
- Local work committed as `eea8f7c` (sampler, alternating tree layout, qMERA
  schematics, tests, examples, and existing session records).
- Fetched `origin/develop` at `c85e3c3`: 11 incoming commits. Used a merge to
  preserve both histories. The integration decisions and validation are below.

## Integration decisions

- Retained the remote module organization and native complex MPS sampling fix.
  Ported the complete local PepsSampler implementation into sampling/peps.py
  and PEPSSampleResult additions into sampling/results.py. Kept the historical
  sampling/samplers.py compatibility module from the remote branch.
- Verified the incoming PEPS/result definitions were pure moves from the
  common base before transferring the local changes.
- Retained remote MPS helper extraction and all local exact-batch functionality.
  Adapted six exact-mode guards in extracted control, norm, and layout helpers
  to recognize exact-batch. AST checks retained all 15 locally changed/new
  definitions; five moved method bodies match after normalizing docstrings
  and the representation of the exact-mode set.
- Preserved the split tree API guide and moved the alternating x/y section
  into tree_layout.md, with a link from the compatibility landing page.
  Combined both changelog and optimizer ownership-map additions. Added local
  implementation notes to the incoming Sphinx index after strict-build checks
  identified 13 otherwise unlisted documents. Converted four journal links
  outside the Sphinx source root to repository links.
- Resolved all five merge conflicts; no remote commits or local work discarded.
  No sibling repository, production job, or shared environment was changed.

## Validation

- Earlier in this session: PEPS suite 132 passed; 32 NumPy/CUDA numerical
  cases passed. See [cutoff evidence](2026-09-26-peps-sampler-auto-cutoff.md).
- Before merge: tree layout / schematic selection 133 passed, 95 deselected.
- After merge: affected sampler, exact-batch, MPS controls/norm/layout, import,
  serialization, public API/layout selection (excluding JAX by keyword):
  467 passed, 18 deselected, one pre-existing installed-version failure.
- Ruff over src/tests/example and Git whitespace checks passed.
- Strict Sphinx HTML build passed with warnings treated as errors, after
  correcting the notes index and four source-journal links.
- Skill catalog validation: 12 skills passed.
- Full CPU suite with two forced JAX CPU devices: 4,832 passed, 103 skipped,
  14 failed in 1,295.26 seconds. One failure is the version mismatch below;
  the other 13 share a global JAX mesh conflict after NetKet import.
- Confirmed the mesh interaction in a standalone process without Pepsy:
  importing installed NetKet changes `jax.sharding.get_mesh()` from an empty
  mesh to `Mesh('S': 2, axis_types=(Auto,))`. Conjugating a committed
  single-device JAX array then raises the same incompatible-device error.
  The installed `netket/utils/config_flags.py` calls `jax.sharding.set_mesh`.
- Fresh-process rerun of all 13 affected JAX cases: 13 passed in 119.53 seconds.

The known version test compares installed/runtime 0.4.0 against project 0.5.0.
Metadata was not modified to mask it. GPU availability was disabled only for
full-suite test processes to leave GPUs available to ongoing simulations.
No whole-suite pass is claimed. No library or shared-environment workaround
was introduced for NetKet's global mesh. The full-suite mesh interaction and
installed-version mismatch remain validation limitations.

### Full-suite JAX failures

These are the cases affected by the shared two-device mesh:

- `test_optimize_qmera.py`:
  `test_qmera_2d_retained_compiled_jax_jit_gradient`,
  `test_qmera_compiled_parametric_loss_jax_jit_smoke`,
  `test_qmera_tfim_jax_jit_compiled_energy_and_gradient_match_eager`.
- `test_peps_sampler.py::test_peps_sampler_jax_nondefault_device`.
- `test_stabilizer_gpu_backend.py`:
  `test_named_replay_matches_numpy_without_host_tensor_construction[jax-StabilizerTreeSimulator]`,
  `test_exact_cooling_keeps_coefficient_tensors_on_backend[jax-StabilizerMpsSimulator]`,
  `test_exact_cooling_keeps_coefficient_tensors_on_backend[jax-StabilizerTreeSimulator]`.
- `test_stabilizer_tn.py::test_optional_array_backends_match_numpy_for_stn_paths[jax-_jax_backend-jax]`.
- `test_symmetric_tensors.py::test_fermion_fields_and_pairing_preserve_jax_backend`.
- `test_tree_gpu_backend.py`:
  `test_backend_replay_matches_numpy_and_does_not_download_gate_matrices[jax]`,
  `test_higher_order_term_sum_stays_on_backend[jax-False]`,
  `test_higher_order_term_sum_stays_on_backend[jax-True]`.
- `test_tree_operator_representation.py::test_backend_operator_conversion_keeps_local_identity_proof[jax-dmrg2]`.

## Publication

The merge commit will contain this handoff; final remote equality and working tree
cleanliness are verified after the requested normal push. No force push,
release, tag change, or main-branch operation is part of this task.

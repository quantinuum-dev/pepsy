# 2026-09-24 — Optimizer imports and direct helper ownership

- Scope: extend the approved MPS import cleanup to remaining optimizer entry
  packages and audit internal use of the tensor compatibility aggregator.
- Branch / baseline commit: `develop`, `64f7e1f`.
- Commit status: uncommitted; no staging, publication, or dependency changes.

## Implementation

- MPO, PEPS, sweep, tree, tree-PEPS, energy, and qMERA packages now resolve
  exports on demand using the existing lazy namespace pattern. Public export
  lists and order, implementation identity, static typing imports, and
  historically accessible child-module attributes are preserved.
- Gate and Hamiltonian helpers import `OneDMap` from `tensors.maps`; gates
  obtain constructors directly from `tensors.constructors`. Boundary states
  obtain backend configuration directly from `backends.config`.
- The `tensors.core` aggregator and its contraction/fidelity patch hooks are
  unchanged. Remaining wrapper callers were not blindly redirected.
- Updated the [package guide](../docs/api/package.md),
  [ownership rules](../docs/development/package_layout.md), and
  [optimizer map](../docs/development/modules/optimizers.md).

## Import measurements

Fresh-process counts of loaded Pepsy modules in the same local environment:

| Optimizer namespace | Before | After |
| --- | --- | --- |
| `mpo` | 25 | 4 |
| `peps` | 38 | 4 |
| `sweep` | 31 | 4 |
| `tree` | 41 | 4 |
| `tree_peps` | 48 | 4 |
| `energy` | 22 | 4 |
| `qmera` | 35 | 4 |

Both `qmera.geometry` and `tree_peps.plan` load seven Pepsy modules and no
NumPy, Quimb, Torch, JAX, or Symmray. Accessing optimizer implementations still
loads their dependencies. No installation-size or simulation-speed improvement
is claimed.

## Validation

- Initial domain selection: **1,117 passed, 5 skipped**, including gate,
  Hamiltonian, boundary, qMERA, tree, tree-PEPS, tree-energy, PEPS, and
  simple-update checks plus import/API/layout checks.
- Final import/API/layout selection with added regressions: **68 passed**.
  Checks cover lazy discovery, geometry serialization, direct helper imports,
  export identity, star imports, child modules, and unknown-attribute errors.
- A separate `python -I -S` probe verified namespace discovery and qMERA
  geometry construction with site packages disabled.
- Compared exports and static declarations against the saved pre-edit files:
  all seven packages preserve their existing surface. Ruff and whitespace
  checks passed; changed Markdown link targets resolve.
- Full repository run: **4,592 passed, 121 skipped, 13 failed** in 319 seconds.
  Compared failing test identifiers with the earlier full run: the same 13
  identifiers failed, with no additions or removals. This is not a clean suite
  and does not establish that every failure has the same underlying cause.
  Log: `/tmp/pepsy-import-cleanup-full.log`.

Failing identifiers retained for follow-up:

```text
tests/test_mpo.py::test_compress_mpo_product_native_dmrg_restores_mpo_boundary
tests/test_mps_replay_metadata.py::test_mixed_maximum_is_refreshed_after_quality_repair
tests/test_mps_replay_metadata.py::test_layered_fit_builds_only_visited_tag_selections_and_can_run_globally
tests/test_optimize_mps.py::test_layered_fit_resolves_boundary_bonds_locally_and_caches_them
tests/test_stabilizer_gpu_backend.py::test_exact_cooling_keeps_coefficient_tensors_on_backend[jax-StabilizerMpsSimulator]
tests/test_stabilizer_gpu_backend.py::test_exact_cooling_keeps_coefficient_tensors_on_backend[jax-StabilizerTreeSimulator]
tests/test_structural_compression.py::test_tree_pepo_compress_validates_once_after_the_full_sweep
tests/test_tree_api_consistency.py::test_mpi_only_options_cannot_be_silently_ignored[options4]
tests/test_tree_gpu_backend.py::test_backend_replay_matches_numpy_and_does_not_download_gate_matrices[jax]
tests/test_tree_gpu_backend.py::test_higher_order_term_sum_stays_on_backend[jax-False]
tests/test_tree_gpu_backend.py::test_higher_order_term_sum_stays_on_backend[jax-True]
tests/test_vmc_torch_compile.py::test_flat_z2_truncated_boundary_vmap_preserves_values_and_gradients
tests/test_vmc_torch_compile.py::test_compiled_boundary_reuse_batches_connected_local_energy
```

## Remaining scope

Direct imports through other legacy aliases and numerical module extractions
remain separate work. In particular, `TreePlan` currently shares MPS control
parsing; lazy package initialization alone does not make every layout helper
independent of replay code. Previous repository numerical failures are tracked
in the [upstream adoption assessment](../docs/development/notes/quimb_symmray_opportunities_2026_09.md).

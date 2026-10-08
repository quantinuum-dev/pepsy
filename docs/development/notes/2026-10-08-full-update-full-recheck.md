# 2026-10-08 — PEPS full-update full recheck

## Outcome and scope

No additional regression attributable to the three uncommitted PEPS full-update
corrections was found. The isolated affected suite passed **486 tests**, including
Torch CPU and CuPy GPU coverage. **The full repository suite is not clean.**

This validates the working tree on `develop`, baseline `ad8ed05`, including
concurrent MPS/JAX edits. Those edits were preserved. No implementation, test,
precision default, or dependency was changed during this recheck. The PEPS patch
saved at the start and end of validation is byte-identical. Nothing was staged,
committed, or published.

See the [correction evidence](2026-10-08-full-update-review-corrections.md) for
the implemented fixes, upstream audit, and earlier scheduler measurements.
This report records new checks; it does not supersede their scoped conclusions.

## Coverage and failure reproduction

| Check | Result |
| --- | --- |
| Unique collected cases exercised | 8,269 / 8,269 |
| Initial broad results, deduplicated | 8,129 passed, 133 failed, 7 skipped |
| Additional collection skips | 2 MPI modules; mpi4py unavailable |
| Initially failing cases rerun in fresh processes | 133 / 133 |
| Passed on fresh rerun | 30 |
| Still failed on fresh rerun | 103 |
| Persistent failures also reproduced on unchanged ad8ed05 | 103 / 103 |
| Fresh isolated affected suite | 486 passed, 98 warnings, no skips; 117.71 s |
| Higher-precision JAX diagnostic | 3 representative cases passed; 42.90 s |

The 103 are failing test cases, not 103 independently established defects.
Baseline reproduction establishes that the PEPS corrections did not introduce
these failures in this environment; it does not establish their root causes.

The broad check used activated `py312`, with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1`,
`LOKY_MAX_CPU_COUNT=2 MPLBACKEND=Agg`, and
`python -m pytest -q -ra -o addopts=''`.

The exact native U1U1 Hubbard test took about 876 seconds. A second process ran
the 6,051 IDs after `test_mps_fermions.py` while the first covered the initial
2,218. After finishing its assigned range, the original process was deliberately
interrupted during duplicate work. Its completed results were retained:
2,331 passed and 53 failed, plus the two collection skips. The second process
finished with 5,964 passed, 80 failed, and 7 skipped. All 166 duplicate IDs had
identical outcomes. Comparing the union against the original collection
confirmed no missing case. A nameless JUnit interruption placeholder was
excluded. This was complete collected-case coverage across batches, **not an
uninterrupted passing full-suite invocation**.

Concurrent GPU use caused memory contention: the original JAX process held
about 18.7 GB on a 24 GB GPU. All 133 failures were therefore repeated after that
process exited, with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Fresh batches gave
59 failed / 13 passed, 2 failed / 15 passed, and 42 failed / 2 passed. The 30
recoveries include both initially failing CuPy full-update cases. Resource
contention and process state/order are relevant; the evidence does not assign
every recovery to a specific cause.

The matching baseline archive was
`/tmp/pepsy_full_recheck_baseline_20261008_ad8ed05`. Its test configuration
imports the archive's own source. The same fresh-process controls reproduced
all remaining failures in baseline batches of 59, 2, and 42. Earlier baseline
probes made during GPU contention are provisional and are not used for this
comparison.

## Persistent failure groups

| Observed failure category | Cases |
| --- | ---: |
| BP convergence / fixed-point prerequisites | 37 |
| Torch compilation rejecting Autoray namespace mutation | 6 |
| MPS replay host scalar conversion guards | 6 |
| Native fermionic CTMRG matrix dimensionality errors | 2 |
| JAX numerical / backend comparisons | 52 |
| Total | 103 |

BP failures include false convergence results and loop-series operations
rejecting unconverged messages. The compilation failures report Torch Dynamo
rejecting Autoray AutoNamespace mutation through `__dict__.__setitem__`.
Native fermionic CTMRG fails with
`ValueError: Matrix multiplication requires <=2D arrays`.

Several JAX comparisons differ by about 1e-4 against tighter tolerances.
For example, the two PEPS sampler efficiency cases disagree on 40/64 and
34/64 probabilities against `rtol=3e-5, atol=3e-6`; a tree fidelity comparison
returns about 0.9997629 against 1 ± 2e-6.

A diagnostic setting `JAX_DEFAULT_MATMUL_PRECISION=highest` in a fresh process
made these three previously failing cases pass:

- `tests/test_tree_gpu_backend.py::test_backend_replay_matches_numpy_and_does_not_download_gate_matrices[jax]`
- `tests/test_cluster_jit_gradients.py::test_complete_fixed_two_site_jax_jit_real_time_gradient[pepo]`
- `tests/test_peps_sampler_efficiency.py::test_factored_proposal_matches_reference_and_preserves_source[dmrg-jax]`

This supports a precision-policy explanation for those cases. The other 49 JAX
comparison failures were not rerun with that setting, so it is not a verified
solution for the whole group. No repository default or assertion was changed.

| Test module | Persistent cases |
| --- | ---: |
| `tests/test_bp_compression.py` | 9 |
| `tests/test_bp_open_series.py` | 15 |
| `tests/test_bp_symmray.py` | 13 |
| `tests/test_cluster_channel_automaton.py` | 2 |
| `tests/test_cluster_channel_pepo.py` | 4 |
| `tests/test_cluster_correctness_review.py` | 1 |
| `tests/test_cluster_jit_gradients.py` | 7 |
| `tests/test_cluster_trace.py` | 1 |
| `tests/test_mps_gpu_backend.py` | 7 |
| `tests/test_peps_sampler_efficiency.py` | 2 |
| `tests/test_stabilizer_gpu_backend.py` | 3 |
| `tests/test_symmetric_tensors.py` | 2 |
| `tests/test_tree_fit_messages.py` | 1 |
| `tests/test_tree_gpu_backend.py` | 35 |
| `tests/test_tree_successive_compression.py` | 1 |

## Skipped coverage and static checks

The seven individual skips cover unavailable Metal (one), insufficient logical
CPU/JAX devices (four), insufficient CUDA devices (one), and a CuPy float32
subnormal-flushing case (one). Separately, `test_mpi_integration.py` and
`test_mps_trajectory_mpi_gpu.py` were skipped at collection because mpi4py
is unavailable.

Ruff (`python -m ruff check src tests`), `git diff --check`, and the skill
catalog validator (12 skills) passed. Relative links in this report and its
handoff were checked.

## Local execution artifacts

These temporary artifacts can disappear; the essential counts and persistent
case IDs are retained here.

- Collection: `/tmp/pepsy_full_recheck_collection.log`.
- Broad batches: `/tmp/pepsy_full_recheck_20261008.{log,xml}` and
  `/tmp/pepsy_full_recheck_remaining.{log,xml}`.
- Affected suite: `/tmp/pepsy_full_recheck_focused.{log,xml}`.
- Fresh current reruns: `/tmp/pepsy_recheck_clean_current.{log,xml}`,
  `/tmp/pepsy_recheck_clean_later.{log,xml}`, and
  `/tmp/pepsy_recheck_clean_extra.{log,xml}`.
- Authoritative baseline reruns: `/tmp/pepsy_recheck_clean_baseline.{log,xml}`,
  `/tmp/pepsy_recheck_clean_later_baseline.{log,xml}`, and
  `/tmp/pepsy_recheck_clean_extra_baseline.{log,xml}`.
- Precision probe: `/tmp/pepsy_recheck_jax_precision_probe.{log,xml}`.
- Deduplication/comparison: `/tmp/pepsy_recheck_finalize.py` and
  `/tmp/pepsy_full_recheck_summary.json`.
- Patch comparison: `/tmp/pepsy_full_recheck_start.patch` and
  `/tmp/pepsy_full_recheck_end.patch`.

## Persistent case IDs

Each ID below failed in fresh current-tree and unchanged-baseline processes
under the original precision setting. They can be passed directly to pytest.

```text
tests/test_bp_compression.py::test_compress_all_gauge_defaults_to_zero_edge_excitations[sequential]
tests/test_bp_compression.py::test_compress_all_gauge_is_the_public_all_bond_convenience_wrapper
tests/test_bp_compression.py::test_cut_edge_loop_series_compression_forwards_cost_policy
tests/test_bp_compression.py::test_parallel_simultaneous_sweep_compresses_all_bonds_and_selects_als_start[pepo]
tests/test_bp_compression.py::test_parallel_simultaneous_sweep_compresses_all_bonds_and_selects_als_start[peps]
tests/test_bp_compression.py::test_sequential_loop_series_compression_reuses_projected_messages
tests/test_bp_compression.py::test_sequential_loop_series_refreshes_topology_cache_after_reduction
tests/test_bp_compression.py::test_simultaneous_loop_series_compression_uses_one_boundary_snapshot
tests/test_bp_compression.py::test_simultaneous_serial_sweep_honors_initializer_candidates
tests/test_bp_open_series.py::test_corridor_mode_can_use_compressed_boundary_contraction
tests/test_bp_open_series.py::test_corridor_mode_limits_geometry_before_contraction
tests/test_bp_open_series.py::test_edge_cutoff_is_the_explicit_name_for_dense_open_series
tests/test_bp_open_series.py::test_open_measurement_diagnostic_selects_auto_route_and_reuses_terms
tests/test_bp_open_series.py::test_open_rho_series_is_exact_for_a_tree_with_a_multi_site_support
tests/test_bp_open_series.py::test_open_rho_series_keeps_the_long_range_path_and_is_exact_on_a_tree
tests/test_bp_open_series.py::test_open_rho_series_reuses_one_d2bp_message_set
tests/test_bp_open_series.py::test_open_rho_series_sweep_accepts_route_specific_edge_cutoffs
tests/test_bp_open_series.py::test_open_rho_series_sweep_reuses_bp_across_supports_and_cutoffs
tests/test_bp_open_series.py::test_open_scalar_series_inserts_a_two_site_gate_and_normalizes_after_sum
tests/test_bp_open_series.py::test_open_scalar_series_reports_and_applies_contraction_cost_limits
tests/test_bp_open_series.py::test_open_series_enumeration_limits_raise_before_partial_contraction
tests/test_bp_open_series.py::test_open_series_production_result_reports_budget_and_resources
tests/test_bp_open_series.py::test_open_series_rejects_mixed_edge_and_cluster_cutoffs
tests/test_bp_open_series.py::test_rho_diagnostic_and_adaptive_corridor_ladder
tests/test_bp_symmray.py::test_cyclic_native_open_series_honors_contraction_cost_limits
tests/test_bp_symmray.py::test_explicit_edge_loop_series_preserves_dense_edge_degree_terms
tests/test_bp_symmray.py::test_explicit_edge_loop_series_uses_fermion_safe_gate_path
tests/test_bp_symmray.py::test_fermionic_local_expectation_loop_series_aligns_charge_support
tests/test_bp_symmray.py::test_fermionic_open_rho_sweep_supports_native_symmetries[U1-2-None]
tests/test_bp_symmray.py::test_fermionic_open_rho_sweep_supports_native_symmetries[U1U1-4-_site_charge]
tests/test_bp_symmray.py::test_fermionic_open_rho_sweep_supports_native_symmetries[Z2-2-None]
tests/test_bp_symmray.py::test_fermionic_u1_two_norm_bp_is_exact_on_a_tree[2]
tests/test_bp_symmray.py::test_fermionic_u1_two_norm_bp_is_exact_on_a_tree[3]
tests/test_bp_symmray.py::test_native_symmray_closed_scalar_network_works_through_one_norm_apis
tests/test_bp_symmray.py::test_open_scalar_series_native_gate_phase_matches_symmetry[U1-4-None-density-False]
tests/test_bp_symmray.py::test_open_scalar_series_native_gate_phase_matches_symmetry[U1U1-4-_site_charge-hopping-True]
tests/test_bp_symmray.py::test_open_scalar_series_native_gate_phase_matches_symmetry[Z2-2-None-density-False]
tests/test_cluster_channel_automaton.py::test_torch_fullgraph_replay_and_jax_trace_gradients[3]
tests/test_cluster_channel_automaton.py::test_torch_fullgraph_replay_and_jax_trace_gradients[None]
tests/test_cluster_channel_pepo.py::test_symbolic_jax_trace_and_torch_assembly_compile[algebraic-graph]
tests/test_cluster_channel_pepo.py::test_symbolic_jax_trace_and_torch_assembly_compile[algebraic-square]
tests/test_cluster_channel_pepo.py::test_symbolic_jax_trace_and_torch_assembly_compile[symbolic-graph]
tests/test_cluster_channel_pepo.py::test_symbolic_jax_trace_and_torch_assembly_compile[symbolic-square]
tests/test_cluster_correctness_review.py::test_fixed_positional_jax_gradient_through_target_alignment
tests/test_cluster_jit_gradients.py::test_complete_fixed_located_square_jax_jit_reused_lower_gradients
tests/test_cluster_jit_gradients.py::test_complete_fixed_square_order_four_jax_jit_gradient[mpo]
tests/test_cluster_jit_gradients.py::test_complete_fixed_square_order_four_jax_jit_gradient[pepo]
tests/test_cluster_jit_gradients.py::test_complete_fixed_three_site_jax_jit_values_and_gradients[mpo]
tests/test_cluster_jit_gradients.py::test_complete_fixed_three_site_jax_jit_values_and_gradients[pepo]
tests/test_cluster_jit_gradients.py::test_complete_fixed_two_site_jax_jit_real_time_gradient[pepo]
tests/test_cluster_jit_gradients.py::test_joint_fixed_two_site_jax_jit_values_and_gradients[pepo]
tests/test_cluster_trace.py::test_joint_two_site_trace_jax_jit_gradients[pepo]
tests/test_mps_gpu_backend.py::test_backend_ledger_matches_cpu_and_preserves_state_dtype[jax-False]
tests/test_mps_gpu_backend.py::test_unitary_ledger_reads_only_one_boolean_per_replay[direct]
tests/test_mps_gpu_backend.py::test_unitary_ledger_reads_only_one_boolean_per_replay[dmrg2]
tests/test_mps_gpu_backend.py::test_unitary_ledger_reads_only_one_boolean_per_replay[mix]
tests/test_mps_gpu_backend.py::test_unitary_ledger_reads_only_one_boolean_per_replay[perm]
tests/test_mps_gpu_backend.py::test_unitary_ledger_reads_only_one_boolean_per_replay[svd]
tests/test_mps_gpu_backend.py::test_unitary_ledger_reads_only_one_boolean_per_replay[swap]
tests/test_peps_sampler_efficiency.py::test_factored_proposal_matches_reference_and_preserves_source[dmrg-jax]
tests/test_peps_sampler_efficiency.py::test_factored_proposal_matches_reference_and_preserves_source[quimb-mps-jax]
tests/test_stabilizer_gpu_backend.py::test_named_replay_matches_numpy_without_host_tensor_construction[jax-StabilizerMpsSimulator]
tests/test_stabilizer_gpu_backend.py::test_named_replay_matches_numpy_without_host_tensor_construction[jax-StabilizerTreeSimulator]
tests/test_stabilizer_gpu_backend.py::test_pauli_builders_construct_on_selected_backend[jax]
tests/test_symmetric_tensors.py::test_native_fermionic_ctmrg_matches_exact_on_small_double_layer[None]
tests/test_symmetric_tensors.py::test_native_fermionic_ctmrg_matches_exact_on_small_double_layer[layered]
tests/test_tree_fit_messages.py::test_auto_path_cached_sweeps_match_full_branches_with_rank_changes[jax]
tests/test_tree_gpu_backend.py::test_backend_replay_matches_numpy_and_does_not_download_gate_matrices[jax]
tests/test_tree_gpu_backend.py::test_branch_final_center_and_following_path_preserve_backend[jax-direct]
tests/test_tree_gpu_backend.py::test_branch_final_center_and_following_path_preserve_backend[jax-dm]
tests/test_tree_gpu_backend.py::test_fit_target_norm_without_ledger_shortcut[jax-False-0.0-dmrg1]
tests/test_tree_gpu_backend.py::test_fit_target_norm_without_ledger_shortcut[jax-False-0.0-dmrg2]
tests/test_tree_gpu_backend.py::test_fit_target_norm_without_ledger_shortcut[jax-False-0.0-dmrg3]
tests/test_tree_gpu_backend.py::test_fit_target_norm_without_ledger_shortcut[jax-False-0.0-dmrg]
tests/test_tree_gpu_backend.py::test_fit_target_norm_without_ledger_shortcut[jax-False-0.0-mix]
tests/test_tree_gpu_backend.py::test_fit_target_norm_without_ledger_shortcut[jax-False-1.0-dmrg1]
tests/test_tree_gpu_backend.py::test_fit_target_norm_without_ledger_shortcut[jax-False-1.0-dmrg2]
tests/test_tree_gpu_backend.py::test_fit_target_norm_without_ledger_shortcut[jax-False-1.0-dmrg3]
tests/test_tree_gpu_backend.py::test_fit_target_norm_without_ledger_shortcut[jax-False-1.0-dmrg]
tests/test_tree_gpu_backend.py::test_fit_target_norm_without_ledger_shortcut[jax-False-1.0-mix]
tests/test_tree_gpu_backend.py::test_fit_target_norm_without_ledger_shortcut[jax-True-1.0-dmrg1]
tests/test_tree_gpu_backend.py::test_fit_target_norm_without_ledger_shortcut[jax-True-1.0-dmrg2]
tests/test_tree_gpu_backend.py::test_fit_target_norm_without_ledger_shortcut[jax-True-1.0-dmrg3]
tests/test_tree_gpu_backend.py::test_fit_target_norm_without_ledger_shortcut[jax-True-1.0-dmrg]
tests/test_tree_gpu_backend.py::test_fit_target_norm_without_ledger_shortcut[jax-True-1.0-mix]
tests/test_tree_gpu_backend.py::test_higher_order_term_sum_stays_on_backend[jax-False]
tests/test_tree_gpu_backend.py::test_higher_order_term_sum_stays_on_backend[jax-True]
tests/test_tree_gpu_backend.py::test_kraus_probabilities_use_exact_local_readout[jax--400.0-where0]
tests/test_tree_gpu_backend.py::test_kraus_probabilities_use_exact_local_readout[jax--400.0-where2]
tests/test_tree_gpu_backend.py::test_kraus_probabilities_use_exact_local_readout[jax-0.0-where0]
tests/test_tree_gpu_backend.py::test_kraus_probabilities_use_exact_local_readout[jax-0.0-where2]
tests/test_tree_gpu_backend.py::test_kraus_probabilities_use_exact_local_readout[jax-400.0-where0]
tests/test_tree_gpu_backend.py::test_kraus_probabilities_use_exact_local_readout[jax-400.0-where2]
tests/test_tree_gpu_backend.py::test_random_fit_uses_backend_generator_and_is_reproducible[jax-random]
tests/test_tree_gpu_backend.py::test_random_fit_uses_backend_generator_and_is_reproducible[jax-random_expand]
tests/test_tree_gpu_backend.py::test_represented_norm_preserves_exponent_range[jax--100.0]
tests/test_tree_gpu_backend.py::test_represented_norm_preserves_exponent_range[jax-100.0]
tests/test_tree_gpu_backend.py::test_stabilization_with_extreme_scale_keeps_backend[jax--400.0-direct]
tests/test_tree_gpu_backend.py::test_stabilization_with_extreme_scale_keeps_backend[jax--400.0-dmrg2]
tests/test_tree_gpu_backend.py::test_stabilization_with_extreme_scale_keeps_backend[jax-400.0-direct]
tests/test_tree_gpu_backend.py::test_stabilization_with_extreme_scale_keeps_backend[jax-400.0-dmrg2]
tests/test_tree_gpu_backend.py::test_warm_control_replay_builds_operators_on_backend[jax]
tests/test_tree_successive_compression.py::test_layered_tree_compression_uses_real_algorithm[sdcr-jax-2]
```

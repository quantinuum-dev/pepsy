"""Basic public API smoke tests for the pepsy package."""

import importlib.util

import pepsy


def test_package_version_available():
    """Package exposes a non-empty version string."""
    assert isinstance(pepsy.__version__, str)
    assert pepsy.__version__


def test_namespace_exports_have_clear_core_and_advanced_groups():
    """Core and advanced namespaces remain discoverable and distinct."""
    core = {
        "backends",
        "boundary",
        "fitting",
        "operators",
        "optimizers",
        "sampling",
        "solvers",
        "tensors",
    }
    advanced = {"bp", "experimental", "vmc"}

    assert core.isdisjoint(advanced)
    assert core | advanced <= set(pepsy.__all__)
    assert all(getattr(pepsy, name) is not None for name in core | advanced)


def test_tree_optimizers_are_available_from_high_level_api():
    """Tree layout and execution helpers resolve from ``import pepsy as py``."""
    from pepsy.optimizers.tree import TreeLayoutFinder, TreeOptimizer, TreePlan
    from pepsy.optimizers.tree_stabilizer import TreeStabOptimizer

    assert pepsy.TreeLayoutFinder is TreeLayoutFinder
    assert pepsy.TreeOptimizer is TreeOptimizer
    assert pepsy.TreePlan is TreePlan
    assert pepsy.TreeStabOptimizer is TreeStabOptimizer


_EXPECTED_IN_ALL = [
    "backends", "boundary", "experimental", "fitting", "operators", "optimizers",
    "sampling", "solvers", "tensors", "vmc",
    "BdyMPS", "CompBdy", "BoundaryContractResult", "BoundaryFitDiagnostic", "contract_boundary",
    "contract_flat", "build_bra_ket", "normalize", "peps_normalize", "boundary_norm", "infidelity",
    "peps_norm", "peps_infidelity", "peps_fidelity", "GlobalOptimizer", "FIT",
    "tns_align", "measure_obs", "build_pepo_from_gates", "build_mpo_from_gates",
    "pauli", "x", "y", "z", "s", "sdg", "t", "tdg", "h", "hadamard",
    "cnot", "cx", "cy", "cz", "swap", "iswap", "phase", "u1", "u2",
    "cphase", "crx", "cry", "crz", "cu1", "cu2", "cu3", "rx", "ry", "rz",
    "rxx", "ryy", "rzz", "u3", "su4", "fsim", "fsimg", "haar_random_state", "hrs_to_mps", "hrs_to_peps", "hrs_to_ttn", "ps_to_peps", "ps_to_3dpeps", "expec_mpo",
    "id_to_mpo", "id_to_pepo", "ps_to_pepo", "ps_to_mpo", "ps_to_ttn", "make_numpy_array_caster", "backend_infer", "to_float", "SweepOptimizer",
    "FDSolver", "MpsEnergyOptimizer", "MpsOptimizer", "MpoOptimizer", "PepsEnergyOptimizer", "PepsOptimizer", "SimpleUpdateGen", "SymDMRG2", "PEPSSampleResult",
    "PepsBpSampler", "MpsSampler", "FermionConfigurationEncoding", "MpsDiagonalEstimate", "MpsBatchSampleResult", "MpsSampleResult", "VecSampler", "gate", "gauge_all", "gauge_all_simple", "compress_all_gauge", "one_norm_bp", "tn_fidelity", "tn_norm",
    "TreeSampler", "TreeBatchSampleResult", "TreeSampleResult",
    "MpsStabOptimizer", "STNState", "StabilizerMpsSimulator",
    "SimulatorCandidate", "SimulatorPlan", "SimulatorPlanner", "recommend_simulator",
    "TreeEnergyOptimizer",
    "TreeLayoutFinder",
    "TreeMPO", "TreeOptimizer", "build_tree_operator", "tree_mpo",
    "TreePlan",
    "TreeStabOptimizer",
    "TreeTensorNetwork",
    "DeferredInjectionRecord", "DeferredInjectionReport", "DeferredProjectionRecord",
    "ImmediateInjectionReport", "ImmediateProjectionRecord", "MeasurementRecord", "NormEventRecord",
    "StabilizerMpsSettingsAdvice", "StabilizerMpsRunResult",
    "StabilizerTreeRunResult", "StreamAnalysisRecord",
    "CoalescedMeasurementRecord", "CoalescedSampleResult", "CoalescedTrajectoryLeaf", "CoalescedTrajectoryResult", "LeakageRecord", "NoisyResult", "MpsNoisyResult", "NoisyShotResult", "PauliErrorModel", "PauliFault",
    "StimCircuitPlan", "StimDetector", "StimHerald", "StimNoiseSample", "StimObservable", "StimObservableRecord", "StimShotResult", "StimSyndromeRecord",
    "CoherentCrosstalkModel", "TrajectoryChannel", "TrajectoryEvent", "TrajectoryMeasurementRecord", "TrajectoryOutcome", "TrajectoryRecord", "TrajectorySample", "TrajectoryShotResult",
    "compile_stim_circuit", "run_coalesced_noisy_shots", "run_coalesced_stim_shots", "run_coalesced_trajectory_shots", "MpsNoisy", "TreeNoisy", "run_noisy_shots", "run_stabilizer_mps_stream", "run_stim_shots", "run_trajectory_shots",
    "sample_coalesced_bits", "sample_noisy_gate_stream", "sample_noisy_gate_streams", "sample_stim_circuit", "sample_stim_circuits", "sample_trajectory_stream",
    "Fermion", "FermionLatticeSetup", "SpinfulFermion", "SpinfulFermionHubbard", "SymmFermions", "SymGateStream", "SymHamiltonian", "SymMPS", "SymPEPS",
    "default_physical_sectors", "draw_symmray_blocks", "draw_symmray_mps", "draw_symmray_mpo", "draw_symmray_peps",
    "fermi_hubbard_u1u1_gate_stream", "fermi_hubbard_u1u1_hopping_gate_stream",
    "fermi_hubbard_u1u1_interaction_gate_stream", "fermi_hubbard_u1u1_light_pulse_gate_stream",
    "fermi_hubbard_u1u1_jw_gate_stream", "fermi_hubbard_u1u1_jw_hopping_gate_stream",
    "fermi_hubbard_u1u1_jw_interaction_gate_stream",
    "fermion_density_param_gen", "fermion_hopping_param_gen", "fermion_interaction_param_gen",
    "sector_index_map",
    "site_charge_alternating", "site_charge_from_map",
    "site_charge_from_occupations", "site_charge_uniform",
    "symmray_block_summary", "symmray_mps_summary", "symmray_mpo_summary", "symmray_peps_summary", "symm_operator_from_dense",
    "reg_native_svd_torch", "reg_native_svd_jax", "reg_rel_svd_torch",
    "reg_real_svd_torch", "reg_complex_svd_torch",
    "reg_real_qr_torch", "reg_complex_qr_torch",
    "reg_rel_svd_jax", "reg_real_svd_jax", "reg_complex_svd_jax",
    "register_jax_linalg", "register_torch_linalg", "reset_linalg_registrations",
    "TorchLinalgConfig", "get_torch_linalg_config",
]

_EXPECTED_NOT_IN_ALL = [
    "norm_peps", "normalize_peps", "loss_peps", "PEPSGlobalOptimizer",
    "gen_long_range_swap_path",
    "gen_long_range_swap_path_1d", "gen_long_range_swap_path_2d",
    "gen_long_range_swap_path_3d", "gate_tn_1d", "gate_tn_2d", "gate_tn_3d",
    "gates_tn_1d", "gates_tn_2d", "gates_tn_3d", "apply_2d_gate",
    "apply_2d_gates", "apply_2dtn_", "gate_2d", "gate_to_pepo", "gate_1d",
    "canonize_mps", "apply_gates_", "expec_TN_1D", "peps_I",
    "reg_stop_gradient_torch", "stop_grad",
    "MPSOptimizer", "MPOOptimizer",
    "boundary_metrics", "boundary_states", "boundary_sweeps", "core", "fit",
    "ft_solver", "gates", "gradient_solver", "ham", "optimize_energy",
    "optimize_global", "optimize_mpo", "optimize_mps", "optimize_sweep",
    "sampler",
]


def test_symbols_exported():
    """All documented public symbols should be in ``__all__``."""
    assert set(_EXPECTED_IN_ALL).issubset(pepsy.__all__)


def test_internal_symbols_not_exported():
    """Internal symbols should not leak into ``__all__``."""
    assert set(_EXPECTED_NOT_IN_ALL).isdisjoint(pepsy.__all__)


_CALLABLE_EXPORTS = [
    "contract_boundary", "contract_flat", "build_bra_ket", "normalize", "peps_normalize",
    "boundary_norm", "peps_norm", "infidelity", "peps_infidelity", "peps_fidelity",
    "backend_infer", "to_float", "gauge_all", "gauge_all_simple", "compress_all_gauge", "one_norm_bp",
    "GlobalOptimizer", "FIT", "tns_align", "measure_obs",
    "build_pepo_from_gates", "build_mpo_from_gates", "pauli",
    "x", "y", "z", "s", "sdg", "t", "tdg", "h", "hadamard",
    "cnot", "cx", "cy", "cz", "swap", "iswap", "phase", "u1", "u2",
    "cphase", "crx", "cry", "crz", "cu1", "cu2", "cu3", "rx", "ry", "rz",
    "rxx", "ryy", "rzz", "u3", "su4", "fsim", "fsimg", "haar_random_state", "hrs_to_mps", "hrs_to_peps", "hrs_to_ttn", "ps_to_peps", "ps_to_3dpeps", "expec_mpo",
    "id_to_mpo", "id_to_pepo", "ps_to_pepo", "ps_to_mpo", "ps_to_ttn", "SweepOptimizer",
    "FDSolver", "MpsEnergyOptimizer", "MpsOptimizer", "MpoOptimizer", "MpsStabOptimizer", "StabilizerMpsSimulator",
    "SimulatorCandidate", "SimulatorPlan", "SimulatorPlanner", "recommend_simulator",
    "DeferredInjectionRecord", "DeferredInjectionReport", "DeferredProjectionRecord",
    "ImmediateInjectionReport", "ImmediateProjectionRecord", "MeasurementRecord", "NormEventRecord",
    "StabilizerMpsSettingsAdvice", "StabilizerMpsRunResult", "StreamAnalysisRecord",
    "PepsEnergyOptimizer", "PepsOptimizer", "SimpleUpdateGen", "SymDMRG2", "PEPSSampleResult", "PepsBpSampler", "CoherentCrosstalkModel", "NoisyResult", "MpsNoisyResult", "compile_stim_circuit", "run_coalesced_noisy_shots", "run_coalesced_stim_shots", "run_coalesced_trajectory_shots", "MpsNoisy", "TreeNoisy", "run_noisy_shots", "run_stabilizer_mps_stream", "run_stabilizer_tree_stream", "run_stim_shots", "run_trajectory_shots", "sample_coalesced_bits", "sample_noisy_gate_stream", "sample_noisy_gate_streams", "sample_stim_circuit", "sample_stim_circuits", "sample_trajectory_stream",
    "TreeEnergyOptimizer",
    "TreeLayoutFinder",
    "TreeMPO", "TreeOptimizer", "build_tree_operator", "tree_mpo",
    "TreePlan",
    "TreeStabOptimizer",
    "TreeTensorNetwork",
    "TreeSampler", "TreeBatchSampleResult", "TreeSampleResult",
    "tn_fidelity", "tn_norm", "Fermion", "FermionLatticeSetup", "SpinfulFermion", "SpinfulFermionHubbard", "SymmFermions", "SymGateStream", "SymHamiltonian", "SymMPS", "SymPEPS",
    "default_physical_sectors", "draw_symmray_blocks", "draw_symmray_mps", "draw_symmray_mpo", "draw_symmray_peps",
    "fermi_hubbard_u1u1_gate_stream", "fermi_hubbard_u1u1_hopping_gate_stream",
    "fermi_hubbard_u1u1_interaction_gate_stream", "fermi_hubbard_u1u1_light_pulse_gate_stream",
    "fermi_hubbard_u1u1_jw_gate_stream", "fermi_hubbard_u1u1_jw_hopping_gate_stream",
    "fermi_hubbard_u1u1_jw_interaction_gate_stream",
    "fermion_density_param_gen", "fermion_hopping_param_gen", "fermion_interaction_param_gen",
    "sector_index_map",
    "site_charge_alternating", "site_charge_from_map",
    "site_charge_from_occupations", "site_charge_uniform",
    "symmray_block_summary", "symmray_mps_summary", "symmray_mpo_summary", "symmray_peps_summary", "symm_operator_from_dense",
    "reg_native_svd_torch", "reg_native_svd_jax", "reg_rel_svd_torch",
    "reg_real_svd_torch", "reg_complex_svd_torch",
    "reg_real_qr_torch", "reg_complex_qr_torch",
    "reg_rel_svd_jax", "reg_real_svd_jax", "reg_complex_svd_jax",
    "register_jax_linalg", "reset_linalg_registrations",
]

_BLOCKED_NAMES = _EXPECTED_NOT_IN_ALL

_MODULE_EXPORTS = [
    "backends", "boundary", "experimental", "fitting", "operators", "optimizers",
    "sampling", "solvers", "tensors", "vmc",
]


def test_all_exports_are_unique():
    """Public export list should not contain duplicate names."""
    assert len(pepsy.__all__) == len(set(pepsy.__all__))


def test_all_exports_resolve():
    """Every advertised public export should resolve."""
    missing = [name for name in pepsy.__all__ if getattr(pepsy, name, None) is None]
    assert not missing, f"unresolved public exports: {missing}"


def test_lazy_callables_resolve():
    """Lazy callable exports should resolve to callables."""
    invalid = [name for name in _CALLABLE_EXPORTS if not callable(getattr(pepsy, name))]
    assert not invalid, f"non-callable public exports: {invalid}"


def test_blocked_names_raise():
    """Internal names should not be advertised through ``__all__``."""
    leaked = [name for name in _BLOCKED_NAMES if name in pepsy.__all__]
    assert not leaked, f"blocked names leaked into package exports: {leaked}"


def test_module_exports_resolve():
    """Submodule export should resolve to a non-None value."""
    missing = [name for name in _MODULE_EXPORTS if getattr(pepsy, name, None) is None]
    assert not missing, f"missing module exports: {missing}"


def test_vmc_torch_package_preserves_lazy_public_exports():
    """The Torch VMC package and lazy VMC namespace expose the same objects."""
    import pepsy.vmc as vmc
    import pepsy.vmc.torch as torch_vmc

    assert torch_vmc.__spec__.submodule_search_locations is not None
    assert torch_vmc.__all__
    for name in torch_vmc.__all__:
        assert getattr(torch_vmc, name) is getattr(vmc, name)


def test_optional_linalg_registrations_resolve():
    """Linalg registrations resolve under tensor namespaces and public wrappers."""
    has_torch = importlib.util.find_spec("torch") is not None
    has_jax = importlib.util.find_spec("jax") is not None
    assert callable(pepsy.tensors.core.reg_stop_gradient_torch)
    assert callable(pepsy.tensors.core.stop_grad)
    assert callable(pepsy.tensors.reg_stop_gradient_torch)
    assert callable(pepsy.tensors.stop_grad)
    assert pepsy.reg_rel_svd_torch is pepsy.tensors.reg_rel_svd_torch
    assert pepsy.reg_real_svd_torch is pepsy.tensors.reg_real_svd_torch
    assert pepsy.reg_complex_svd_torch is pepsy.tensors.reg_complex_svd_torch
    assert pepsy.reg_native_svd_torch is pepsy.tensors.reg_native_svd_torch
    assert pepsy.reg_native_svd_jax is pepsy.tensors.reg_native_svd_jax
    assert pepsy.reg_real_qr_torch is pepsy.tensors.reg_real_qr_torch
    assert pepsy.reg_complex_qr_torch is pepsy.tensors.reg_complex_qr_torch
    assert pepsy.reg_rel_svd_jax is pepsy.tensors.reg_rel_svd_jax
    assert pepsy.reg_real_svd_jax is pepsy.tensors.reg_real_svd_jax
    assert pepsy.reg_complex_svd_jax is pepsy.tensors.reg_complex_svd_jax
    assert pepsy.register_jax_linalg is pepsy.backends.register_jax_linalg
    assert pepsy.register_torch_linalg is pepsy.backends.register_torch_linalg
    assert pepsy.TorchLinalgConfig is pepsy.backends.TorchLinalgConfig
    assert pepsy.get_torch_linalg_config is pepsy.backends.get_torch_linalg_config
    assert pepsy.reset_linalg_registrations is pepsy.backends.reset_linalg_registrations
    if has_torch:
        import torch

        assert callable(pepsy.tensors.core.reg_rel_svd_torch)
        assert callable(pepsy.tensors.reg_rel_svd_torch)
        assert callable(pepsy.tensors.core.reg_native_svd_torch)
        assert callable(pepsy.tensors.reg_native_svd_torch)
        assert callable(pepsy.tensors.core.reg_real_svd_torch)
        assert callable(pepsy.tensors.reg_real_svd_torch)
        assert callable(pepsy.tensors.core.reg_complex_svd_torch)
        assert callable(pepsy.tensors.reg_complex_svd_torch)
        assert callable(pepsy.tensors.core.reg_real_qr_torch)
        assert callable(pepsy.tensors.reg_real_qr_torch)
        assert callable(pepsy.tensors.core.reg_complex_qr_torch)
        assert callable(pepsy.tensors.reg_complex_qr_torch)
        x = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        y = pepsy.tensors.stop_grad(x)
        assert not y.requires_grad
        assert y is not x
        assert y.data_ptr() != x.data_ptr()
    if has_jax:
        assert callable(pepsy.tensors.core.reg_rel_svd_jax)
        assert callable(pepsy.tensors.reg_rel_svd_jax)
        assert callable(pepsy.tensors.core.reg_real_svd_jax)
        assert callable(pepsy.tensors.reg_real_svd_jax)
        assert callable(pepsy.tensors.core.reg_complex_svd_jax)
        assert callable(pepsy.tensors.reg_complex_svd_jax)
        assert callable(pepsy.tensors.core.reg_native_svd_jax)
        assert callable(pepsy.tensors.reg_native_svd_jax)
        assert callable(pepsy.tensors.core.register_jax_linalg)
        assert callable(pepsy.tensors.register_jax_linalg)

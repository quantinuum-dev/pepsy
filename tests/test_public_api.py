"""Basic public API smoke tests for the pepsy package."""

import importlib.util

import pepsy
import pytest


def test_package_version_available():
    """Package exposes a non-empty version string."""
    assert isinstance(pepsy.__version__, str)
    assert pepsy.__version__


_EXPECTED_IN_ALL = [
    "backends", "boundary", "fitting", "operators", "optimizers",
    "sampling", "solvers", "tensors", "vmc",
    "BdyMPS", "CompBdy", "BoundaryContractResult", "contract_boundary",
    "contract_flat", "build_bra_ket", "normalize", "peps_normalize", "boundary_norm", "infidelity",
    "peps_norm", "peps_infidelity", "peps_fidelity", "GlobalOptimizer", "FIT",
    "tns_align", "measure_obs", "build_pepo_from_gates", "build_mpo_from_gates",
    "pauli", "x", "y", "z", "s", "sdg", "t", "tdg", "h", "hadamard",
    "cnot", "cx", "cy", "cz", "swap", "iswap", "phase", "u1", "u2",
    "cphase", "crx", "cry", "crz", "cu1", "cu2", "cu3", "rx", "ry", "rz",
    "rxx", "ryy", "rzz", "u3", "su4", "fsim", "fsimg", "haar_random_state", "ps_to_peps", "ps_to_3dpeps", "expec_mpo",
    "id_to_mpo", "id_to_pepo", "ps_to_pepo", "ps_to_mpo", "make_numpy_array_caster", "to_float", "SweepOptimizer",
    "FDSolver", "MpsEnergyOptimizer", "MpsOptimizer", "MpoOptimizer", "PepsEnergyOptimizer", "PepsOptimizer", "SimpleUpdateGen", "SymDMRG2", "PEPSSampleResult",
    "PepsBpSampler", "MpsSampler", "MpsBatchSampleResult", "MpsSampleResult", "VecSampler", "gate", "gauge_all", "gauge_all_simple", "one_norm_bp", "tn_fidelity", "tn_norm",
    "MpsStabOptimizer", "STNState", "NoisyShotResult", "PauliErrorModel", "PauliFault",
    "StimCircuitPlan", "StimHerald", "StimNoiseSample", "StimShotResult",
    "TrajectoryChannel", "TrajectoryEvent", "TrajectoryOutcome", "TrajectoryRecord", "TrajectorySample", "TrajectoryShotResult",
    "compile_stim_circuit", "run_noisy_shots", "run_stim_shots", "run_trajectory_shots",
    "sample_noisy_gate_stream", "sample_noisy_gate_streams", "sample_stim_circuit", "sample_stim_circuits", "sample_trajectory_stream",
    "SymGateStream", "SymHamiltonian", "SymMPS", "SymPEPS",
    "default_physical_sectors", "draw_symmray_blocks", "draw_symmray_mps", "draw_symmray_mpo", "draw_symmray_peps",
    "fermi_hubbard_u1u1_gate_stream", "fermi_hubbard_u1u1_hopping_gate_stream",
    "fermi_hubbard_u1u1_interaction_gate_stream", "fermi_hubbard_u1u1_light_pulse_gate_stream",
    "fermi_hubbard_u1u1_jw_gate_stream", "fermi_hubbard_u1u1_jw_hopping_gate_stream",
    "fermi_hubbard_u1u1_jw_interaction_gate_stream",
    "sector_index_map",
    "site_charge_alternating", "site_charge_from_map",
    "site_charge_from_occupations", "site_charge_uniform",
    "symmray_block_summary", "symmray_mps_summary", "symmray_mpo_summary", "symmray_peps_summary", "symm_operator_from_dense",
    "reg_rel_svd_torch", "reg_real_svd_torch", "reg_complex_svd_torch",
    "reg_real_qr_torch", "reg_complex_qr_torch",
    "reg_rel_svd_jax", "reg_real_svd_jax", "reg_complex_svd_jax",
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


@pytest.mark.parametrize("name", _EXPECTED_IN_ALL)
def test_symbol_exported(name):
    """Public symbol should be in __all__."""
    assert name in pepsy.__all__


@pytest.mark.parametrize("name", _EXPECTED_NOT_IN_ALL)
def test_internal_symbol_not_exported(name):
    """Internal symbol should not leak into __all__."""
    assert name not in pepsy.__all__


_CALLABLE_EXPORTS = [
    "contract_boundary", "contract_flat", "build_bra_ket", "normalize", "peps_normalize",
    "boundary_norm", "peps_norm", "infidelity", "peps_infidelity", "peps_fidelity",
    "to_float", "gauge_all", "gauge_all_simple", "one_norm_bp",
    "GlobalOptimizer", "FIT", "tns_align", "measure_obs",
    "build_pepo_from_gates", "build_mpo_from_gates", "pauli",
    "x", "y", "z", "s", "sdg", "t", "tdg", "h", "hadamard",
    "cnot", "cx", "cy", "cz", "swap", "iswap", "phase", "u1", "u2",
    "cphase", "crx", "cry", "crz", "cu1", "cu2", "cu3", "rx", "ry", "rz",
    "rxx", "ryy", "rzz", "u3", "su4", "fsim", "fsimg", "haar_random_state", "ps_to_peps", "ps_to_3dpeps", "expec_mpo",
    "id_to_mpo", "id_to_pepo", "ps_to_pepo", "ps_to_mpo", "SweepOptimizer",
    "FDSolver", "MpsEnergyOptimizer", "MpsOptimizer", "MpoOptimizer", "PepsEnergyOptimizer", "PepsOptimizer", "SimpleUpdateGen", "SymDMRG2", "PEPSSampleResult", "PepsBpSampler", "compile_stim_circuit", "run_noisy_shots", "run_stim_shots", "run_trajectory_shots", "sample_noisy_gate_stream", "sample_noisy_gate_streams", "sample_stim_circuit", "sample_stim_circuits", "sample_trajectory_stream",
    "tn_fidelity", "tn_norm", "SymGateStream", "SymHamiltonian", "SymMPS", "SymPEPS",
    "default_physical_sectors", "draw_symmray_blocks", "draw_symmray_mps", "draw_symmray_mpo", "draw_symmray_peps",
    "fermi_hubbard_u1u1_gate_stream", "fermi_hubbard_u1u1_hopping_gate_stream",
    "fermi_hubbard_u1u1_interaction_gate_stream", "fermi_hubbard_u1u1_light_pulse_gate_stream",
    "fermi_hubbard_u1u1_jw_gate_stream", "fermi_hubbard_u1u1_jw_hopping_gate_stream",
    "fermi_hubbard_u1u1_jw_interaction_gate_stream",
    "sector_index_map",
    "site_charge_alternating", "site_charge_from_map",
    "site_charge_from_occupations", "site_charge_uniform",
    "symmray_block_summary", "symmray_mps_summary", "symmray_mpo_summary", "symmray_peps_summary", "symm_operator_from_dense",
    "reg_rel_svd_torch", "reg_real_svd_torch", "reg_complex_svd_torch",
    "reg_real_qr_torch", "reg_complex_qr_torch",
    "reg_rel_svd_jax", "reg_real_svd_jax", "reg_complex_svd_jax",
]

_BLOCKED_NAMES = _EXPECTED_NOT_IN_ALL

_MODULE_EXPORTS = [
    "backends", "boundary", "fitting", "operators", "optimizers",
    "sampling", "solvers", "tensors", "vmc",
]


def test_all_exports_are_unique():
    """Public export list should not contain duplicate names."""
    assert len(pepsy.__all__) == len(set(pepsy.__all__))


@pytest.mark.parametrize("name", pepsy.__all__)
def test_all_exports_resolve(name):
    """Every advertised public export should resolve."""
    assert getattr(pepsy, name) is not None


@pytest.mark.parametrize("name", _CALLABLE_EXPORTS)
def test_lazy_callable_resolves(name):
    """Lazy callable export should resolve to a callable."""
    assert callable(getattr(pepsy, name))


@pytest.mark.parametrize("name", _BLOCKED_NAMES)
def test_blocked_name_raises(name):
    """Internal name should raise AttributeError."""
    with pytest.raises(AttributeError):
        getattr(pepsy, name)


@pytest.mark.parametrize("name", _MODULE_EXPORTS)
def test_module_export_resolves(name):
    """Submodule export should resolve to a non-None value."""
    assert getattr(pepsy, name) is not None


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
    assert pepsy.reg_real_qr_torch is pepsy.tensors.reg_real_qr_torch
    assert pepsy.reg_complex_qr_torch is pepsy.tensors.reg_complex_qr_torch
    assert pepsy.reg_rel_svd_jax is pepsy.tensors.reg_rel_svd_jax
    assert pepsy.reg_real_svd_jax is pepsy.tensors.reg_real_svd_jax
    assert pepsy.reg_complex_svd_jax is pepsy.tensors.reg_complex_svd_jax
    if has_torch:
        import torch

        assert callable(pepsy.tensors.core.reg_rel_svd_torch)
        assert callable(pepsy.tensors.reg_rel_svd_torch)
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

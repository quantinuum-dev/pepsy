"""Basic public API smoke tests for the pepsy package."""

import importlib.util

import pepsy
import pytest


def test_package_version_available():
    """Package exposes a non-empty version string."""
    assert isinstance(pepsy.__version__, str)
    assert pepsy.__version__


_EXPECTED_IN_ALL = [
    "BdyMPS", "CompBdy", "BoundaryContractResult", "contract_boundary",
    "build_bra_ket", "normalize", "infidelity", "GlobalOptimizer", "FIT",
    "tns_align", "measure_obs", "build_pepo_from_gates", "build_mpo_from_gates",
    "pauli", "x", "y", "z", "s", "sdg", "t", "tdg", "h", "hadamard",
    "cnot", "cx", "cy", "cz", "swap", "iswap", "phase", "u1", "u2",
    "cphase", "crx", "cry", "crz", "cu1", "cu2", "cu3", "rx", "ry", "rz",
    "rxx", "ryy", "rzz", "u3", "su4", "fsim", "fsimg", "ps_to_peps", "expec_mpo",
    "id_to_mpo", "id_to_pepo", "ps_to_pepo", "ps_to_mpo", "make_numpy_array_caster", "SweepOptimizer",
    "FDSolver", "MpsOptimizer", "MpoOptimizer", "PEPSSampleResult",
    "PepsBpSampler", "optimize_global", "optimize_sweep",
    "optimize_mps", "gate", "gradient_solver", "ft_solver", "sampler",
]

_EXPECTED_NOT_IN_ALL = [
    "norm_peps", "normalize_peps", "loss_peps", "PEPSGlobalOptimizer",
    "tn_norm", "tn_fidelity", "gen_long_range_swap_path",
    "gen_long_range_swap_path_1d", "gen_long_range_swap_path_2d",
    "gen_long_range_swap_path_3d", "gate_tn_1d", "gate_tn_2d", "gate_tn_3d",
    "gates_tn_1d", "gates_tn_2d", "gates_tn_3d", "apply_2d_gate",
    "apply_2d_gates", "apply_2dtn_", "gate_2d", "gate_to_pepo", "gate_1d",
    "canonize_mps", "apply_gates_", "expec_TN_1D", "peps_I",
    "reg_complex_svd_torch", "reg_complex_svd_jax",
    "MPSOptimizer", "MPOOptimizer",
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
    "contract_boundary", "build_bra_ket", "normalize", "infidelity",
    "GlobalOptimizer", "FIT", "tns_align", "measure_obs",
    "build_pepo_from_gates", "build_mpo_from_gates", "pauli",
    "x", "y", "z", "s", "sdg", "t", "tdg", "h", "hadamard",
    "cnot", "cx", "cy", "cz", "swap", "iswap", "phase", "u1", "u2",
    "cphase", "crx", "cry", "crz", "cu1", "cu2", "cu3", "rx", "ry", "rz",
    "rxx", "ryy", "rzz", "u3", "su4", "fsim", "fsimg", "ps_to_peps", "expec_mpo",
    "id_to_mpo", "id_to_pepo", "ps_to_pepo", "ps_to_mpo", "SweepOptimizer",
    "FDSolver", "MpsOptimizer", "MpoOptimizer", "PEPSSampleResult", "PepsBpSampler",
]

_BLOCKED_NAMES = [
    "norm_peps", "normalize_peps", "loss_peps", "PEPSGlobalOptimizer",
    "gen_long_range_swap_path", "tn_norm", "tn_fidelity",
    "gen_long_range_swap_path_1d", "gen_long_range_swap_path_2d",
    "gen_long_range_swap_path_3d", "gate_tn_1d", "gate_tn_2d", "gate_tn_3d",
    "gates_tn_1d", "gates_tn_2d", "gates_tn_3d", "apply_2d_gate",
    "apply_2d_gates", "apply_2dtn_", "gate_2d", "gate_to_pepo", "gate_1d",
    "canonize_mps", "apply_gates_", "expec_TN_1D", "peps_I",
    "reg_complex_svd_torch", "reg_complex_svd_jax",
    "MPSOptimizer", "MPOOptimizer",
]

_MODULE_EXPORTS = ["optimize_global", "optimize_sweep", "optimize_mps", "gate", "gradient_solver", "ft_solver", "sampler"]


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
    """Linalg registrations are exposed under pepsy.core only."""
    has_torch = importlib.util.find_spec("torch") is not None
    has_jax = importlib.util.find_spec("jax") is not None
    if has_torch:
        assert callable(pepsy.core.reg_complex_svd_torch)
    if has_jax:
        assert callable(pepsy.core.reg_complex_svd_jax)

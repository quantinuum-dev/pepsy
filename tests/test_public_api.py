"""Basic public API smoke tests for the pepsy package."""

import pepsy
import pytest


def test_package_version_available():
    """Package exposes a non-empty version string."""
    assert isinstance(pepsy.__version__, str)
    assert pepsy.__version__


def test_core_symbols_exported():
    """Top-level package exports expected boundary API symbols."""
    assert "BdyMPS" in pepsy.__all__
    assert "CompBdy" in pepsy.__all__
    assert "BoundaryContractResult" in pepsy.__all__
    assert "ContractBoundary" in pepsy.__all__
    assert "prepare_boundary_inputs" in pepsy.__all__
    assert "normalize" in pepsy.__all__
    assert "infidelity" in pepsy.__all__
    assert "norm_peps" not in pepsy.__all__
    assert "normalize_peps" not in pepsy.__all__
    assert "loss_peps" not in pepsy.__all__
    assert "GlobalOptimizer" in pepsy.__all__
    assert "FIT" in pepsy.__all__
    assert "PEPSGlobalOptimizer" not in pepsy.__all__
    assert "tn_applied" in pepsy.__all__
    assert "gen_long_range_swap_path" in pepsy.__all__
    assert "apply_2dtn_" in pepsy.__all__
    assert "apply_gates" in pepsy.__all__
    assert "gate_1d" in pepsy.__all__
    assert "canonize_mps" in pepsy.__all__
    assert "apply_gates_" not in pepsy.__all__
    assert "product_state_peps" in pepsy.__all__
    assert "peps_I" not in pepsy.__all__
    assert "reg_complex_svd_torch" in pepsy.__all__
    assert "reg_complex_svd_jax" in pepsy.__all__
    assert "make_numpy_array_caster" in pepsy.__all__
    assert "PEPSSweepOptimizer" in pepsy.__all__
    assert "SweepResult" in pepsy.__all__
    assert "plot_sweep_diagnostics" in pepsy.__all__
    assert "plot_inner_loss" in pepsy.__all__
    assert "optimize_global" in pepsy.__all__
    assert "optimize_sweep" in pepsy.__all__
    assert "gate" in pepsy.__all__
    assert "gradient_solver" in pepsy.__all__
    assert "debug" in pepsy.__all__


def test_lazy_exports_resolve():
    """Selected lazy exports should resolve to callables/modules."""
    assert callable(pepsy.ContractBoundary)
    assert callable(pepsy.normalize)
    assert callable(pepsy.infidelity)
    with pytest.raises(AttributeError):
        _ = pepsy.norm_peps
    with pytest.raises(AttributeError):
        _ = pepsy.normalize_peps
    with pytest.raises(AttributeError):
        _ = pepsy.loss_peps
    assert callable(pepsy.GlobalOptimizer)
    assert callable(pepsy.FIT)
    with pytest.raises(AttributeError):
        _ = pepsy.PEPSGlobalOptimizer
    assert callable(pepsy.tn_applied)
    assert callable(pepsy.gen_long_range_swap_path)
    assert callable(pepsy.apply_2dtn_)
    assert callable(pepsy.apply_gates)
    assert callable(pepsy.gate_1d)
    assert callable(pepsy.canonize_mps)
    with pytest.raises(AttributeError):
        _ = pepsy.apply_gates_
    assert callable(pepsy.product_state_peps)
    with pytest.raises(AttributeError):
        _ = pepsy.peps_I
    assert callable(pepsy.reg_complex_svd_torch)
    assert callable(pepsy.reg_complex_svd_jax)
    assert callable(pepsy.plot_sweep_diagnostics)
    assert callable(pepsy.plot_inner_loss)
    assert pepsy.optimize_global is not None
    assert pepsy.optimize_sweep is not None
    assert pepsy.gate is not None
    assert pepsy.gradient_solver is not None

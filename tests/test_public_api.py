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
    assert "contract_boundary" in pepsy.__all__
    assert "build_bra_ket" in pepsy.__all__
    assert "normalize" in pepsy.__all__
    assert "infidelity" in pepsy.__all__
    assert "norm_peps" not in pepsy.__all__
    assert "normalize_peps" not in pepsy.__all__
    assert "loss_peps" not in pepsy.__all__
    assert "GlobalOptimizer" in pepsy.__all__
    assert "FIT" in pepsy.__all__
    assert "PEPSGlobalOptimizer" not in pepsy.__all__
    assert "tns_align" in pepsy.__all__
    assert "tn_norm" not in pepsy.__all__
    assert "tn_fidelity" not in pepsy.__all__
    assert "gen_long_range_swap_path" in pepsy.__all__
    assert "apply_gate_2d" in pepsy.__all__
    assert "apply_gates_2d" in pepsy.__all__
    assert "apply_2d_gate" not in pepsy.__all__
    assert "apply_2d_gates" not in pepsy.__all__
    assert "apply_2dtn_" not in pepsy.__all__
    assert "gate_2d" not in pepsy.__all__
    assert "gates_to_pepo" in pepsy.__all__
    assert "gate_to_pepo" not in pepsy.__all__
    assert "apply_gate_1d" in pepsy.__all__
    assert "gate_1d" not in pepsy.__all__
    assert "pauli" in pepsy.__all__
    assert "canonize_mps" not in pepsy.__all__
    assert "x" in pepsy.__all__
    assert "y" in pepsy.__all__
    assert "z" in pepsy.__all__
    assert "s" in pepsy.__all__
    assert "sdg" in pepsy.__all__
    assert "t" in pepsy.__all__
    assert "tdg" in pepsy.__all__
    assert "h" in pepsy.__all__
    assert "hadamard" in pepsy.__all__
    assert "cnot" in pepsy.__all__
    assert "cx" in pepsy.__all__
    assert "cy" in pepsy.__all__
    assert "cz" in pepsy.__all__
    assert "swap" in pepsy.__all__
    assert "iswap" in pepsy.__all__
    assert "phase" in pepsy.__all__
    assert "u1" in pepsy.__all__
    assert "u2" in pepsy.__all__
    assert "cphase" in pepsy.__all__
    assert "crx" in pepsy.__all__
    assert "cry" in pepsy.__all__
    assert "crz" in pepsy.__all__
    assert "cu1" in pepsy.__all__
    assert "cu2" in pepsy.__all__
    assert "cu3" in pepsy.__all__
    assert "rx" in pepsy.__all__
    assert "ry" in pepsy.__all__
    assert "rz" in pepsy.__all__
    assert "rxx" in pepsy.__all__
    assert "ryy" in pepsy.__all__
    assert "rzz" in pepsy.__all__
    assert "u3" in pepsy.__all__
    assert "su4" in pepsy.__all__
    assert "apply_gates_" not in pepsy.__all__
    assert "ps_to_peps" in pepsy.__all__
    assert "peps_I" not in pepsy.__all__
    assert "reg_complex_svd_torch" in pepsy.__all__
    assert "reg_complex_svd_jax" in pepsy.__all__
    assert "make_numpy_array_caster" in pepsy.__all__
    assert "SweepOptimizer" in pepsy.__all__
    assert "MpsOptimizer" in pepsy.__all__
    assert "optimize_global" in pepsy.__all__
    assert "optimize_sweep" in pepsy.__all__
    assert "gate" in pepsy.__all__
    assert "gradient_solver" in pepsy.__all__


def test_lazy_exports_resolve():
    """Selected lazy exports should resolve to callables/modules."""
    assert callable(pepsy.contract_boundary)
    assert callable(pepsy.build_bra_ket)
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
    assert callable(pepsy.tns_align)
    with pytest.raises(AttributeError):
        _ = pepsy.tn_norm
    with pytest.raises(AttributeError):
        _ = pepsy.tn_fidelity
    assert callable(pepsy.gen_long_range_swap_path)
    assert callable(pepsy.apply_gate_2d)
    assert callable(pepsy.apply_gates_2d)
    with pytest.raises(AttributeError):
        _ = pepsy.apply_2d_gate
    with pytest.raises(AttributeError):
        _ = pepsy.apply_2d_gates
    with pytest.raises(AttributeError):
        _ = pepsy.apply_2dtn_
    with pytest.raises(AttributeError):
        _ = pepsy.gate_2d
    assert callable(pepsy.gates_to_pepo)
    with pytest.raises(AttributeError):
        _ = pepsy.gate_to_pepo
    assert callable(pepsy.apply_gate_1d)
    with pytest.raises(AttributeError):
        _ = pepsy.gate_1d
    assert callable(pepsy.pauli)
    with pytest.raises(AttributeError):
        _ = pepsy.canonize_mps
    assert callable(pepsy.x)
    assert callable(pepsy.y)
    assert callable(pepsy.z)
    assert callable(pepsy.s)
    assert callable(pepsy.sdg)
    assert callable(pepsy.t)
    assert callable(pepsy.tdg)
    assert callable(pepsy.h)
    assert callable(pepsy.hadamard)
    assert callable(pepsy.cnot)
    assert callable(pepsy.cx)
    assert callable(pepsy.cy)
    assert callable(pepsy.cz)
    assert callable(pepsy.swap)
    assert callable(pepsy.iswap)
    assert callable(pepsy.phase)
    assert callable(pepsy.u1)
    assert callable(pepsy.u2)
    assert callable(pepsy.cphase)
    assert callable(pepsy.crx)
    assert callable(pepsy.cry)
    assert callable(pepsy.crz)
    assert callable(pepsy.cu1)
    assert callable(pepsy.cu2)
    assert callable(pepsy.cu3)
    assert callable(pepsy.rx)
    assert callable(pepsy.ry)
    assert callable(pepsy.rz)
    assert callable(pepsy.rxx)
    assert callable(pepsy.ryy)
    assert callable(pepsy.rzz)
    assert callable(pepsy.u3)
    assert callable(pepsy.su4)
    with pytest.raises(AttributeError):
        _ = pepsy.apply_gates_
    assert callable(pepsy.ps_to_peps)
    with pytest.raises(AttributeError):
        _ = pepsy.peps_I
    assert callable(pepsy.reg_complex_svd_torch)
    assert callable(pepsy.reg_complex_svd_jax)
    assert callable(pepsy.SweepOptimizer)
    assert callable(pepsy.MpsOptimizer)
    assert pepsy.optimize_global is not None
    assert pepsy.optimize_sweep is not None
    assert pepsy.gate is not None
    assert pepsy.gradient_solver is not None

"""Tests for :mod:`pepsy.optimize_mps`."""

from numbers import Real

import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy as py


def test_mps_optimizer_accepts_svd_mode():
    """SVD mode should be accepted by ``MpsOptimizer`` mode validation."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0, gates=[], chi=8, mode="svd")
    assert opt.mode == "svd"


def test_mps_optimizer_svd_smoke():
    """SVD mode should apply mixed 1q/2q gates without errors."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        ((1,), qu.hadamard()),
        ((0, 3), qu.CNOT()),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12)

    assert opt.p.L == 4
    assert opt.p.max_bond() <= 8


@pytest.mark.parametrize("mode", ["mpo", "svd", "swap"])
def test_mps_optimizer_compression_modes_sample_norm_proxy(mode):
    """Compression modes should append norm-proxy samples to fidelity trace."""
    p0 = qtn.MPS_computational_state("00000", dtype="complex128")
    gates = [
        ((1,), qu.hadamard()),
        ((0, 4), qu.CNOT()),
        ((2,), qu.phase_gate(0.2)),
        ((1, 3), qu.CNOT()),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode=mode)
    if mode == "swap" and not hasattr(opt.p, "gate_with_auto_swap_"):
        pytest.skip("swap mode requires gate_with_auto_swap_ in this quimb version.")

    before = len(opt.get_fidelities())
    opt.run(progbar=False, cutoff=1e-12, fidelity_samples=2)
    trace = opt.get_fidelities()

    assert len(trace) >= before + 1
    assert isinstance(trace[-1], Real)


def test_mps_optimizer_rejects_negative_fidelity_samples():
    """Negative fidelity_samples should fail clearly."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [((0, 3), qu.CNOT())]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="mpo")
    with pytest.raises(ValueError, match="fidelity_samples must be >= 0"):
        opt.run(progbar=False, cutoff=1e-12, fidelity_samples=-1)

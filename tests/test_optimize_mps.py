"""Tests for :mod:`pepsy.optimize_mps`."""

from numbers import Real

import numpy as np
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
    G = [qu.hadamard(), qu.CNOT()]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12)

    assert opt.p.L == 4
    assert opt.p.max_bond() <= 8


def test_mps_optimizer_accepts_bundled_gate_stream():
    """Construction should accept ``[(gate, where), ...]`` with ``where=None``."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [(qu.hadamard(), (1,)), (qu.CNOT(), (0, 3))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12)

    assert opt.p.L == 4
    assert opt.where == [(1,), (0, 3)]


def test_mps_optimizer_set_and_add_gates_accept_bundled_gate_stream():
    """set_gates/add_gates should accept bundled gate-stream entries."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0.copy(), chi=8, mode="svd")

    opt.set_gates([(qu.hadamard(), (1,))])
    opt.add_gates([(qu.CNOT(), (0, 3))])

    assert len(opt.G) == 2
    assert opt.where == [(1,), (0, 3)]


def test_mps_optimizer_default_inplace_false_keeps_input_unchanged():
    """Default construction should work on a copy and keep input state intact."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    p0_ref = p0.copy()
    gates = [(qu.hadamard(), (1,))]

    opt = py.MpsOptimizer(p0, gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12)

    assert opt.p is not p0
    assert np.allclose(p0.to_dense(), p0_ref.to_dense())
    assert not np.allclose(out.to_dense(), p0_ref.to_dense())


def test_mps_optimizer_inplace_true_updates_input_state():
    """inplace=True should optimize the original input state object."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    p0_ref = p0.copy()
    gates = [(qu.hadamard(), (1,))]

    opt = py.MpsOptimizer(p0, gates=gates, chi=8, mode="svd", inplace=True)
    out = opt.run(progbar=False, cutoff=1e-12)

    assert opt.p is p0
    assert out is p0
    assert not np.allclose(p0.to_dense(), p0_ref.to_dense())


def test_mps_optimizer_rejects_noncanonical_bundled_gate_aliases():
    """Bundled gate input should require the canonical list/tuple shapes."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")

    opt = py.MpsOptimizer(p0.copy(), gates=((qu.hadamard(), (1,)),), chi=8, mode="svd")
    assert opt.where == [(1,)]

    with pytest.raises(ValueError, match="exact shape"):
        py.MpsOptimizer(p0.copy(), gates=[(qu.hadamard(), (1,)), qu.CNOT()], chi=8, mode="svd")


def test_mps_optimizer_run_returns_state_for_empty_queue():
    """run() should return the managed MPS even when there are no gates."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0.copy(), gates=[], chi=8, mode="svd")

    out = opt.run(progbar=False, cutoff=1e-12)

    assert out is opt.p


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "svd", "exact"])
def test_mps_optimizer_run_returns_state_after_updates(mode):
    """run() should return the updated MPS for every execution mode."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    G = [qu.hadamard(), qu.CNOT()]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode=mode)

    out = opt.run(progbar=False, cutoff=1e-12, n_iter=2, fidelity_samples=1)

    assert out is opt.p


@pytest.mark.parametrize("mode", ["mpo", "svd", "swap"])
def test_mps_optimizer_compression_modes_sample_norm_proxy(mode):
    """Compression modes should append norm-proxy samples to fidelity trace."""
    p0 = qtn.MPS_computational_state("00000", dtype="complex128")
    G = [qu.hadamard(), qu.CNOT(), qu.phase_gate(0.2), qu.CNOT()]
    where = [(1,), (0, 4), (2,), (1, 3)]
    gates = list(zip(G, where))

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
    G = [qu.CNOT()]
    where = [(0, 3)]
    gates = list(zip(G, where))

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="mpo")
    with pytest.raises(ValueError, match="fidelity_samples must be >= 0"):
        opt.run(progbar=False, cutoff=1e-12, fidelity_samples=-1)

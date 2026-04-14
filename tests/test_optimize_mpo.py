"""Tests for :mod:`pepsy.optimize_mpo`."""

import quimb as qu
import quimb.tensor as qtn

import pepsy as py


def test_mpo_optimizer_exported():
    """Public API should expose MpoOptimizer and optimize_mpo module."""
    assert "MpoOptimizer" in py.__all__
    assert "optimize_mpo" in py.__all__
    assert py.MpoOptimizer is not None
    assert py.optimize_mpo is not None


def test_mpo_optimizer_accepts_svd_mode():
    """SVD mode should be accepted by ``MpoOptimizer`` mode validation."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=8, mode="svd")
    assert opt.mode == "svd"


def test_mpo_prepare_gate_pair_uses_matrix_transpose_for_2q_quimb_gate():
    """For rank-2 two-site gates, use direct matrix transpose (no reshape path)."""
    gate = qu.CNOT()
    g_k, g_b = py.MpoOptimizer._prepare_gate_pair(gate, n_sites=2)

    # Simplified branch: same rank and direct transpose.
    assert g_k.shape == gate.shape
    assert (g_k == gate.T).all()
    assert (g_b == g_k.conj()).all()

    # Equivalent to the previous reshape -> (2,3,0,1) mapping.
    old_style = gate.reshape(2, 2, 2, 2).transpose(2, 3, 0, 1).reshape(4, 4)
    assert (g_k == old_style).all()


def test_mpo_optimizer_dmrg_smoke():
    """MpoOptimizer should apply mixed 1q/2q gates without error."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [
        ((1,), qu.hadamard()),
        ((0, 3), qu.CNOT()),
    ]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="dmrg")
    out = opt.run(n_iter=2, progbar=False, cutoff=1e-12)

    assert out.L == 4
    assert out.max_bond() >= 1


def test_mpo_optimizer_canonicalization_state_initialized():
    """Construction should initialize canonicalization metadata."""
    mpo0 = qtn.MPO_identity(5, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=6, mode="dmrg")

    assert isinstance(opt.info_c, dict)
    assert "cur_orthog" in opt.info_c
    cur = opt.info_c["cur_orthog"]
    assert isinstance(cur, tuple)
    assert len(cur) == 2


def test_mpo_optimizer_prepare_dmrg_state_expands_to_chi():
    """DMRG preparation should expand low-bond MPOs up to ``chi``."""
    mpo0 = qtn.MPO_identity(6, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=7, mode="dmrg")

    # Force a low-bond starting point then check expansion.
    opt.mpo = qtn.MPO_identity(6, dtype="complex128")
    opt.p = opt.mpo
    assert opt.p.max_bond() == 1

    opt._prepare_dmrg_state()
    assert opt.p.max_bond() >= 7


def test_mpo_optimizer_tracks_fidelity_proxy_for_two_site_fit():
    """Two-site DMRG updates should append to the local-fidelity proxy trace."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [
        ((0, 1), qu.CNOT()),
        ((2,), qu.hadamard()),
        ((2, 3), qu.CNOT()),
    ]

    chi = 6
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=chi, mode="dmrg")
    out = opt.run(n_iter=2, progbar=False, cutoff=1e-12)

    assert len(opt.get_fidelities()) >= 3
    assert out.max_bond() <= chi


def test_mpo_optimizer_rejects_unknown_mode():
    """Unknown modes should fail with a clear supported-modes message."""
    mpo0 = qtn.MPO_identity(3, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=4, mode="dmrg")

    try:
        opt.set_mode("mpo")
    except ValueError as exc:
        assert "Supported modes:" in str(exc)
        assert "dmrg" in str(exc)
        assert "svd" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported mode")


def test_mpo_optimizer_svd_smoke():
    """SVD mode should apply mixed 1q/2q gates without errors."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [
        ((1,), qu.hadamard()),
        ((0, 3), qu.CNOT()),
    ]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=2)

    assert out.L == 4
    assert out.max_bond() <= 8


def test_mpo_optimizer_svd_rejects_negative_fidelity_samples():
    """Negative fidelity_samples should fail clearly in SVD mode."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [((0, 3), qu.CNOT())]
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")

    try:
        opt.run(progbar=False, cutoff=1e-12, fidelity_samples=-1)
    except ValueError as exc:
        assert "fidelity_samples must be >= 0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative fidelity_samples")


def test_mpo_prepare_gate_pair_uses_explicit_bra_when_provided():
    """When B is provided, use B on bra indices instead of conj(G_k)."""
    g = qu.CNOT()
    b = qu.swap()
    g_k, g_b = py.MpoOptimizer._prepare_gate_pair(g, n_sites=2, bra_gate=b)

    assert (g_k == g.T).all()
    assert (g_b == b.T).all()


def test_mpo_optimizer_accepts_three_tuple_gate_spec():
    """Gate specs may be (where, G, B) with explicit bra-side operator."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [
        ((1,), qu.hadamard(), qu.hadamard()),
        ((0, 3), qu.CNOT(), qu.swap()),
    ]
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=1)
    assert out.L == 4

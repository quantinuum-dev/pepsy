"""Tests for :mod:`pepsy.optimize_mpo`."""

import numpy as np
import pytest
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
    """For rank-2 two-site gates, use direct matrix transpose on ket only."""
    gate = qu.CNOT()
    g_k, g_b = py.MpoOptimizer._prepare_gate_pair(gate, n_sites=2)

    # Simplified branch: same rank and direct transpose.
    assert g_k.shape == gate.shape
    assert (g_k == gate.T).all()
    assert g_b is None

    # Equivalent to the previous reshape -> (2,3,0,1) mapping.
    old_style = gate.reshape(2, 2, 2, 2).transpose(2, 3, 0, 1).reshape(4, 4)
    assert (g_k == old_style).all()


def test_mpo_optimizer_single_gate_defaults_to_unitary_conjugation():
    """A bare ``G`` entry should apply ``G`` on ket and ``G†`` on bra."""
    mpo0 = qtn.MPO_identity(2, dtype="complex128")
    gate = qu.phase_gate(0.37)
    gates = [(gate, (0,))]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=4, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert np.allclose(out.to_dense(), mpo0.to_dense())


def test_mpo_optimizer_explicit_ket_only_pair_changes_identity():
    """Explicit ``(G, None)`` should keep ket-only update semantics."""
    mpo0 = qtn.MPO_identity(2, dtype="complex128")
    gate = qu.phase_gate(0.37)
    gates = [((gate, None), (0,))]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=4, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert not np.allclose(out.to_dense(), mpo0.to_dense())


def test_mpo_optimizer_accepts_singleton_ket_only_shorthand():
    """A ``(G,)`` entry should be treated as explicit ket-only shorthand."""
    mpo0 = qtn.MPO_identity(2, dtype="complex128")
    gate = qu.phase_gate(0.37)
    gates = [((gate,), (0,))]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=4, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert not np.allclose(out.to_dense(), mpo0.to_dense())


def test_mpo_optimizer_dmrg_smoke():
    """MpoOptimizer should apply mixed 1q/2q gates without error."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    G = [qu.hadamard(), qu.CNOT()]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="dmrg")
    out = opt.run(n_iter=2, progbar=False, cutoff=1e-12)

    assert out.L == 4
    assert out.max_bond() >= 1


def test_mpo_optimizer_accepts_bundled_gate_stream():
    """Construction should accept bundled ``[(gate, where), ...]`` entries."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [((qu.hadamard(), None), (1,)), ((qu.CNOT(), None), (0, 3))]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=1)

    assert out.L == 4
    assert opt.where == [(1,), (0, 3)]


def test_mpo_optimizer_default_inplace_false_keeps_input_unchanged():
    """Default construction should work on a copy and keep input MPO intact."""
    mpo0 = qtn.MPO_identity(2, dtype="complex128")
    mpo0_ref = mpo0.copy()
    gate = qu.phase_gate(0.37)
    gates = [((gate, None), (0,))]

    opt = py.MpoOptimizer(mpo0, gates=gates, chi=4, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert opt.p is not mpo0
    assert np.allclose(mpo0.to_dense(), mpo0_ref.to_dense())
    assert not np.allclose(out.to_dense(), mpo0_ref.to_dense())


def test_mpo_optimizer_inplace_true_updates_input_mpo():
    """inplace=True should optimize the original input MPO object."""
    mpo0 = qtn.MPO_identity(2, dtype="complex128")
    mpo0_ref = mpo0.copy()
    gate = qu.phase_gate(0.37)
    gates = [((gate, None), (0,))]

    opt = py.MpoOptimizer(mpo0, gates=gates, chi=4, mode="svd", inplace=True)
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert opt.p is mpo0
    assert out is mpo0
    assert not np.allclose(mpo0.to_dense(), mpo0_ref.to_dense())


def test_mpo_optimizer_rejects_noncanonical_bundled_gate_aliases():
    """Bundled gate input should require the canonical list/tuple shapes."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")

    opt = py.MpoOptimizer(
        mpo0.copy(),
        gates=(((qu.hadamard(), None), (1,)),),
        chi=8,
        mode="svd",
    )
    assert opt.where == [(1,)]

    with pytest.raises(ValueError, match="exact shape"):
        py.MpoOptimizer(
            mpo0.copy(),
            gates=[((qu.hadamard(), None), (1,)), (qu.CNOT(), None)],
            chi=8,
            mode="svd",
        )


def test_mpo_optimizer_canonicalization_state_initialized():
    """Construction should initialize canonicalization metadata."""
    mpo0 = qtn.MPO_identity(5, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=6, mode="dmrg")

    assert isinstance(opt.info_c, dict)
    assert "cur_orthog" in opt.info_c
    cur = opt.info_c["cur_orthog"]
    assert isinstance(cur, tuple)
    assert len(cur) == 2
    assert np.isclose(opt.norm_mpo, py.core.tn_norm(mpo0, contraction_opt=opt.contraction_opt))


def test_mpo_optimizer_prepare_dmrg_state_expands_to_chi():
    """DMRG preparation should expand low-bond MPOs up to ``chi``."""
    mpo0 = qtn.MPO_identity(6, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=7, mode="dmrg")

    # Force a low-bond starting point then check expansion.
    opt.p = qtn.MPO_identity(6, dtype="complex128")
    assert opt.p.max_bond() == 1

    opt._prepare_dmrg_state()
    assert opt.p.max_bond() >= 7


def test_mpo_optimizer_tracks_fidelity_proxy_for_two_site_fit():
    """Two-site DMRG updates should append to the local-fidelity proxy trace."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    G = [qu.CNOT(), qu.hadamard(), qu.CNOT()]
    where = [(0, 1), (2,), (2, 3)]
    gates = list(zip(G, where))

    chi = 6
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=chi, mode="dmrg")
    out = opt.run(n_iter=2, progbar=False, cutoff=1e-12)

    assert len(opt.get_fidelities()) >= 3
    assert out.max_bond() <= chi


def test_mpo_optimizer_dmrg_accepts_k_2q_batch():
    """DMRG MPO mode should accept batching sequential two-site gates."""
    mpo0 = qtn.MPO_identity(5, dtype="complex128")
    G = [qu.CNOT(), qu.hadamard(), qu.CNOT(), qu.CNOT()]
    where = [(0, 1), (2,), (2, 3), (3, 4)]
    gates = list(zip(G, where))

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="dmrg")
    out = opt.run(n_iter=2, progbar=False, cutoff=1e-12, k_2q_batch=2)

    assert out.L == 5
    assert out.max_bond() <= 8
    assert len(opt.get_fidelities()) >= 2


def test_mpo_optimizer_dmrg_rejects_invalid_k_2q_batch():
    """DMRG MPO batching count should fail clearly when invalid."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [(qu.CNOT(), (0, 1))]
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="dmrg")

    with pytest.raises(ValueError, match="k_2q_batch must be >= 1"):
        opt.run(n_iter=2, progbar=False, cutoff=1e-12, k_2q_batch=0)


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
    G = [qu.hadamard(), qu.CNOT()]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=2)

    assert out.L == 4
    assert out.max_bond() <= 8


def test_mpo_optimizer_normalized_norm_trace_stays_one_for_unitary_identity_evolution():
    """Two-sided unitary MPO evolution should preserve the normalized norm trace."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (1, 2)),
        (qu.phase_gate(0.37), (3,)),
    ]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=len(gates))

    assert np.allclose(out.to_dense(), mpo0.to_dense())
    assert all(np.isclose(val, 1.0, atol=1e-10) for val in opt.get_fidelities())


def test_mpo_optimizer_svd_rejects_negative_fidelity_samples():
    """Negative fidelity_samples should fail clearly in SVD mode."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    G = [qu.CNOT()]
    where = [(0, 3)]
    gates = list(zip(G, where))
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")

    try:
        opt.run(progbar=False, cutoff=1e-12, fidelity_samples=-1)
    except ValueError as exc:
        assert "fidelity_samples must be >= 0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative fidelity_samples")


def test_mpo_prepare_gate_pair_uses_explicit_bra_when_provided():
    """When B is provided, use B† on bra indices."""
    g = qu.CNOT()
    b = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    g_k, g_b = py.MpoOptimizer._prepare_gate_pair(g, n_sites=2, bra_gate=b)

    assert (g_k == g.T).all()
    assert (g_b == b.conj().T).all()


def test_mpo_optimizer_accepts_three_tuple_gate_spec():
    """Each ``G`` entry may be ``(G, B)`` with explicit bra-side operator B†."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    G = [(qu.hadamard(), qu.hadamard()), (qu.CNOT(), qu.swap())]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=1)
    assert out.L == 4


def test_mpo_optimizer_accepts_ket_only_gate_pair():
    """A ``(G, None)`` entry should apply ket side only."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    G = [(qu.hadamard(), None), (qu.CNOT(), None)]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=1)
    assert out.L == 4


def test_mpo_optimizer_accepts_bra_only_gate_pair():
    """A ``(None, B)`` entry should apply bra side only."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    G = [(None, qu.hadamard()), (None, qu.CNOT())]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=1)
    assert out.L == 4


def test_mpo_optimizer_rejects_empty_gate_pair():
    """A ``(None, None)`` gate entry should fail clearly."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    G = [(None, None)]
    where = [(1,)]
    gates = list(zip(G, where))
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")

    try:
        opt.run(progbar=False, cutoff=1e-12, fidelity_samples=1)
    except ValueError as exc:
        assert "at least one of G or B" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty gate spec")

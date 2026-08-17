"""Tests for :mod:`pepsy.optimizers.mpo.optimizer`."""

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy as py


def test_mpo_optimizer_exported():
    """Public API should expose MpoOptimizer and the optimizers namespace."""
    assert "MpoOptimizer" in py.__all__
    assert "optimizers" in py.__all__
    assert py.MpoOptimizer is not None
    assert py.optimizers.mpo is not None


def test_mpo_optimizer_accepts_svd_mode():
    """SVD mode should be accepted by ``MpoOptimizer`` mode validation."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=8, mode="svd")
    assert opt.mode == "svd"


def test_fit_two_site_preserves_mpo_view_and_dense_readout():
    """Direct FIT on an MPO must return a functional MPO, not an MPS view."""
    guess = qtn.MPO_rand(
        3, bond_dim=1, phys_dim=2, dtype="complex128", seed=212
    )
    target = qtn.MPO_rand(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=213
    )
    fit = py.FIT(target, p=guess, range_int=[0, 2])

    fit.run_gate(
        n_iter=2,
        block_size=2,
        sweep_sequence="RL",
        max_bond=2,
    )

    assert isinstance(fit.p, qtn.MatrixProductOperator)
    assert fit.p.upper_ind_id == guess.upper_ind_id
    assert fit.p.lower_ind_id == guess.lower_ind_id
    assert fit.p.to_dense().shape == (8, 8)
    assert fit.p.max_bond() <= 2


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


def test_mpo_optimizer_apply_gate_pair_separates_ket_and_bra_indices(monkeypatch):
    """Ket updates should use k indices and bra updates should use b indices."""
    calls = []

    def _fake_apply_gate(tn, gate, where, **kwargs):
        calls.append((where, kwargs.copy()))
        return tn

    monkeypatch.setattr("pepsy.optimizers.mpo.optimizer.apply_gate", _fake_apply_gate)

    mpo0 = qtn.MPO_identity(2, dtype="complex128")
    gate = qu.phase_gate(0.37)
    opt = py.MpoOptimizer(
        mpo0.copy(),
        gates=[],
        chi=4,
        mode="svd",
        ind_id_k="k{}",
        ind_id_b="b{}",
    )

    opt._apply_gate_pair(
        opt.p,
        gate,
        (1,),
        bra_gate=gate,
        cutoff=1.0e-12,
        contract=True,
    )

    assert len(calls) == 2
    assert [kwargs["ind_id"] for _, kwargs in calls] == ["k{}", "b{}"]
    assert all(where == (1,) for where, _ in calls)


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
    expected_m, expected_e = py.tensors.tn_norm(
        mpo0, contraction_opt=opt.contraction_opt, strip_exponent=True
    )
    got_m, got_e = opt.norm_mpo
    assert np.isclose(got_m, expected_m)
    assert np.isclose(got_e, expected_e)


def test_mpo_optimizer_current_orthog_normalizes_supported_shapes():
    """Cached orthogonality metadata should accept 1-site and 2-site forms."""
    mpo0 = qtn.MPO_identity(5, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=6, mode="dmrg")

    opt.info_c["cur_orthog"] = 2
    assert opt._current_orthog() == (2, 2)

    opt.info_c["cur_orthog"] = (3,)
    assert opt._current_orthog() == (3, 3)

    opt.info_c["cur_orthog"] = (4, 1)
    assert opt._current_orthog() == (1, 4)

    opt.info_c["cur_orthog"] = (1, 2, 3)
    with pytest.raises(ValueError, match="cur_orthog must be"):
        opt._current_orthog()


def test_mpo_optimizer_canonize_mpo_accepts_supported_where_shapes():
    """canonize_mpo should accept int, singleton, and pair site selectors."""
    mpo0 = qtn.MPO_identity(5, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=6, mode="dmrg")

    opt.canonize_mpo(opt.p, 2)
    assert opt.info_c["cur_orthog"] == (2, 2)

    opt.canonize_mpo(opt.p, (3,))
    assert opt.info_c["cur_orthog"] == (3, 3)

    opt.canonize_mpo(opt.p, (4, 1))
    assert opt.info_c["cur_orthog"] == (1, 4)

    with pytest.raises(ValueError, match="where must be"):
        opt.canonize_mpo(opt.p, (1, 2, 3))


def test_mpo_optimizer_prepare_dmrg_state_expands_to_chi():
    """DMRG preparation should expand low-bond MPOs up to ``chi``."""
    mpo0 = qtn.MPO_identity(6, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=7, mode="dmrg")

    # Force a low-bond starting point then check expansion.
    opt.p = qtn.MPO_identity(6, dtype="complex128")
    assert opt.p.max_bond() == 1

    opt._prepare_dmrg_state()
    assert opt.p.max_bond() >= 7


@pytest.mark.parametrize("fit_block_size", (2, 3))
def test_mpo_optimizer_dmrg_forwards_native_fit_controls(
    monkeypatch,
    fit_block_size,
):
    """MPO DMRG must pass block and SVD policy into the FIT kernel."""
    calls = []
    original_run_gate = py.FIT.run_gate

    def recording_run_gate(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_run_gate(self, *args, **kwargs)

    monkeypatch.setattr(py.FIT, "run_gate", recording_run_gate)

    opt = py.MpoOptimizer(
        qtn.MPO_identity(5, dtype="complex128"),
        gates=[((qu.CNOT(), None), (0, 4))],
        chi=2,
        mode="dmrg",
    )
    out = opt.run(
        n_iter=1,
        progbar=False,
        cutoff=2.0e-2,
        cutoff_mode="rel",
        fit_block_size=fit_block_size,
        fit_sweep_sequence="L",
        target_cutoff=0.0,
    )

    assert out.max_bond() <= 2
    assert len(calls) == 1
    assert calls[0]["block_size"] == fit_block_size
    assert calls[0]["sweep_sequence"] == "L"
    assert calls[0]["max_bond"] == 2
    assert calls[0]["cutoff"] == pytest.approx(2.0e-2)
    assert calls[0]["cutoff_mode"] == "rel"


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

    with pytest.raises(ValueError, match="Supported modes:"):
        opt.set_mode("invalid_mode")


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

    with pytest.raises(ValueError, match="fidelity_samples must be >= 0"):
        opt.run(progbar=False, cutoff=1e-12, fidelity_samples=-1)


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

    with pytest.raises(ValueError, match="at least one of G or B"):
        opt.run(progbar=False, cutoff=1e-12, fidelity_samples=1)


def test_mpo_optimizer_mpo_mode_smoke():
    """MPO mode should apply mixed 1q/2q gates via gate_nonlocal_opt without errors."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [
        (qu.hadamard(), (1,)),
        (qu.CNOT(), (0, 2)),
    ]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="mpo")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=2)

    assert out.L == 4
    assert out.max_bond() <= 8


def test_mpo_optimizer_mpo_mode_supports_three_site_gate():
    """MPO mode should accept a non-contiguous three-site gate support."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    z = np.diag([1.0, -1.0])
    local_xyz = np.kron(np.kron(x, y), z)
    gate = scipy_linalg.expm(-1j * 0.03 * local_xyz)
    mpo0 = qtn.MPO_identity(4, dtype="complex128")

    opt = py.MpoOptimizer(
        mpo0.copy(),
        gates=[((gate.T, None), (0, 1, 3))],
        chi=16,
        mode="mpo",
    )
    out = opt.run(progbar=False, cutoff=0.0, fidelity_samples=0)

    assert out.L == 4
    assert out.max_bond() <= 16
    assert not np.allclose(out.to_dense(), mpo0.to_dense())


def test_mpo_optimizer_mpo_mode_unitary_evolution_preserves_norm():
    """Two-sided unitary evolution in MPO mode should preserve the normalized norm."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (1, 2)),
        (qu.phase_gate(0.37), (3,)),
    ]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="mpo")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=len(gates))

    assert np.allclose(out.to_dense(), mpo0.to_dense(), atol=1e-10)
    assert all(np.isclose(val, 1.0, atol=1e-8) for val in opt.get_fidelities())


def test_mpo_optimizer_mpo_mode_ket_only_gate():
    """MPO mode with ket-only gate should not crash and should change the MPO."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [((qu.CNOT(), None), (0, 2))]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="mpo")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert out.L == 4
    assert not np.allclose(out.to_dense(), mpo0.to_dense())


def test_mpo_optimizer_mpo_mode_bra_only_gate():
    """MPO mode with bra-only gate should not crash and should change the MPO."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [((None, qu.CNOT()), (0, 2))]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="mpo")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert out.L == 4
    assert not np.allclose(out.to_dense(), mpo0.to_dense())


def _native_u1u1_identity_mpo(L=3):
    """Make a small native graded MPO fixture without testing construction."""
    pytest.importorskip("symmray")
    import symmray.utils as sr_utils

    phys_map = [(0, 0), (0, 1), (1, 0), (1, 1)]
    arrays = []
    for site in range(L):
        if site == 0:
            data = np.zeros((1, 4, 4), dtype="complex128")
            data[0] = np.eye(4)
            index_maps = [[(0, 0)], phys_map, phys_map]
            duals = (False, False, True)
        elif site == L - 1:
            data = np.zeros((1, 4, 4), dtype="complex128")
            data[0] = np.eye(4)
            index_maps = [[(0, 0)], phys_map, phys_map]
            duals = (True, False, True)
        else:
            data = np.zeros((1, 1, 4, 4), dtype="complex128")
            data[0, 0] = np.eye(4)
            index_maps = [[(0, 0)], [(0, 0)], phys_map, phys_map]
            duals = (True, False, False, True)
        arrays.append(
            sr_utils.from_dense(
                data,
                symmetry="U1U1",
                index_maps=index_maps,
                duals=duals,
                fermionic=True,
                charge=(0, 0),
            )
        )
    return qtn.MatrixProductOperator(
        arrays,
        shape="lrud",
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        site_tag_id="I{}",
    )


@pytest.mark.parametrize("mode", ["svd", "mpo", "dmrg"])
def test_mpo_optimizer_replays_native_graded_mpo_without_dense_fallback(mode):
    """Native graded MPO inputs remain FermionicArray-backed through replay."""
    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    gates = fermion.strang_gate_stream(
        [(0, 1), (1, 2)],
        dt=0.01,
        t=1.0,
        U=2.0,
        mu=0.1,
    )

    out = py.MpoOptimizer(
        _native_u1u1_identity_mpo(),
        gates=gates,
        chi=8,
        mode=mode,
    ).run(progbar=False, cutoff=1e-10, fidelity_samples=0, n_iter=1)

    assert all(type(tensor.data).__name__ == "U1U1FermionicArray" for tensor in out)


def test_mpo_optimizer_native_dmrg_uses_fit_controls(monkeypatch):
    """Native Symmray MPO DMRG must use block-aware FIT, not direct SVD."""
    calls = []
    original_run_gate = py.FIT.run_gate

    def recording_run_gate(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_run_gate(self, *args, **kwargs)

    monkeypatch.setattr(py.FIT, "run_gate", recording_run_gate)

    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    gates = fermion.strang_gate_stream(
        [(0, 2)],
        dt=0.01,
        t=1.0,
        U=2.0,
        mu=0.1,
    )
    out = py.MpoOptimizer(
        _native_u1u1_identity_mpo(),
        gates=gates,
        chi=8,
        mode="dmrg",
    ).run(
        progbar=False,
        cutoff=2.0e-2,
        cutoff_mode="rel",
        target_cutoff=0.0,
        fit_block_size=3,
        fit_sweep_sequence="L",
        fidelity_samples=0,
        n_iter=1,
    )

    assert calls
    assert all(call["block_size"] == 3 for call in calls)
    assert all(call["sweep_sequence"] == "L" for call in calls)
    assert all(call["max_bond"] == 8 for call in calls)
    assert all(call["cutoff"] == pytest.approx(2.0e-2) for call in calls)
    assert all(call["cutoff_mode"] == "rel" for call in calls)
    assert all(type(tensor.data).__name__ == "U1U1FermionicArray" for tensor in out)


@pytest.mark.parametrize(
    ("spinful", "symmetry", "params", "array_name"),
    [
        (False, "U1", {"V": 0.4, "mu": 0.1}, "U1FermionicArray"),
        (False, "Z2", {"V": 0.4, "mu": 0.1}, "Z2FermionicArray"),
        (True, "U1", {"U": 2.0, "mu": 0.1}, "U1FermionicArray"),
        (True, "U1U1", {"U": 2.0, "mu": 0.1}, "U1U1FermionicArray"),
        (True, "Z2", {"U": 2.0, "mu": 0.1}, "Z2FermionicArray"),
    ],
)
def test_mpo_optimizer_native_fermion_symmetries_use_dmrg_fit(
    spinful, symmetry, params, array_name,
):
    """Native fermion U1/U1U1/Z2 MPOs remain block-sparse under FIT."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(spinful=spinful, symmetry=symmetry)
    edges = [(0, 1), (1, 2), (2, 3)]
    hamiltonian = fermion.hamiltonian(edges, t=1.0, **params)
    mpo = fermion.to_mpo(hamiltonian=hamiltonian, L=4, compress=False)
    gates = fermion.gate_stream(edges, 0.002, t=1.0, **params)

    out = py.MpoOptimizer(
        mpo,
        gates=gates,
        chi=8,
        mode="dmrg",
    ).run(
        progbar=False,
        cutoff=1.0e-12,
        fidelity_samples=0,
        n_iter=1,
        fit_block_size=3,
    )

    assert out.max_bond() <= 8
    assert all(type(tensor.data).__name__ == array_name for tensor in out)


@pytest.mark.parametrize("mode", ["svd", "mpo"])
@pytest.mark.parametrize(
    ("spinful", "symmetry", "params", "array_name"),
    [
        (False, "U1", {"V": 0.4, "mu": 0.1}, "U1FermionicArray"),
        (False, "Z2", {"V": 0.4, "mu": 0.1}, "Z2FermionicArray"),
        (True, "U1", {"U": 2.0, "mu": 0.1}, "U1FermionicArray"),
        (True, "U1U1", {"U": 2.0, "mu": 0.1}, "U1U1FermionicArray"),
        (True, "Z2", {"U": 2.0, "mu": 0.1}, "Z2FermionicArray"),
    ],
)
def test_mpo_optimizer_native_fermion_symmetries_use_direct_modes(
    mode, spinful, symmetry, params, array_name,
):
    """Native fermion MPOs survive direct SVD and MPO-mode replay."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(spinful=spinful, symmetry=symmetry)
    edges = [(0, 3)]
    hamiltonian = fermion.hamiltonian(edges, t=1.0, **params)
    mpo = fermion.to_mpo(hamiltonian=hamiltonian, L=4, compress=False)
    gates = fermion.strang_gate_stream(edges, 0.002, t=1.0, **params)

    out = py.MpoOptimizer(
        mpo,
        gates=gates,
        chi=8,
        mode=mode,
    ).run(
        progbar=False,
        cutoff=2.0e-2,
        cutoff_mode="rel",
        fidelity_samples=0,
    )

    assert out.max_bond() <= 8
    assert all(type(tensor.data).__name__ == array_name for tensor in out)


def test_mpo_optimizer_materializes_native_long_range_split_gates():
    """Long-range native split gates are canonicalizable after replay."""
    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    gates = fermion.strang_gate_stream(
        [(0, 3)],
        dt=0.01,
        t=1.0,
        U=2.0,
        mu=0.1,
    )

    out = py.MpoOptimizer(
        _native_u1u1_identity_mpo(L=4),
        gates=gates,
        chi=8,
        mode="svd",
    ).run(progbar=False, cutoff=1e-10, fidelity_samples=0)

    assert out.L == 4
    assert all(len(out.tag_map[f"I{site}"]) == 1 for site in range(4))
    assert all(type(tensor.data).__name__ == "U1U1FermionicArray" for tensor in out)


def test_mpo_optimizer_adapts_long_range_native_gate_to_jw_symmray_mpo():
    """The current JW MPO path also handles long-range native even gates."""
    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    mpo = fermion.build_mpo(
        [(0, 3)],
        L=4,
        t=1.0,
        U=2.0,
        mu=0.1,
        fermionic=False,
    )
    gates = fermion.strang_gate_stream(
        [(0, 3)],
        dt=0.01,
        t=1.0,
        U=2.0,
        mu=0.1,
    )

    out = py.MpoOptimizer(mpo, gates=gates, chi=8, mode="svd").run(
        progbar=False,
        cutoff=1e-10,
        fidelity_samples=0,
    )

    assert out.L == 4
    assert all(type(tensor.data).__name__ == "U1U1Array" for tensor in out)


@pytest.mark.parametrize("mode", ["svd", "mpo", "dmrg"])
def test_mpo_optimizer_handles_fermion_symmray_mpo_and_native_gate_stream(mode):
    """The optimizer adapts native gates onto the current U1U1 MPO path."""
    pytest.importorskip("symmray")

    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    edges = [(0, 1), (1, 2)]
    mpo = fermion.build_mpo(
        edges,
        L=3,
        t=1.0,
        U=2.0,
        mu=0.1,
        fermionic=False,
        max_bond=16,
        cutoff=1e-12,
    )
    gates = fermion.strang_gate_stream(
        edges,
        dt=0.01,
        t=1.0,
        U=2.0,
        mu=0.1,
    )

    optimizer = py.MpoOptimizer(mpo, gates=gates, chi=8, mode=mode)
    out = optimizer.run(progbar=False, cutoff=1e-10, fidelity_samples=0, n_iter=1)

    assert out.L == 3
    assert out.max_bond() <= 8
    assert all(type(tensor.data).__name__ == "U1U1Array" for tensor in out)


def test_fermion_build_mpo_and_ham_tn_adapter_preserve_symmetry():
    """Both public MPO builders preserve the model's U1U1 symmetry."""
    pytest.importorskip("symmray")

    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    edges = [(0, 1), (1, 2)]
    builder = py.ham_tn(Lx=3, Ly=1, data_type="complex128")

    direct = fermion.build_mpo(
        edges, L=3, t=1.0, U=0.0, mu=0.0, fermionic=True,
    )
    adapted = builder.build_mpo(
        fermion=fermion,
        edges=edges,
        phys_dim=4,
        t=1.0,
        U=0.0,
        mu=0.0,
        fermionic=True,
    )
    positional = builder.build_mpo(
        fermion,
        edges=edges,
        t=1.0,
        U=0.0,
        mu=0.0,
        fermionic=True,
    )

    assert direct.L == adapted.L == positional.L == 3
    assert all(
        type(tensor.data).__name__ == "U1U1FermionicArray" for tensor in direct
    )
    assert all(
        type(tensor.data).__name__ == "U1U1FermionicArray" for tensor in adapted
    )
    assert all(
        type(tensor.data).__name__ == "U1U1FermionicArray"
        for tensor in positional
    )


def test_build_mpo_defaults_to_native_and_to_mpo_is_its_alias():
    """The model-facing builder has one native default and one alias."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(spinful=True, symmetry="U1U1")

    native = fermion.build_mpo(
        [(0, 1)],
        L=2,
        t=1.0,
        U=0.0,
        mu=0.0,
        compress=False,
    )
    direct = fermion.to_mpo(
        [(0, 1)],
        L=2,
        t=1.0,
        U=0.0,
        mu=0.0,
        compress=False,
    )

    assert all(
        type(tensor.data).__name__ == "U1U1FermionicArray"
        for tensor in native
    )
    assert type(fermion).to_mpo is type(fermion).build_mpo
    assert native.to_dense().allclose(direct.to_dense())


def test_mpo_optimizer_explicit_compress_handles_empty_symmray_stream():
    """The optimizer compresses symmetry-preserving Symmray MPOs directly."""
    pytest.importorskip("symmray")

    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    mpo = fermion.build_mpo(
        [(0, 1), (1, 2)],
        L=3,
        t=1.0,
        U=0.0,
        mu=0.0,
        fermionic=False,
        compress=False,
    )
    raw_bond = mpo.max_bond()
    optimizer = py.MpoOptimizer(mpo, gates=[], chi=2, mode="svd")
    out = optimizer.compress(cutoff=1e-10)

    # Symmray can retain a small sector-multiplicity overshoot for a requested
    # cap, but the compression must still reduce the raw MPO bond.
    assert out.max_bond() < raw_bond
    assert all(type(tensor.data).__name__ == "U1U1Array" for tensor in out)


def test_fermion_to_mpo_builds_native_mpo_for_optimizer_replay():
    """The native Fermion.to_mpo path feeds the MPO optimizer directly."""
    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    hopping = fermion.hopping_operator()
    two_site_mpo = fermion.to_mpo(
        {(0, 1): hopping},
        L=2,
        compress=False,
    )
    assert two_site_mpo.to_dense().allclose(hopping.fuse((0, 1), (2, 3)))

    hamiltonian = fermion.hamiltonian(
        [(0, 1), (1, 2)],
        t=1.0,
        U=2.0,
        mu=0.1,
    )
    mpo = fermion.to_mpo(
        hamiltonian=hamiltonian,
        L=3,
        max_bond=16,
        cutoff=1e-12,
    )

    assert all(type(tensor.data).__name__ == "U1U1FermionicArray" for tensor in mpo)

    out = py.MpoOptimizer(
        mpo,
        gates=hamiltonian.trotter_gates(0.01),
        chi=8,
        mode="svd",
    ).run(progbar=False, cutoff=1e-10, fidelity_samples=0)
    assert all(type(tensor.data).__name__ == "U1U1FermionicArray" for tensor in out)


def test_fermion_to_mpo_preserves_configured_backend():
    """Native MPO conversion applies the Fermion backend to every block."""
    torch = pytest.importorskip("torch")
    backend = py.backend_torch(dtype=torch.complex128, device="cpu")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        to_backend=backend,
    )

    mpo = fermion.to_mpo(
        [(0, 1)],
        L=2,
        t=1.0,
        U=2.0,
        mu=0.1,
        compress=False,
    )

    for tensor in mpo:
        assert tensor.data.backend == "torch"
        assert all(isinstance(block, torch.Tensor) for block in tensor.data.blocks.values())


def test_fermion_to_mpo_accepts_arbitrary_neutral_term_support():
    """Native MPO conversion supports non-contiguous multi-site terms."""
    fermion = py.Fermion(spinful=False, symmetry="U1")
    term = fermion.operator_term(
        [(1.0, ((2, "create"), (0, "number"), (3, "annihilate")))],
        sites=(2, 0, 3),
    )
    hamiltonian = fermion.hamiltonian({(2, 0, 3): term})
    mpo = fermion.to_mpo(
        hamiltonian=hamiltonian,
        L=4,
        compress=False,
    )

    assert mpo.L == 4
    assert all(type(tensor.data).__name__ == "U1FermionicArray" for tensor in mpo)
    embedded_term = fermion.operator_term(
        [(1.0, ((2, "create"), (0, "number"), (3, "annihilate")))],
        sites=(0, 1, 2, 3),
    )
    assert mpo.to_dense().allclose(
        embedded_term.fuse((0, 1, 2, 3), (4, 5, 6, 7))
    )


def test_fermion_to_mpo_handles_one_site_native_term():
    """Native MPO construction also handles the no-virtual-bond case."""
    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    term = fermion.interaction_operator()
    mpo = fermion.to_mpo({(0,): term}, L=1, compress=False)

    assert mpo.to_dense().allclose(term)
    assert mpo.pepsy_compression_report["raw_max_bond"] == 1

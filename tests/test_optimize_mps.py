"""Tests for :mod:`pepsy.optimizers.mps.optimizer`."""

from numbers import Real
import sys
import types

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy as py
import pepsy.optimizers.mps.optimizer as mps_optimizer_module


def _non_unitary_entangling_gate():
    """Return a small two-site filter that creates entanglement from |++>."""
    return np.diag([1.0, 0.5, 0.5, 2.0]).astype(complex)


def _two_branch_flip_submpo(*, L, sites, targets, w0=0.7, w1=0.3):
    """Return ``w0 * I + w1 * prod(X_targets)`` as a sparse-site MPO."""
    eye = np.eye(2, dtype=np.complex128)
    flip = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sites = tuple(sites)
    targets = set(targets)
    branch0 = [eye.copy() for _site in sites]
    branch1 = [flip.copy() if site in targets else eye.copy() for site in sites]
    branch0[0] *= w0
    branch1[0] *= w1
    mpo0 = qtn.MPO_product_operator(
        branch0,
        sites=sites,
        L=L,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
    )
    mpo1 = qtn.MPO_product_operator(
        branch1,
        sites=sites,
        L=L,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
    )
    return mpo0.add_MPO(mpo1)


def _mps_data_norm(mps):
    """Return the MPS norm without its stored global exponent."""
    mps_data = mps.copy()
    mps_data.exponent = 0.0
    return mps_data.norm()


def _perm_mps_to_logical_dense(opt):
    """Return a permuted-mode MPS dense state in logical site order."""
    physical = opt.p.to_dense().reshape((2,) * opt.p.L)
    logical_axes = [opt.qubits.index(site) for site in range(opt.p.L)]
    return np.transpose(physical, logical_axes).reshape(-1)


def _tensor_data_norm(mps, site):
    """Return the Frobenius norm of one MPS tensor's data."""
    return float(np.linalg.norm(np.asarray(mps[site].data)))


def _nonuniform_product_mps():
    """Return a non-translationally-invariant complex product state."""
    return qtn.MPS_product_state(
        [
            np.array([1.0, 0.0], dtype=complex),
            np.array([0.0, 1.0], dtype=complex),
            np.array([np.cos(0.3), np.sin(0.3)], dtype=complex),
            np.array([np.cos(0.5), 1j * np.sin(0.5)], dtype=complex),
        ]
    )


def _assert_event_sites_locally_normalized(mps, event):
    """Check that every tensor rescaled by an event has local norm one."""
    for site in event["sites"]:
        assert _tensor_data_norm(mps, site) == pytest.approx(1.0)


def test_mps_optimizer_accepts_svd_mode():
    """SVD mode should be accepted by ``MpsOptimizer`` mode validation."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0, gates=[], chi=8, mode="svd")
    assert opt.mode == "svd"


def test_mps_optimizer_accepts_perm_mode():
    """Perm mode should expose an identity logical-to-physical ordering initially."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0, gates=[], chi=8, mode="perm")

    assert opt.mode == "perm"
    assert opt.qubits == [0, 1, 2, 3]


def test_mps_optimizer_perm_tracks_lazy_order_and_logical_state():
    """Perm mode should leave swaps in place while preserving logical evolution."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (0, 2)),
    ]
    perm = py.MpsOptimizer(p0.copy(), gates=gates, chi=16, mode="perm")
    reference = py.MpsOptimizer(p0.copy(), gates=gates, chi=16, mode="swap")

    perm.run(progbar=False, cutoff=1e-12, fidelity_samples=0)
    reference.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert perm.qubits == [0, 2, 3, 1]
    assert np.allclose(_perm_mps_to_logical_dense(perm), reference.p.to_dense().reshape(-1))

    perm.restore_qubit_order()
    assert perm.qubits == [0, 1, 2, 3]
    assert np.allclose(perm.p.to_dense().reshape(-1), reference.p.to_dense().reshape(-1))


def test_mps_optimizer_perm_maps_control_events_to_logical_sites():
    """Controls after lazy swaps should still address their logical site labels."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000"),
        [
            (qu.hadamard(), (0,)),
            (qu.CNOT(), (0, 3)),
            ("measure", "Z", 3, +1),
        ],
        chi=8,
        mode="perm",
    )

    opt.run(progbar=False, fidelity_samples=0)

    assert opt.qubits == [0, 3, 1, 2]
    assert opt.measurements[0][1] == (3,)
    assert opt.measurements[0][2] == 1


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


def test_mps_optimizer_mix_warms_up_with_mpo_then_uses_dmrg():
    """Mix mode should use MPO below chi and DMRG after reaching chi."""
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 1)),
        (qu.CNOT(), (1, 2)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="mix")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0, n_iter=3)

    assert out.max_bond() <= 2
    assert [event["backend"] for event in opt.mix_history][:2] == ["mpo", "mpo"]
    assert opt.mix_history[-1]["backend"] == "dmrg"
    assert opt.last_mix_summary["mpo_steps"] == 2
    assert opt.last_mix_summary["dmrg_steps"] == 1
    assert opt.last_mix_summary["fallback_steps"] == 0


def test_mps_optimizer_mix_starts_with_dmrg_at_target_bond():
    """Mix mode should not do MPO warmup when the initial MPS is already at chi."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=17)
    assert p0.max_bond() == 2
    gates = [(qu.hadamard(), (1,))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="mix")
    opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert opt.mix_history[0]["backend"] == "dmrg"
    assert opt.mix_history[0]["reason"] == "bond_at_chi"


def test_mps_optimizer_mix_falls_back_to_mpo_on_nonfinite_dmrg(monkeypatch):
    """Mix mode should restore and use MPO if DMRG leaves non-finite data."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=19)
    gates = [(qu.hadamard(), (1,))]
    original_run_dmrg = py.MpsOptimizer._run_dmrg

    def nonfinite_dmrg(self, *args, **kwargs):
        original_run_dmrg(self, *args, **kwargs)
        data = np.asarray(self.p[0].data)
        self.p[0].modify(data=np.full_like(data, np.nan))

    monkeypatch.setattr(py.MpsOptimizer, "_run_dmrg", nonfinite_dmrg)
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="mix")

    opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert opt.mix_history[0]["backend"] == "mpo"
    assert opt.mix_history[0]["reason"] == "dmrg_fallback"
    assert "non-finite" in opt.mix_history[0]["fallback_error"]
    assert opt.last_mix_summary["fallback_steps"] == 1
    assert py.MpsOptimizer._mps_data_is_finite(opt.p)


def test_mps_optimizer_mix_rejects_non_unitary_stream_controls():
    """Mix mode is intentionally restricted to unitary streams."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    gates = [(qu.CNOT(), (0, 1))]
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="mix")

    with pytest.raises(ValueError, match="only for unitary"):
        opt.run(non_unitary=True, normalize_every=True)


def test_mps_optimizer_accepts_bundled_gate_stream():
    """Construction should accept ``[(gate, where), ...]`` with ``where=None``."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [(qu.hadamard(), (1,)), (qu.CNOT(), (0, 3))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12)

    assert opt.p.L == 4
    assert opt.where == [(1,), (0, 3)]


def test_mps_optimizer_forwards_custom_ind_id_to_gate_application():
    """Optimizer gate application should honor non-default physical indices."""
    p0 = qtn.MPS_computational_state("000", dtype=np.complex128)
    p0.reindex_({f"k{i}": f"b{i}" for i in range(3)})
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[(x_gate, (1,))],
        chi=4,
        mode="svd",
        ind_id="b{}",
    )
    out = opt.run(progbar=False, cutoff=1e-12)

    assert out is opt.p
    assert set(out.outer_inds()) == {"b0", "b1", "b2"}


def test_mps_optimizer_set_and_add_gates_accept_bundled_gate_stream():
    """set_gates/add_gates should accept bundled gate-stream entries."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0.copy(), chi=8, mode="svd")

    opt.set_gates([(qu.hadamard(), (1,))])
    opt.add_gates([(qu.CNOT(), (0, 3))])

    assert len(opt.G) == 2
    assert opt.where == [(1,), (0, 3)]


def test_mps_optimizer_layout_finder_api_is_separate_module():
    """Layout finder should live outside the optimizer implementation file."""
    from pepsy.optimizers.mps import MpsGateStreamLayoutFinder
    from pepsy.optimizers.mps.layout import MpsGateStreamLayoutFinder as LayoutFinder

    assert MpsGateStreamLayoutFinder is LayoutFinder
    assert py.MpsOptimizer.LayoutFinder is LayoutFinder


def test_mps_optimizer_gate_stream_layout_remaps_long_range_path():
    """Gate-stream layout should find a short order without changing the stream."""
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
        (qu.CNOT(), (1, 2)),
    ]

    plan = py.MpsOptimizer.gate_stream_layout(gates, L=4)

    assert set(plan["site_order"]) == {0, 1, 2, 3}
    assert plan["stats"]["max_span"] == 1
    assert plan["stats"]["long_range_events"] == 0
    assert plan["input_stats"]["long_range_events"] == 2
    assert plan["stats"]["loss"] <= plan["input_stats"]["loss"]
    assert plan["score"] == plan["stats"]["loss"]
    assert plan["layout"] == plan["site_map"]
    assert "recursive_refined" in plan["candidate_scores"]
    assert "gate_stream" not in plan
    assert "gates" not in plan
    assert plan["where"] == tuple(where for _gate, where in gates)
    assert set(plan["inverse_site_map"]) == {0, 1, 2, 3}
    assert all(
        abs(where[0] - where[1]) == 1
        for where in plan["mapped_where"]
    )


def test_mps_optimizer_gate_stream_layout_accepts_weight_fn():
    """User event weights should feed the weighted graph and report."""
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (1, 2)),
    ]

    def weight_fn(_payload, support, _event_type):
        return 10.0 if tuple(support) == (0, 3) else 1.0

    plan = py.MpsOptimizer.gate_stream_layout(
        gates,
        L=4,
        order="input",
        weight_fn=weight_fn,
    )

    assert plan["event_weights"] == (10.0, 1.0)
    assert plan["input_stats"]["total_edge_weight"] == pytest.approx(11.0)
    assert plan["input_stats"]["weighted_long_range_events"] == pytest.approx(10.0)


def test_mps_optimizer_gate_stream_layout_can_use_nevergrad():
    """Optional nevergrad candidate should be usable without touching streams."""
    pytest.importorskip("nevergrad")
    gates = [
        (qu.CNOT(), (0, 4)),
        (qu.CNOT(), (4, 1)),
        (qu.CNOT(), (1, 3)),
        (qu.CNOT(), (3, 2)),
    ]

    plan = py.MpsOptimizer.gate_stream_layout(
        gates,
        L=5,
        order="nevergrad",
        nevergrad_budget=8,
        refine_passes=1,
    )

    assert plan["selected_order"] == "nevergrad"
    assert "nevergrad" in plan["candidate_scores"]
    assert set(plan["site_order"]) == set(range(5))
    assert plan["where"] == tuple(where for _gate, where in gates)


def test_mps_optimizer_gate_stream_layout_kahypar_requires_config(monkeypatch):
    """Explicit KaHyPar layouts need a user-supplied config path."""
    monkeypatch.delenv("PEPSY_KAHYPAR_CONFIG", raising=False)
    gates = [(qu.CNOT(), (0, 3)), (qu.CNOT(), (3, 1))]

    with pytest.raises(ValueError, match="kahypar_config_path"):
        py.MpsOptimizer.gate_stream_layout(gates, L=4, order="kahypar")


def test_mps_optimizer_current_gate_stream_layout_uses_state_length():
    """Instance helper should include untouched MPS sites via ``p.L``."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[(qu.CNOT(), (0, 2))],
        chi=8,
        mode="svd",
    )

    plan = opt.current_gate_stream_layout(order="input")

    assert plan["site_order"] == (0, 1, 2, 3)
    assert plan["where"] == ((0, 2),)
    assert plan["mapped_where"] == ((0, 2),)


def test_mps_optimizer_gate_stream_layout_preserves_submpo_events():
    """Layout planning should not rewrite explicit sub-MPO stream events."""
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    stream = [
        py.MpsOptimizer.submpo_event(mpo, (0, 3)),
        (qu.CNOT(), (3, 1)),
    ]

    plan = py.MpsOptimizer.gate_stream_layout(stream, L=4)

    assert plan["event_types"] == ("submpo", "gate")
    assert stream[0][1] is mpo
    assert stream[0][2] == (0, 3)
    assert plan["where"][0] == (0, 3)
    assert plan["mapped_where"][0] != plan["where"][0]


def test_mps_optimizer_layout_run_restores_original_mps_order_and_stream():
    """Layout-aware replay should be internal and return original site labels."""
    p0 = qtn.MPS_computational_state("0101", dtype="complex128")
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
        (qu.CNOT(), (1, 2)),
    ]
    ref = py.MpsOptimizer(
        p0.copy(),
        gates=gates,
        chi=16,
        mode="svd",
    ).run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=16, mode="svd")
    out = opt.run(
        use_layout_finder=True,
        progbar=False,
        cutoff=1e-12,
        fidelity_samples=0,
    )

    inds = ["k0", "k1", "k2", "k3"]
    assert np.allclose(out.to_dense(inds), ref.to_dense(inds))
    assert out.outer_inds() == tuple(inds)
    assert opt.where == [(0, 3), (3, 1), (1, 2)]
    assert all(
        actual is expected
        for actual, (expected, _where) in zip(opt.G, gates)
    )
    assert opt.last_layout_plan is not None


def test_mps_optimizer_apply_layout_relabels_product_state_without_swaps(monkeypatch):
    """A nonuniform bond-one state should relabel without any SVD swaps."""
    calls = []
    original = qtn.MatrixProductState.swap_site_to_

    def fail_swap(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductState, "swap_site_to_", fail_swap)

    opt = py.MpsOptimizer(
        _nonuniform_product_mps(),
        gates=[(qu.CNOT(), (0, 3))],
        chi=8,
        mode="svd",
    )
    opt.apply_layout((0, 2, 3, 1), layout_report=False)

    assert calls == []
    assert opt.logical_order == [0, 2, 3, 1]
    assert opt.p.max_bond() == 1
    assert [opt.logical_site(pos) for pos in range(4)] == [0, 2, 3, 1]
    assert [opt.position(site) for site in range(4)] == [0, 3, 1, 2]


def test_mps_optimizer_persistent_layout_reuses_order_and_remaps_readout():
    """Persistent layout replay should agree with identity replay over repeats."""
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
    ]
    reference = py.MpsOptimizer(
        _nonuniform_product_mps(), gates=gates, chi=8, mode="svd"
    )
    reference.run(progbar=False, cutoff=1e-12, fidelity_samples=0)
    reference.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    laid_out = py.MpsOptimizer(
        _nonuniform_product_mps(), gates=gates, chi=8, mode="svd"
    )
    laid_out.apply_layout((0, 2, 3, 1), layout_report=False)
    laid_out.run(progbar=False, cutoff=1e-12, fidelity_samples=0)
    laid_out.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    reference_dense = np.asarray(reference.to_dense()).reshape(-1)
    laid_out_dense = np.asarray(laid_out.to_dense()).reshape(-1)
    overlap = np.vdot(reference_dense, laid_out_dense)
    assert abs(overlap) == pytest.approx(1.0, abs=1e-10)
    assert laid_out.logical_order == [0, 2, 3, 1]
    assert laid_out.p.max_bond() <= laid_out.chi
    assert laid_out.layout_plan is laid_out.last_layout_plan

    physical_configs = py.MpsSampler(laid_out.p, backend="quimb").sample(
        n_samples=24, seed=19
    ).configs_1d
    physical_dense = np.asarray(laid_out.p.to_dense()).reshape(-1)
    for physical_config in physical_configs:
        logical_config = laid_out.remap_sample(physical_config).tolist()
        physical_index = int("".join(map(str, physical_config)), 2)
        logical_index = int("".join(map(str, logical_config)), 2)
        assert abs(physical_dense[physical_index]) ** 2 == pytest.approx(
            abs(reference_dense[logical_index]) ** 2,
            abs=1e-10,
        )


def test_mps_optimizer_persistent_layout_rejects_entangled_state_by_default():
    """Entangled initialization needs explicit permission for one-time loss."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, bond_dim=2, dtype="complex128", seed=23),
        gates=[(qu.CNOT(), (0, 3))],
        chi=8,
        mode="svd",
    )
    before = np.asarray(opt.p.to_dense()).copy()

    with pytest.raises(ValueError, match="initially product MPS"):
        opt.apply_layout((0, 2, 3, 1), layout_report=False)

    assert opt.logical_order == [0, 1, 2, 3]
    assert np.allclose(np.asarray(opt.p.to_dense()), before)


def test_mps_optimizer_persistent_layout_entangled_reorder_uses_cutoff(monkeypatch):
    """Lossy persistent initialization should use the caller's cutoff once."""
    calls = []
    original = qtn.MatrixProductState.swap_site_to_

    def counting(self, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductState, "swap_site_to_", counting)
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, bond_dim=2, dtype="complex128", seed=23),
        gates=[(qu.CNOT(), (0, 3))],
        chi=8,
        mode="svd",
    )
    opt.apply_layout(
        (0, 2, 3, 1),
        cutoff=1e-7,
        allow_lossy_reorder=True,
        layout_report=False,
    )

    assert opt.logical_order == [0, 2, 3, 1]
    assert calls
    assert all(call["cutoff"] == pytest.approx(1e-7) for call in calls)


def test_mps_optimizer_persistent_layout_controls_keep_logical_labels():
    """Persistent layout control events execute physically but record logically."""
    opt = py.MpsOptimizer(
        _nonuniform_product_mps(),
        gates=[
            (qu.hadamard(), (3,)),
            (qu.CNOT(), (0, 3)),
            ("measure", "Z", 3, +1),
        ],
        chi=8,
        mode="mpo",
    )
    opt.apply_layout((0, 2, 3, 1), layout_report=False)
    opt.run(progbar=False, fidelity_samples=0)

    assert opt.measurements[0][0:3] == ("Z", (3,), 1)
    assert np.isclose(
        py.MpsOptimizer._real_float(
            opt._state_expectation("Z", (opt.position(3),))
        ),
        1.0,
    )


def test_mps_optimizer_persistent_layout_remaps_submpo_without_mutating_stream():
    """Persistent layout should copy/remap each sub-MPO on every replay."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    stream = [py.MpsOptimizer.submpo_event(mpo, (0, 3))]
    reference = py.MpsOptimizer(p0.copy(), gates=stream, chi=16, mode="mpo")
    reference.run(progbar=False, cutoff=1e-12, fidelity_samples=0)
    reference.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    opt = py.MpsOptimizer(p0.copy(), gates=stream, chi=16, mode="mpo")
    opt.apply_layout((0, 2, 3, 1), layout_report=False)
    opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)
    opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert np.allclose(
        np.abs(np.asarray(opt.to_dense()).reshape(-1)),
        np.abs(np.asarray(reference.to_dense()).reshape(-1)),
    )
    assert stream[0][1] is mpo
    assert stream[0][2] == (0, 3)


def test_mps_optimizer_persistent_layout_rejects_cap_events():
    """Persistent layout cannot survive a stream that changes MPS length."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000"),
        gates=[("cap", 1, [1.0, 1.0])],
        chi=8,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="cap control events"):
        opt.apply_layout((0, 2, 3, 1), layout_report=False)


def test_mps_optimizer_layout_run_reports_score_reduction(capsys):
    """Layout-aware replay should print a concise before/after report."""
    p0 = qtn.MPS_computational_state("0101", dtype="complex128")
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
        (qu.CNOT(), (1, 2)),
    ]
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=16, mode="svd")

    opt.run(
        use_layout_finder=True,
        progbar=False,
        cutoff=1e-12,
        fidelity_samples=0,
        layout_report=True,
    )

    report = capsys.readouterr().out
    assert "MpsOptimizer layout finder:" in report
    assert "long-range events:" in report
    assert "score:" in report
    assert "graph span:" in report


def test_mps_optimizer_layout_run_copies_submpo_payloads():
    """Layout replay should remap sub-MPO copies without mutating the stream."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    stream = [py.MpsOptimizer.submpo_event(mpo, (0, 3))]
    ref = py.MpsOptimizer(
        p0.copy(),
        gates=stream,
        chi=16,
        mode="mpo",
    ).run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    opt = py.MpsOptimizer(p0.copy(), gates=stream, chi=16, mode="mpo")
    out = opt.run(
        use_layout_finder=True,
        progbar=False,
        cutoff=1e-12,
        fidelity_samples=0,
    )

    inds = ["k0", "k1", "k2", "k3"]
    assert np.allclose(out.to_dense(inds), ref.to_dense(inds))
    assert out.outer_inds() == tuple(inds)
    assert out.site_inds == tuple(inds)
    assert stream[0][1] is mpo
    assert stream[0][2] == (0, 3)
    assert opt.where == [(0, 3)]


def test_mps_optimizer_mpo_mode_applies_submpo_stream_event():
    """MPO mode should apply explicit sparse sub-MPO stream events."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(1, 3), targets=(1, 3))
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (1, 3))],
        chi=8,
        mode="mpo",
    )

    out = opt.run(progbar=False, cutoff=0.0, fidelity_samples=0)
    vec = out.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 0.7
    expected[5] = 0.3

    assert opt.event_types == ["submpo"]
    assert opt.where == [(1, 3)]
    assert np.allclose(vec, expected)
    assert out.max_bond() <= 8


def test_mps_optimizer_mpo_mode_accepts_submpo_mapping_event():
    """Mapping events should provide a readable public sub-MPO stream API."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 2), targets=(0, 2))
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[{"kind": "submpo", "mpo": mpo, "where": [0, 2]}],
        chi=8,
        mode="mpo",
    )

    out = opt.run(progbar=False, cutoff=0.0, fidelity_samples=0)
    vec = out.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 0.7
    expected[10] = 0.3

    assert opt.event_types == ["submpo"]
    assert opt.where == [(0, 2)]
    assert np.allclose(vec, expected)


def test_mps_optimizer_public_submpo_event_helpers():
    """Public helpers should own the sub-MPO stream event contract."""
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 2), targets=(0, 2))
    tuple_event = py.MpsOptimizer.submpo_event(mpo, [0, 2])
    mapping_event = {"kind": "submpo", "mpo": mpo, "where": [0, 2]}
    gate_event = (np.eye(2), (0,))

    assert py.MpsOptimizer.is_submpo_event(tuple_event)
    assert py.MpsOptimizer.is_submpo_event(mapping_event)
    assert not py.MpsOptimizer.is_submpo_event(gate_event)

    assert py.MpsOptimizer.submpo_event_parts(tuple_event) == (mpo, (0, 2))
    assert py.MpsOptimizer.submpo_event_parts(
        mapping_event,
        normalize_where=True,
    ) == (mpo, (0, 2))
    assert py.MpsOptimizer.submpo_event_parts(gate_event) is None
    assert py.optimizers.mps.is_submpo_event(mapping_event)
    assert py.optimizers.mps.normalize_submpo_where([0, 2]) == (0, 2)

    bad_mapping = {"kind": "submpo", "mpo": mpo}
    with pytest.raises(ValueError, match="mpo.*where"):
        py.MpsOptimizer.submpo_event_parts(bad_mapping)


def test_mps_optimizer_submpo_diagnostics_do_not_consume_event_mpo():
    """Diagnostic target construction should not mutate reusable event MPOs."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(1, 3), targets=(1, 3))

    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (1, 3))],
        chi=8,
        mode="mpo",
    )
    out = opt.run(
        progbar=False,
        cutoff=0.0,
        fidelity_samples=0,
        non_unitary=True,
        track_norm_infidelity=True,
    )
    vec = out.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 0.7
    expected[5] = 0.3

    assert len(opt.get_norm_infidelity_samples()) == 1
    assert np.allclose(vec, expected)

    reuse = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (1, 3))],
        chi=8,
        mode="mpo",
    ).run(progbar=False, cutoff=0.0, fidelity_samples=0)
    reuse_vec = reuse.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    assert np.allclose(reuse_vec, expected)


def test_mps_optimizer_submpo_method_and_optimize_are_forwarded(monkeypatch):
    """Sub-MPO replay should expose compression method and optimizer choice."""
    p0 = qtn.MPS_computational_state("000000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=6, sites=(0, 5), targets=(0, 5))
    calls = []
    optimize = object()

    def fake_gate_with_submpo_(
        self,
        submpo,
        *,
        where=None,
        method="direct",
        info=None,
        optimize=None,
        **_kwargs,
    ):
        calls.append((submpo, tuple(where), method, optimize))
        if info is not None:
            info["cur_orthog"] = (min(where), min(where))
        return self

    monkeypatch.setattr(
        qtn.MatrixProductState,
        "gate_with_submpo_",
        fake_gate_with_submpo_,
    )

    direct = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (0, 5))],
        chi=8,
        mode="mpo",
        contraction_opt=optimize,
    )
    direct.run(
        progbar=False,
        cutoff=0.0,
        fidelity_samples=0,
        submpo_method="direct",
    )

    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (0, 5))],
        chi=8,
        mode="mpo",
        contraction_opt=optimize,
    )
    opt.run(
        progbar=False,
        cutoff=0.0,
        fidelity_samples=0,
        submpo_method="fit-zipup",
    )

    assert calls == [
        (mpo, (0, 5), "direct", None),
        (mpo, (0, 5), "fit-zipup", optimize),
    ]


def test_mps_optimizer_submpo_method_validation(monkeypatch):
    """Unknown sub-MPO methods should be rejected clearly."""
    p0 = qtn.MPS_computational_state("000000", dtype="complex128")
    short_mpo = _two_branch_flip_submpo(L=6, sites=(0, 3), targets=(0, 3))

    def fake_gate_with_submpo_(
        self,
        _submpo,
        *,
        where=None,
        method="direct",
        info=None,
        **_kwargs,
    ):
        if info is not None:
            info["cur_orthog"] = (min(where), min(where))
        return self

    monkeypatch.setattr(
        qtn.MatrixProductState,
        "gate_with_submpo_",
        fake_gate_with_submpo_,
    )

    bad = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(short_mpo, (0, 3))],
        chi=8,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="Unknown subMPO method"):
        bad.run(progbar=False, submpo_method="bad")


def test_mps_optimizer_submpo_stream_events_require_mpo_mode():
    """Non-MPO modes should reject sub-MPO stream events clearly."""
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=3, sites=(0, 2), targets=(0, 2))
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[("submpo", mpo, (0, 2))],
        chi=8,
        mode="svd",
    )

    with pytest.raises(ValueError, match="subMPO stream events"):
        opt.run(progbar=False)


def test_mps_optimizer_submpo_stream_validates_support_sites():
    """Sub-MPO support should be a unique in-range set of 1D sites."""
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=3, sites=(0, 2), targets=(0, 2))

    repeated = py.MpsOptimizer(
        p0.copy(),
        gates=[("submpo", mpo, (0, 0))],
        chi=8,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="repeated site"):
        repeated.run(progbar=False)

    out_of_range = py.MpsOptimizer(
        p0.copy(),
        gates=[("submpo", mpo, (0, 3))],
        chi=8,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="outside the MPS range"):
        out_of_range.run(progbar=False)


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


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "perm", "svd", "exact"])
def test_mps_optimizer_run_returns_state_after_updates(mode):
    """run() should return the updated MPS for every execution mode."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    G = [qu.hadamard(), qu.CNOT()]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode=mode)

    out = opt.run(progbar=False, cutoff=1e-12, n_iter=2, fidelity_samples=1)

    assert out is opt.p


@pytest.mark.parametrize("mode", ["mpo", "perm", "svd", "swap"])
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


def test_mps_optimizer_unitary_default_still_samples_norm_proxy():
    """Default/unitary runs should keep the historical norm-proxy sampling."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [(qu.hadamard(), (1,)), (qu.CNOT(), (0, 3))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12)

    assert len(opt.get_fidelities()) > 1


def test_mps_optimizer_svd_forwards_cutoff_mode_to_final_compression(monkeypatch):
    """SVD mode should honor explicit cutoff_mode in its chi compression pass."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [(qu.CNOT(), (0, 3))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="svd")
    calls = []
    original_left_compress = opt.p.left_compress

    def _recording_left_compress(*args, **kwargs):
        calls.append(dict(kwargs))
        return original_left_compress(*args, **kwargs)

    monkeypatch.setattr(opt.p, "left_compress", _recording_left_compress)

    opt.run(progbar=False, cutoff=1.0e-9, cutoff_mode="rsum2")

    assert calls
    assert calls[-1]["cutoff"] == pytest.approx(1.0e-9)
    assert calls[-1]["cutoff_mode"] == "rsum2"


def test_mps_optimizer_rejects_negative_fidelity_samples():
    """Negative fidelity_samples should fail clearly."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    G = [qu.CNOT()]
    where = [(0, 3)]
    gates = list(zip(G, where))

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="mpo")
    with pytest.raises(ValueError, match="fidelity_samples must be >= 0"):
        opt.run(progbar=False, cutoff=1e-12, fidelity_samples=-1)


def test_mps_optimizer_non_unitary_flag_normalizes_one_site_gate():
    """non_unitary=True should normalize at run end for short streams."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)

    opt = py.MpsOptimizer(p0.copy(), gates=[(scale, (1,))], chi=8, mode="svd")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=True,
        normalize_final=True,
    )

    events = opt.get_normalizations()
    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
    assert opt.p.norm() == pytest.approx(2.0)
    assert opt.p.exponent == pytest.approx(np.log10(2.0))
    assert opt.get_infidelities() == [0.0]
    assert opt.get_norm_infidelity_samples() == []
    assert len(events) == 1
    assert events[0]["step"] == 1
    assert events[0]["old_norm"] == pytest.approx(4.0)
    assert events[0]["span"] == (1, 1)
    assert events[0]["insert"] == 1
    assert events[0]["exponent"] == pytest.approx(np.log10(2.0))
    assert opt.info_c["cur_orthog"] == (1, 1)


def test_mps_optimizer_manual_normalize_accumulates_exponent():
    """Manual normalization should preserve represented norm via ``p.exponent``."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    p0[0].modify(data=2.0 * p0[0].data)

    opt = py.MpsOptimizer(p0.copy(), gates=[], chi=8, mode="svd")
    old_norm = opt.normalize(insert=0)

    assert old_norm == pytest.approx(4.0)
    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
    assert opt.p.norm() == pytest.approx(2.0)
    assert opt.p.exponent == pytest.approx(np.log10(2.0))


def test_mps_optimizer_non_unitary_default_does_not_normalize():
    """The non-unitary flag should not enable scale control by default."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
        (qu.hadamard(), (2,)),
        (_non_unitary_entangling_gate(), (2, 3)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt_none = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_final=False)
    opt_none.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=None,
        normalize_final=False,
    )

    assert opt.get_normalizations() == []
    assert opt_none.get_normalizations() == []
    assert opt.p.exponent == pytest.approx(0.0)
    assert opt_none.p.exponent == pytest.approx(0.0)


def test_mps_optimizer_non_unitary_scale_control_skips_diagnostics():
    """Fast non-unitary scale control should not collect diagnostics by default."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="svd")
    ref = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="svd")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=True,
        normalize_final=True,
    )
    ref.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_every=False)

    events = opt.get_normalizations()
    assert opt.p.norm() == pytest.approx(ref.p.norm())
    assert opt.p.exponent == pytest.approx(sum(event["log10_scale"] for event in events))
    assert opt.get_fidelities() == [1.0]
    assert opt.get_infidelities() == [0.0]
    assert opt.get_true_infidelities() == [0.0]
    assert opt.get_infidelity_samples() == []
    assert opt.get_norm_infidelity_samples() == []
    assert [event["step"] for event in events] == [3]
    assert events[0]["reason"] == "compression"
    _assert_event_sites_locally_normalized(opt.p, events[0])


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "swap", "perm", "svd"])
def test_mps_optimizer_normalization_insert_site_stays_inside_span(mode):
    """Normalization events should insert factors inside the canonical span."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [(qu.CNOT(), (0, 1)), (qu.CNOT(), (2, 3))]
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode=mode)
    if mode == "swap" and not hasattr(opt.p, "gate_with_auto_swap_"):
        pytest.skip("swap mode requires gate_with_auto_swap_ in this quimb version.")

    opt.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_every=1)

    assert opt.p.norm() == pytest.approx(1.0)
    assert [event["span"] for event in opt.get_normalizations()] == [(0, 1), (2, 3)]
    assert all(
        event["span"][0] <= event["insert"] <= event["span"][1]
        for event in opt.get_normalizations()
    )


def test_mps_optimizer_normalize_every_enables_compression_and_final():
    """Enabled normalize_every should scale compressed updates and final tails."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    gates = [(scale, (0,)), (_non_unitary_entangling_gate(), (0, 1)), (scale, (0,))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    ref = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12, normalize_every=2, non_unitary=True, normalize_final=True)
    ref.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_every=False)

    events = opt.get_normalizations()
    assert opt.p.norm() == pytest.approx(ref.p.norm())
    assert [event["step"] for event in events] == [2, 3]
    assert [event["reason"] for event in events] == ["compression", "final"]
    assert opt.p.exponent == pytest.approx(sum(event["log10_scale"] for event in events))
    _assert_event_sites_locally_normalized(opt.p, events[-1])


def test_mps_optimizer_normalize_final_can_be_disabled():
    """normalize_final=False should skip the trailing normalization."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    gates = [(scale, (0,)), (_non_unitary_entangling_gate(), (0, 1)), (scale, (0,))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    ref = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12, normalize_every=2, normalize_final=False, non_unitary=True)
    ref.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_every=False)

    assert opt.p.norm() == pytest.approx(ref.p.norm())
    assert [event["step"] for event in opt.get_normalizations()] == [2]
    assert opt.get_normalizations()[0]["reason"] == "compression"


def test_mps_optimizer_automatic_normalization_rejects_exact_mode():
    """Exact mode has no MPS canonicalization range for automatic normalization."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    opt = py.MpsOptimizer(p0.copy(), gates=[(scale, (0,))], chi=8, mode="exact")

    with pytest.raises(ValueError, match="not available in exact mode"):
        opt.run(progbar=False, non_unitary=True, normalize_every=True)


def test_mps_optimizer_track_infidelity_rejects_exact_mode():
    """Exact mode has no compressed target for infidelity diagnostics."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    opt = py.MpsOptimizer(p0.copy(), gates=[(qu.CNOT(), (0, 1))], chi=8, mode="exact")

    with pytest.raises(ValueError, match="not available in exact mode"):
        opt.run(progbar=False, track_infidelity=True)


def test_mps_optimizer_exact_mode_keeps_canonical_metadata_separate():
    """Switching through exact mode rebuilds an MPS before canonical use."""
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    gates = [(qu.hadamard(), (0,)), (qu.CNOT(), (0, 2))]
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")

    opt.set_mode("exact")
    opt.run(progbar=False)
    assert opt.info_c == {}

    exact_dense = opt.to_dense()
    opt.set_gates([])
    opt.set_mode("svd")

    assert isinstance(opt.p, qtn.MatrixProductState)
    assert opt.info_c["cur_orthog"] not in (None, "calc")
    assert np.allclose(opt.p.to_dense().reshape(-1), exact_dense)


def test_mps_optimizer_persistent_layout_rejects_exact_mode_switch():
    """Exact mode cannot silently discard a persistent logical-site map."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000"), gates=[], chi=4, mode="svd"
    )
    opt.apply_layout((2, 0, 1), layout_report=False)

    with pytest.raises(ValueError, match="persistent-layout"):
        opt.set_mode("exact")


def test_mps_optimizer_rejects_invalid_normalize_every():
    """normalize_every should fail clearly for non-positive intervals."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    scale = np.eye(2, dtype=complex)
    opt = py.MpsOptimizer(p0.copy(), gates=[(scale, (0,))], chi=8, mode="svd")

    with pytest.raises(ValueError, match="normalize_every must be >= 1"):
        opt.run(progbar=False, normalize_every=0, non_unitary=True)


def test_mps_optimizer_normalization_options_require_non_unitary():
    """Normalization controls should not act as aliases for ``non_unitary=True``."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    scale = np.eye(2, dtype=complex)
    opt = py.MpsOptimizer(p0.copy(), gates=[(scale, (0,))], chi=8, mode="svd")

    with pytest.raises(ValueError, match="normalize_every requires non_unitary=True"):
        opt.run(progbar=False, normalize_every=1)

    with pytest.raises(ValueError, match="normalize_final requires non_unitary=True"):
        opt.run(progbar=False, normalize_final=True)


@pytest.mark.parametrize("where", [(1, 2), (0, 3)])
def test_mps_optimizer_canonical_span_norm_matches_full_target_norm(where):
    """Canonical span norm should match full norm for split-gate targets."""
    p0 = qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128")
    gate = _non_unitary_entangling_gate()
    opt = py.MpsOptimizer(p0.copy(), gates=[], chi=8, mode="svd")

    xmin, xmax = sorted(where)
    opt.canonize_mps(opt.p, (xmin, xmax))
    target = opt._build_norm_target(  # pylint: disable=protected-access
        opt.p,
        gate,
        where,
        cutoff=1e-12,
        cutoff_mode="rel",
    )

    local_norm = opt._canonical_span_norm(target, (xmin, xmax))  # pylint: disable=protected-access
    assert local_norm == pytest.approx(target.norm())


def test_mps_optimizer_target_norm_does_not_mutate_live_canonical_metadata():
    """Temporary norm targets must not overwrite the live MPS center cache."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128", seed=8),
        gates=[],
        chi=8,
        mode="svd",
    )
    opt.canonize_mps(opt.p, (0, 1))
    before = dict(opt.info_c)
    target = opt._build_norm_target(  # pylint: disable=protected-access
        opt.p,
        _non_unitary_entangling_gate(),
        (0, 3),
        cutoff=1e-12,
    )

    measured = opt._canonical_span_norm(  # pylint: disable=protected-access
        target, (0, 3)
    )

    assert measured == pytest.approx(target.norm())
    assert opt.info_c == before


def test_mps_optimizer_local_normalization_reuses_tracked_center(monkeypatch):
    """Local scale control should not rescan a live canonical MPS."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128", seed=9),
        gates=[],
        chi=8,
        mode="svd",
    )
    opt.canonize_mps(opt.p, (0, 2))

    def fail_scan(*args, **kwargs):
        raise AssertionError("normalization should reuse the tracked centre")

    monkeypatch.setattr(qtn.MatrixProductState, "calc_current_orthog_center", fail_scan)
    event = opt._normalize_orthog_tensors(  # pylint: disable=protected-access
        opt.p,
        (0, 2),
        step=1,
        reason="test",
        canonicalize=False,
    )

    assert event is not None
    assert event["insert"] in (0, 1, 2)
    assert opt.info_c["cur_orthog"] == (0, 2)


def test_mps_optimizer_canonical_span_norm_ignores_stored_exponent():
    """Internal normalization should measure raw data, not represented scale."""
    p0 = qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128")
    opt = py.MpsOptimizer(p0.copy(), gates=[], chi=8, mode="svd")
    opt.p.exponent = 3.0

    raw = opt.p.copy()
    raw.exponent = 0.0
    measured = opt._canonical_span_norm(opt.p, (0, 3))  # pylint: disable=protected-access

    assert measured == pytest.approx(raw.norm())
    assert opt.p.exponent == pytest.approx(3.0)


def test_mps_optimizer_norm_infidelity_uses_single_center_norm(monkeypatch):
    """Norm diagnostics should avoid a full doubled-network contraction."""
    def _fail_tn_norm(*args, **kwargs):
        raise AssertionError("single-center norm should not call tn_norm")

    monkeypatch.setattr(mps_optimizer_module, "tn_norm", _fail_tn_norm, raising=False)

    p0 = qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128", seed=3)
    opt = py.MpsOptimizer(p0.copy(), gates=[], chi=8, mode="svd")
    raw = opt.p.copy()
    raw.exponent = 0.0

    measured = opt._canonical_span_norm(opt.p, (0, 3))  # pylint: disable=protected-access

    assert measured == pytest.approx(raw.norm())
    assert opt.info_c["cur_orthog"] == (3, 3)


def test_mps_optimizer_unitary_norm_infidelity_mpo_skips_target_build(monkeypatch):
    """Unitary MPO norm diagnostics should not build an unbounded target MPS."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 3)),
    ]

    def fail_build_target(*_args, **_kwargs):
        raise AssertionError("unitary norm tracking should use the pre-gate norm")

    monkeypatch.setattr(py.MpsOptimizer, "_build_norm_target", fail_build_target)
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="mpo")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        fidelity_samples=0,
        track_norm_infidelity=True,
    )

    samples = opt.get_norm_infidelity_samples()
    assert len(samples) == 1
    assert samples[0]["step"] == 2
    assert samples[0]["where"] == (0, 3)
    assert samples[0]["target_norm"] == pytest.approx(1.0)
    assert opt.p.max_bond() <= 1


def test_mps_optimizer_unitary_submpo_norm_infidelity_skips_target_gate(monkeypatch):
    """Unitary sub-MPO norm diagnostics should not apply an uncapped target."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    flip = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    mpo = qtn.MPO_product_operator(
        [flip.copy(), flip.copy()],
        sites=(0, 3),
        L=4,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
    )
    max_bonds = []
    original_gate_with_submpo = qtn.MatrixProductState.gate_with_submpo_

    def wrapped_gate_with_submpo(self, *args, **kwargs):
        max_bonds.append(kwargs.get("max_bond"))
        if kwargs.get("max_bond") is None:
            raise AssertionError("unitary sub-MPO diagnostics should be target-free")
        return original_gate_with_submpo(self, *args, **kwargs)

    monkeypatch.setattr(
        qtn.MatrixProductState,
        "gate_with_submpo_",
        wrapped_gate_with_submpo,
    )
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (0, 3))],
        chi=1,
        mode="mpo",
    )
    opt.run(
        progbar=False,
        cutoff=1e-12,
        fidelity_samples=0,
        track_norm_infidelity=True,
    )

    samples = opt.get_norm_infidelity_samples()
    assert max_bonds == [1]
    assert len(samples) == 1
    assert samples[0]["target_norm"] == pytest.approx(1.0)
    assert samples[0]["approx_norm"] == pytest.approx(1.0)


def test_mps_optimizer_non_unitary_norm_infidelity_matches_svd_target():
    """SVD non-unitary proxy should match quimb's target infidelity."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
    ]

    target = p0.copy()
    for gate, where in gates[:2]:
        py.gate(target, gate, where, contract=True, cutoff=1e-12, inplace=True)
    target = py.gate(
        target,
        gates[-1][0],
        gates[-1][1],
        contract="reduce-split",
        cutoff=1e-12,
        cutoff_mode="rel",
        inplace=False,
    )

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="svd")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=True,
        normalize_final=True,
        track_norm_infidelity=True,
    )

    samples = opt.get_norm_infidelity_samples()
    proxy = opt.get_infidelities()[-1]
    actual = target.distance_normalized(opt.p, normalized="infidelity", optimize="auto-hq")

    assert len(samples) == 1
    assert samples[0]["step"] == 3
    assert samples[0]["where"] == (0, 1)
    assert samples[0]["local_infidelity"] == pytest.approx(proxy)
    assert proxy == pytest.approx(float(np.real(actual)))
    _assert_event_sites_locally_normalized(opt.p, opt.get_normalizations()[-1])
    assert opt.p.norm() > 0.0


def test_mps_optimizer_true_infidelity_matches_tn_fidelity():
    """track_infidelity=True should record the true normalized-overlap metric."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
    ]

    target = p0.copy()
    for gate, where in gates[:2]:
        py.gate(target, gate, where, contract=True, cutoff=1e-12, inplace=True)
    target = py.gate(
        target,
        gates[-1][0],
        gates[-1][1],
        contract="reduce-split",
        cutoff=1e-12,
        cutoff_mode="rel",
        inplace=False,
    )

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="svd")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=True,
        normalize_final=True,
        track_infidelity=True,
    )

    samples = opt.get_infidelity_samples()
    actual_fidelity = float(np.real(py.tn_fidelity(opt.p, target, contraction_opt="auto-hq")))
    actual_fidelity = min(1.0, max(0.0, actual_fidelity))

    assert len(samples) == 1
    assert samples[0]["step"] == 3
    assert samples[0]["where"] == (0, 1)
    assert samples[0]["fidelity"] == pytest.approx(actual_fidelity)
    assert samples[0]["local_infidelity"] == pytest.approx(1.0 - actual_fidelity)
    assert opt.get_true_infidelities()[-1] == pytest.approx(1.0 - actual_fidelity)
    assert opt.get_infidelities()[-1] == pytest.approx(1.0 - actual_fidelity)
    assert opt.get_norm_infidelity_samples() == []
    _assert_event_sites_locally_normalized(opt.p, opt.get_normalizations()[-1])
    assert opt.p.norm() > 0.0


def test_mps_optimizer_true_infidelity_losses_are_geometric_mean():
    """True-fidelity tracking should store geometric-mean progress in losses."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    h_gate = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    filter_gate = _non_unitary_entangling_gate()
    gates = [
        (h_gate, (0,)),
        (h_gate, (1,)),
        (h_gate, (2,)),
        (h_gate, (3,)),
        (filter_gate, (0, 3)),
        (filter_gate, (1, 2)),
        (filter_gate, (0, 1)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="svd")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=1,
        fidelity_samples=5,
        track_infidelity=True,
    )

    log_sum = 0.0
    expected = []
    for count, sample in enumerate(opt.get_infidelity_samples(), start=1):
        fidelity = sample["fidelity"]
        if fidelity <= 0.0 or np.isneginf(log_sum):
            log_sum = -np.inf
            geometric_fidelity = 0.0
        else:
            log_sum += np.log(fidelity)
            geometric_fidelity = float(np.exp(log_sum / count))
        expected.append(geometric_fidelity)
        assert sample["geometric_fidelity"] == pytest.approx(geometric_fidelity)

    assert len(opt.get_infidelity_samples()) == 3
    assert opt.get_fidelities()[0] == pytest.approx(1.0)
    assert opt.get_fidelities()[1:] == pytest.approx(expected)
    assert len(opt.get_fidelities()) == len(opt.get_infidelity_samples()) + 1


def test_mps_optimizer_true_infidelity_progress_reports_infidelity(monkeypatch):
    """With track_infidelity=True, tqdm postfix should show infidelity."""
    progress_instances = []

    class _FakeTqdm:
        def __init__(self, **kwargs):
            self.total = kwargs["total"]
            self.n = 0
            self.postfix_calls = []
            progress_instances.append(self)

        def set_postfix(self, postfix):
            self.postfix_calls.append(dict(postfix))

        def update(self, amount):
            self.n += amount

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "tqdm", types.SimpleNamespace(tqdm=_FakeTqdm))

    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (_non_unitary_entangling_gate(), (0, 3)),
    ]
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="svd")

    opt.run(
        progbar=True,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=1,
        track_infidelity=True,
    )

    progress = progress_instances[-1]
    assert progress.n == len(gates)
    last = progress.postfix_calls[-1]
    assert "Icum" in last
    assert "Fgeom" not in last
    assert last["Icum"] == opt._format_progress_infidelity(opt.get_infidelities()[-1])


def test_mps_optimizer_norm_infidelity_progress_reports_norm_proxy(monkeypatch):
    """With track_norm_infidelity=True, MPO tqdm postfix should show norm loss."""
    progress_instances = []

    class _FakeTqdm:
        def __init__(self, **kwargs):
            self.total = kwargs["total"]
            self.n = 0
            self.postfix_calls = []
            progress_instances.append(self)

        def set_postfix(self, postfix):
            self.postfix_calls.append(dict(postfix))

        def update(self, amount):
            self.n += amount

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "tqdm", types.SimpleNamespace(tqdm=_FakeTqdm))

    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 3)),
    ]
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="mpo")

    opt.run(
        progbar=True,
        cutoff=1e-12,
        fidelity_samples=0,
        track_norm_infidelity=True,
    )

    progress = progress_instances[-1]
    assert progress.n == len(gates)
    last = progress.postfix_calls[-1]
    assert "infidelity" in last
    assert "Icum" not in last
    assert last["infidelity"] == opt._format_progress_infidelity(opt.get_infidelities()[-1])


def test_mps_optimizer_progress_infidelity_uses_compact_scientific_format():
    """Tiny displayed infidelities should not round to 0.000000."""
    assert py.MpsOptimizer._format_progress_infidelity(1e-9) == "1.e-9"
    assert py.MpsOptimizer._format_progress_infidelity(0.0) == "0.e+0"


@pytest.mark.parametrize("mode", ["dmrg", "mpo"])
def test_mps_optimizer_true_infidelity_smoke_other_modes(mode):
    """Other compressed modes should support the true infidelity diagnostic."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode=mode)
    opt.run(
        progbar=False,
        cutoff=1e-12,
        n_iter=2,
        non_unitary=True,
        normalize_every=True,
        normalize_final=True,
        track_infidelity=True,
    )

    samples = opt.get_infidelity_samples()
    assert len(samples) == 1
    assert 0.0 <= samples[0]["fidelity"] <= 1.0
    assert 0.0 <= samples[0]["local_infidelity"] <= 1.0
    assert 0.0 <= opt.get_true_infidelities()[-1] <= 1.0
    assert opt.get_norm_infidelity_samples() == []
    _assert_event_sites_locally_normalized(opt.p, opt.get_normalizations()[-1])
    assert opt.p.norm() > 1.0


def test_mps_optimizer_dmrg_non_unitary_matches_mpo_accuracy():
    """DMRG should match MPO accuracy for normalized non-unitary MPS updates."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    h_gate = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    filter_gate = _non_unitary_entangling_gate()
    gates = [
        (h_gate, (0,)),
        (h_gate, (1,)),
        (h_gate, (2,)),
        (h_gate, (3,)),
        (filter_gate, (0, 3)),
        (filter_gate, (1, 2)),
        (filter_gate, (0, 1)),
    ]

    target = p0.copy()
    for gate, where in gates:
        py.gate(target, gate, where, contract=True, cutoff=1e-12, inplace=True)

    results = {}
    for mode in ("dmrg", "mpo"):
        opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode=mode)
        opt.run(
            progbar=False,
            cutoff=1e-12,
            n_iter=20,
            non_unitary=True,
            normalize_every=1,
            track_infidelity=True,
        )

        fidelity = float(np.real(py.tn_fidelity(opt.p, target, contraction_opt="auto-hq")))
        results[mode] = {
            "fidelity": fidelity,
            "represented_norm": float(np.real(opt.p.norm())),
            "cumulative_infidelity": opt.get_infidelities()[-1],
        }

        events = opt.get_normalizations()
        assert len(events) == 3
        assert all(event["reason"] == "compression" for event in events)
        _assert_event_sites_locally_normalized(opt.p, events[-1])
        assert len(opt.get_infidelity_samples()) == 3
        assert fidelity > 0.92

    assert results["dmrg"]["fidelity"] == pytest.approx(
        results["mpo"]["fidelity"],
        abs=1e-12,
    )
    assert results["dmrg"]["represented_norm"] == pytest.approx(
        results["mpo"]["represented_norm"],
        abs=1e-12,
    )
    assert results["dmrg"]["cumulative_infidelity"] == pytest.approx(
        results["mpo"]["cumulative_infidelity"],
        abs=1e-12,
    )


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "swap", "perm", "svd"])
def test_mps_optimizer_non_unitary_norm_infidelity_smoke_other_modes(mode):
    """All compressed modes should expose a bounded non-unitary proxy."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode=mode)
    if mode == "swap" and not hasattr(opt.p, "gate_with_auto_swap_"):
        pytest.skip("swap mode requires gate_with_auto_swap_ in this quimb version.")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        n_iter=2,
        non_unitary=True,
        normalize_every=True,
        normalize_final=True,
        track_norm_infidelity=True,
    )

    samples = opt.get_norm_infidelity_samples()
    assert len(samples) == 1
    assert 0.0 <= samples[0]["local_infidelity"] <= 1.0
    assert 0.0 <= opt.get_infidelities()[-1] <= 1.0
    _assert_event_sites_locally_normalized(opt.p, opt.get_normalizations()[-1])
    assert opt.p.norm() > 1.0


# --------------------------------------------------------------------------- #
# Control events: measure / cap / reset
# --------------------------------------------------------------------------- #
_PAULI_1Q_TEST = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _dense_pauli_expectation(mps, pauli, where):
    """Return ``<psi|P|psi> / <psi|psi>`` from the dense statevector."""
    psi = mps.to_dense().reshape(-1)
    ops = [np.eye(2, dtype=complex) for _ in range(mps.L)]
    for axis, site in zip(pauli, where):
        ops[site] = _PAULI_1Q_TEST[axis]
    operator = ops[0]
    for op in ops[1:]:
        operator = np.kron(operator, op)
    return complex(psi.conj() @ (operator @ psi) / (psi.conj() @ psi)).real


def _full_network_pauli_expectation(mps, pauli, where, optimize="auto-hq"):
    """Return a Pauli expectation from an explicit full MPS overlap."""
    op = _PAULI_1Q_TEST[pauli[0]]
    for axis in pauli[1:]:
        op = np.kron(op, _PAULI_1Q_TEST[axis])

    acted = mps.copy()
    acted.gate_nonlocal_(
        op,
        tuple(int(site) for site in where),
        max_bond=None,
        info={},
        method="direct",
        cutoff=0.0,
        cutoff_mode="abs",
    )
    numerator = (mps.H & acted).contract(all, output_inds=(), optimize=optimize)
    denominator = (mps.H & mps).contract(all, output_inds=(), optimize=optimize)
    return float(np.real(complex(numerator / denominator)))


def test_mps_optimizer_measure_forced_outcome_collapses_and_records():
    """A forced measurement should collapse the state and record the result."""
    m = qtn.MPS_rand_state(6, 4, seed=2)
    opt = py.MpsOptimizer(m.copy(), [("measure", "Z", 2, +1)], chi=8, mode="mpo")
    opt.run(progbar=False)

    assert np.isclose(_dense_pauli_expectation(opt.p, "Z", (2,)), 1.0)
    assert np.isclose(float(abs(opt.p.norm())), 1.0)
    assert len(opt.measurements) == 1
    pauli, where, outcome, prob = opt.measurements[0]
    assert pauli == "Z"
    assert where == (2,)
    assert outcome == 1
    assert 0.0 <= prob <= 1.0


def test_mps_optimizer_measure_multisite_pauli():
    """Multi-qubit Pauli measurements should collapse onto the eigenspace."""
    m = qtn.MPS_rand_state(6, 4, seed=2)
    opt = py.MpsOptimizer(m.copy(), [("measure", "ZZ", (1, 3), -1)], chi=8, mode="mpo")
    opt.run(progbar=False)

    assert np.isclose(_dense_pauli_expectation(opt.p, "ZZ", (1, 3)), -1.0)
    assert opt.measurements[0][:3] == ("ZZ", (1, 3), -1)


def test_mps_optimizer_expectation_uses_local_canonical_path(monkeypatch):
    """MPS expectations should use Quimb's local canonical evaluator."""
    calls = []
    original = qtn.MatrixProductState.local_expectation_canonical

    def counting(self, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductState, "local_expectation_canonical", counting)

    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(6, 4, seed=7), gates=[], chi=8, mode="mpo"
    )
    observed = opt._state_expectation("ZZ", (1, 4))  # pylint: disable=protected-access
    expected = _dense_pauli_expectation(opt.p, "ZZ", (1, 4))

    assert observed == pytest.approx(expected)
    assert len(calls) == 1
    assert calls[0]["normalized"] is True
    assert calls[0]["info"] is opt.info_c


@pytest.mark.parametrize(
    ("pauli", "where"),
    [("X", (4,)), ("YZ", (1, 4))],
)
def test_mps_optimizer_expectation_reuses_tracked_center_without_rescan(
    monkeypatch, pauli, where
):
    """Local Pauli expectations should move a known centre without rescanning."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(6, 4, seed=11), gates=[], chi=8, mode="mpo"
    )
    opt.canonize_mps(opt.p, 0)
    assert opt.info_c["cur_orthog"] == (0, 0)

    def fail_scan(*args, **kwargs):
        raise AssertionError("expectation should reuse the tracked canonical centre")

    monkeypatch.setattr(qtn.MatrixProductState, "calc_current_orthog_center", fail_scan)

    observed = opt._state_expectation(pauli, where)  # pylint: disable=protected-access
    expected = _dense_pauli_expectation(opt.p, pauli, where)

    assert observed == pytest.approx(expected)
    center = opt.info_c["cur_orthog"]
    assert center[0] == center[1]
    assert min(where) <= center[0] <= max(where)


@pytest.mark.parametrize(
    ("pauli", "where"),
    [("X", (4,)), ("YZ", (1, 4))],
)
def test_mps_optimizer_local_expectation_matches_full_network(pauli, where):
    """Local canonical and full-network Pauli expectations should agree."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(6, 4, seed=13), gates=[], chi=8, mode="mpo"
    )

    local = opt._state_expectation(pauli, where)  # pylint: disable=protected-access
    full = _full_network_pauli_expectation(opt.p, pauli, where)

    assert local == pytest.approx(full, abs=1e-10)


def test_mps_optimizer_measure_born_statistics():
    """Sampled outcomes should follow the Born rule for a biased qubit."""
    theta = np.pi / 3
    ry = np.array(
        [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=complex,
    )
    n_shots = 800
    plus = 0
    for shot in range(n_shots):
        opt = py.MpsOptimizer(
            qtn.MPS_computational_state("0"),
            [(ry, (0,)), ("measure", "Z", 0)],
            chi=2,
            mode="mpo",
        )
        opt.run(progbar=False, seed=shot)
        if opt.measurements[0][2] == 1:
            plus += 1
    expected = np.cos(theta / 2) ** 2
    assert abs(plus / n_shots - expected) < 0.05


def test_mps_optimizer_measure_forced_zero_probability_raises():
    """Forcing an impossible outcome should fail clearly."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0"),
        [("measure", "Z", 0, -1)],
        chi=2,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="probability"):
        opt.run(progbar=False)


def test_mps_optimizer_cap_matches_dense_projection_and_shortens():
    """A cap event should shorten the MPS and match the dense contraction."""
    m = qtn.MPS_rand_state(6, 4, seed=2)
    vec = np.array([1.0, 1.0])
    dense = m.to_dense().reshape([2] * 6)
    expected = np.tensordot(dense, vec, axes=([2], [0]))

    for absorb in ("left", "right"):
        opt = py.MpsOptimizer(m.copy(), [("cap", 2, vec, absorb)], chi=8, mode="mpo")
        opt.run(progbar=False)
        assert isinstance(opt.p, qtn.MatrixProductState)
        assert opt.p.L == 5
        got = opt.p.to_dense().reshape([2] * 5)
        assert np.allclose(got, expected)


def test_mps_optimizer_cap_boundary_sites():
    """Capping the first or last site should stay a valid shorter MPS."""
    m = qtn.MPS_rand_state(5, 3, seed=7)
    dense = m.to_dense().reshape([2] * 5)

    first = py.MpsOptimizer(m.copy(), [("cap", 0, [1.0, 0.0])], chi=8, mode="svd")
    first.run(progbar=False)
    assert first.p.L == 4
    assert np.allclose(
        first.p.to_dense().reshape([2] * 4),
        np.tensordot([1.0, 0.0], dense, axes=([0], [0])),
    )

    last = py.MpsOptimizer(m.copy(), [("cap", 4, [0.0, 1.0])], chi=8, mode="svd")
    last.run(progbar=False)
    assert last.p.L == 4
    assert np.allclose(
        last.p.to_dense().reshape([2] * 4),
        np.tensordot(dense, [0.0, 1.0], axes=([4], [0])),
    )


def test_mps_optimizer_cap_length_one_raises():
    """Capping the single site of a length-1 MPS should fail clearly."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0"),
        [("cap", 0, [1.0, 1.0])],
        chi=2,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="length-1"):
        opt.run(progbar=False)


def test_mps_optimizer_reset_returns_qubit_to_zero():
    """Reset should leave the target qubit in |0> without changing length."""
    hadamard = qu.hadamard()
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000"),
        [(hadamard, (1,)), ("reset", 1)],
        chi=4,
        mode="mpo",
    )
    opt.run(progbar=False, seed=0)

    assert opt.p.L == 3
    assert np.isclose(_dense_pauli_expectation(opt.p, "Z", (1,)), 1.0)
    assert opt.measurements == []


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_mps_optimizer_reset_supports_pauli_bases(axis):
    """Reset should return the target to the +1 eigenstate of X/Y/Z."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0"),
        [(qu.hadamard(), (0,)), ("reset", 0, axis)],
        chi=4,
        mode="mpo",
    )
    opt.run(progbar=False, seed=7)

    assert opt.p.L == 1
    assert np.isclose(_dense_pauli_expectation(opt.p, axis, (0,)), 1.0)
    assert opt.measurements == []


@pytest.mark.parametrize(
    ("axis", "bits", "outcome"),
    [("Z", "1", -1), ("X", "0", -1), ("Y", "0", -1)],
)
def test_mps_optimizer_measure_reset_records_then_resets(axis, bits, outcome):
    """MR should record the measured eigenvalue and leave the + basis state."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state(bits),
        [("measure_reset", axis, 0, outcome)],
        chi=4,
        mode="mpo",
    )
    opt.run(progbar=False)

    assert opt.measurements[0][:3] == (axis, (0,), outcome)
    assert np.isclose(_dense_pauli_expectation(opt.p, axis, (0,)), 1.0)


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "mix", "swap", "perm", "svd", "exact"])
def test_mps_optimizer_control_events_all_modes(mode):
    """measure/cap/reset should work in every run mode."""
    m = qtn.MPS_rand_state(6, 4, seed=2)
    opt = py.MpsOptimizer(
        m.copy(),
        [("measure", "Z", 2, +1), ("reset", 0), ("cap", 4, [1.0, 1.0])],
        chi=8,
        mode=mode,
    )
    if mode == "swap" and not hasattr(opt.p, "gate_with_auto_swap_"):
        pytest.skip("swap mode requires gate_with_auto_swap_ in this quimb version.")
    opt.run(progbar=False, seed=3)

    assert opt.p.L == 5
    assert np.isclose(_dense_pauli_expectation(opt.p, "Z", (2,)), 1.0)
    assert len(opt.measurements) == 1


def test_mps_optimizer_gates_and_control_interleaved():
    """Gates and control events should interleave and stay consistent."""
    hadamard = qu.hadamard()
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000"),
        [
            (hadamard, (0,)),
            (cnot, (0, 1)),
            ("measure", "Z", 0, +1),
            ("cap", 3, [1.0, 1.0]),
        ],
        chi=8,
        mode="mpo",
    )
    opt.run(progbar=False)

    # H then CNOT builds a Bell pair on (0, 1); forcing Z_0 = +1 puts both in |0>.
    assert opt.p.L == 3
    assert np.isclose(_dense_pauli_expectation(opt.p, "Z", (0,)), 1.0)
    assert np.isclose(_dense_pauli_expectation(opt.p, "Z", (1,)), 1.0)
    assert opt.measurements[0][:3] == ("Z", (0,), 1)


def test_mps_optimizer_control_event_seed_is_reproducible():
    """The same seed should reproduce sampled measurement outcomes."""
    hadamard = qu.hadamard()
    stream = [(hadamard, (0,)), ("measure", "Z", 0)]
    first = py.MpsOptimizer(qtn.MPS_computational_state("0"), stream, chi=2, mode="mpo")
    first.run(progbar=False, seed=123)
    second = py.MpsOptimizer(qtn.MPS_computational_state("0"), stream, chi=2, mode="mpo")
    second.run(progbar=False, seed=123)
    assert first.measurements[0][2] == second.measurements[0][2]


def test_mps_optimizer_control_event_mapping_forms():
    """Mapping-form control events should parse into the same queue metadata."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000"),
        [
            {"kind": "measure", "pauli": "Z", "where": 1, "outcome": +1},
            {"kind": "cap", "where": 2, "vec": [1.0, 1.0]},
        ],
        chi=4,
        mode="mpo",
    )
    assert opt.event_types == ["measure", "cap"]
    assert opt.where == [(1,), (2,)]
    opt.run(progbar=False)
    assert opt.p.L == 2
    assert opt.measurements[0][:3] == ("Z", (1,), 1)


def test_mps_optimizer_control_event_public_helpers():
    """Public event builders and detectors should own the control contract."""
    measure = py.MpsOptimizer.measure_event("Z", 2, +1)
    cap = py.MpsOptimizer.cap_event(1, [1, 1], absorb="right")
    reset = py.MpsOptimizer.reset_event([0, 3])
    reset_x = py.MpsOptimizer.reset_event(0, basis="X")
    measure_reset = py.MpsOptimizer.measure_reset_event("Y", 1, -1)

    assert measure == ("measure", "Z", (2,), 1)
    assert cap[0] == "cap" and cap[1] == 1 and cap[3] == "right"
    assert reset == ("reset", (0, 3))
    assert reset_x == ("reset", (0,), "X")
    assert measure_reset == ("measure_reset", "Y", (1,), -1)

    assert py.MpsOptimizer.is_control_event(measure)
    assert py.MpsOptimizer.is_control_event(cap)
    assert py.MpsOptimizer.is_control_event(measure_reset)
    assert py.MpsOptimizer.is_control_event(("mrx", 0, -1))
    assert not py.MpsOptimizer.is_control_event((np.eye(2), (0,)))

    name, payload, where = py.MpsOptimizer.control_event_parts(measure)
    assert name == "measure"
    assert payload["pauli"] == "Z"
    assert payload["outcome"] == 1
    assert where == (2,)


def test_mps_optimizer_measure_reset_support_layout_finder():
    """measure/reset should replay correctly under the layout finder."""
    su4 = qu.rand_uni(4, seed=5)
    hadamard = qu.hadamard()
    stream = [
        (su4, (0, 7)),
        (su4, (1, 6)),
        (hadamard, (3,)),
        ("measure", "Z", 3, +1),
        (su4, (2, 5)),
        ("reset", 0),
        ("measure", "ZZ", (1, 6), +1),
    ]
    init = qtn.MPS_computational_state("0" * 8, dtype="complex128")

    ref = py.MpsOptimizer(init.copy(), list(stream), chi=32, mode="mpo")
    ref.run(progbar=False, seed=7)

    lay = py.MpsOptimizer(init.copy(), list(stream), chi=32, mode="mpo")
    lay.run(progbar=False, seed=7, use_layout_finder=True, layout_report=False)

    inds = [f"k{i}" for i in range(8)]
    assert isinstance(lay.p, qtn.MatrixProductState)
    assert lay.p.site_inds == tuple(inds)
    # Recorded sites use logical labels, not layout-order labels.
    assert [rec[:2] for rec in lay.measurements] == [("Z", (3,)), ("ZZ", (1, 6))]
    assert [rec[:2] for rec in ref.measurements] == [("Z", (3,)), ("ZZ", (1, 6))]
    assert np.allclose(np.abs(lay.p.to_dense(inds)), np.abs(ref.p.to_dense(inds)))


def test_mps_optimizer_cap_events_reject_layout_finder():
    """cap events change the MPS length, so the layout finder is rejected."""
    su4 = qu.rand_uni(4, seed=1)
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000"),
        [(su4, (0, 3)), ("cap", 1, [1.0, 1.0])],
        chi=8,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="cap control"):
        opt.run(progbar=False, use_layout_finder=True)


def test_mps_optimizer_control_events_track_canonical_center(monkeypatch):
    """Control events move the orthogonality centre explicitly (never a rescan)."""
    calls = {"n": 0}
    original = qtn.MatrixProductState.calc_current_orthog_center

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    su4 = qu.rand_uni(4, seed=5)
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(6, 4, seed=2),
        [
            (su4, (0, 3)),
            ("measure", "Z", 2, +1),
            ("reset", 0),
            ("measure", "ZZ", (1, 4), +1),
            ("cap", 4, [1.0, 1.0]),
        ],
        chi=16,
        mode="mpo",
    )
    # Prime the queued gate segment (which legitimately locates the centre once),
    # then assert no rescans happen while the control events run.
    monkeypatch.setattr(
        qtn.MatrixProductState, "calc_current_orthog_center", counting
    )
    opt.run(progbar=False, seed=1)

    assert calls["n"] == 0
    center = opt.info_c.get("cur_orthog")
    assert isinstance(center, tuple) and len(center) == 2
    assert center not in ("calc", None)
    # The tracked centre is a genuine orthogonality centre of the final MPS.
    canonical = opt.p.copy()
    canonical.canonize(list(center))
    assert np.allclose(
        np.abs(canonical.to_dense()), np.abs(opt.p.to_dense())
    )

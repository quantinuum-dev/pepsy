"""MPS normalization regression tests."""


import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn
import pepsy as py
import pepsy.optimizers.mps.optimizer as mps_optimizer_module


from _mps_test_helpers import (
    _mps_data_norm,
    _non_unitary_entangling_gate,
)


pytestmark = [pytest.mark.core, pytest.mark.mps]


def _tensor_data_norm(mps, site):
    """Return the Frobenius norm of one MPS tensor's data."""
    return float(np.linalg.norm(np.asarray(mps[site].data)))


def _assert_event_sites_locally_normalized(mps, event):
    """Check that every tensor rescaled by an event has local norm one."""
    for site in event["sites"]:
        assert _tensor_data_norm(mps, site) == pytest.approx(1.0)


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


def test_mps_optimizer_manual_normalize_reuses_singleton_center(monkeypatch):
    """Default manual normalization should avoid a full norm and QR sweep."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0, gates=[], chi=8, mode="svd")
    center = opt.info_c["cur_orthog"][0]
    opt.p[center].modify(data=3.0 * opt.p[center].data)

    def fail_full_normalize(*_args, **_kwargs):
        raise AssertionError("open-MPS normalization must use its tracked center")

    def fail_canonize(*_args, **_kwargs):
        raise AssertionError("a singleton center must not be moved")

    monkeypatch.setattr(qtn.MatrixProductState, "normalize", fail_full_normalize)
    monkeypatch.setattr(qtn.MatrixProductState, "canonize", fail_canonize)

    old_norm = opt.normalize()

    assert old_norm == pytest.approx(9.0)
    assert opt.info_c["cur_orthog"] == (center, center)
    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
    assert opt.p.norm() == pytest.approx(3.0)
    assert opt.p.exponent == pytest.approx(np.log10(3.0))


def test_mps_optimizer_manual_normalize_rejects_zero_center_transactionally():
    """An undefined normalization must not alter scale or center metadata."""
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    opt = py.MpsOptimizer(p0, gates=[], chi=8, mode="svd")
    center = opt.info_c["cur_orthog"][0]
    opt.p[center].modify(data=np.zeros_like(opt.p[center].data))
    exponent = opt.p.exponent

    with pytest.raises(FloatingPointError, match="zero or non-finite"):
        opt.normalize()

    assert opt.info_c["cur_orthog"] == (center, center)
    assert opt.p.exponent == exponent


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


def test_mps_optimizer_non_unitary_scale_control_preserves_normalization():
    """Fast non-unitary scale control preserves normalization bookkeeping."""
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
    assert [event["step"] for event in events] == [1, 2, 3]
    assert [event["reason"] for event in events] == ["step", "step", "compression"]
    assert all(event["sites"] == (event["insert"],) for event in events)
    _assert_event_sites_locally_normalized(opt.p, events[-1])


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "swap", "svd"])
def test_mps_optimizer_non_unitary_compression_works_without_diagnostics(mode):
    """Non-unitary compression remains usable without diagnostic flags."""
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
        n_iter=4,
        non_unitary=True,
        normalize_every=True,
    )

    assert opt.p.norm() > 0.0
    assert opt.get_normalizations()


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


def test_mps_optimizer_normalize_every_normalizes_every_step():
    """Enabled normalize_every should scale each replay step at one center."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    gates = [(scale, (0,)), (_non_unitary_entangling_gate(), (0, 1)), (scale, (0,))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    ref = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12, normalize_every=2, non_unitary=True, normalize_final=True)
    ref.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_every=False)

    events = opt.get_normalizations()
    assert opt.p.norm() == pytest.approx(ref.p.norm())
    assert [event["step"] for event in events] == [1, 2, 3]
    assert [event["reason"] for event in events] == ["step", "compression", "step"]
    assert all(event["sites"] == (event["insert"],) for event in events)
    assert opt.p.exponent == pytest.approx(sum(event["log10_scale"] for event in events))
    _assert_event_sites_locally_normalized(opt.p, events[-1])


def test_mps_optimizer_normalize_final_can_be_disabled():
    """Per-step normalization should not depend on normalize_final."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    gates = [(scale, (0,)), (_non_unitary_entangling_gate(), (0, 1)), (scale, (0,))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    ref = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12, normalize_every=2, normalize_final=False, non_unitary=True)
    ref.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_every=False)

    assert opt.p.norm() == pytest.approx(ref.p.norm())
    events = opt.get_normalizations()
    assert [event["step"] for event in events] == [1, 2, 3]
    assert events[1]["reason"] == "compression"
    assert all(event["sites"] == (event["insert"],) for event in events)


def test_mps_optimizer_automatic_normalization_rejects_exact_mode():
    """Exact mode has no MPS canonicalization range for automatic normalization."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    opt = py.MpsOptimizer(p0.copy(), gates=[(scale, (0,))], chi=8, mode="exact")

    with pytest.raises(ValueError, match="not available in exact mode"):
        opt.run(progbar=False, non_unitary=True, normalize_every=True)
    with pytest.raises(ValueError, match="not available in exact mode"):
        opt.run(progbar=False, non_unitary=True, normalize_final=True)


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


def test_mps_optimizer_canonical_modes_reject_cyclic_mps():
    """A periodic MPS has no exact one-tensor mixed-canonical norm."""
    cyclic = qtn.MPS_rand_state(
        4,
        bond_dim=2,
        phys_dim=2,
        cyclic=True,
        dtype="complex128",
        seed=92,
    )

    with pytest.raises(ValueError, match="open-boundary MPS"):
        py.MpsOptimizer(cyclic, gates=[], chi=4, mode="mpo")


def test_mps_optimizer_cyclic_rejection_is_transactional():
    """Rejected replacement and mode switches must preserve optimizer state."""
    cyclic = qtn.MPS_rand_state(
        4,
        bond_dim=2,
        phys_dim=2,
        cyclic=True,
        dtype="complex128",
        seed=93,
    )
    open_opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[],
        chi=4,
        mode="mpo",
        inplace=True,
    )
    original_state = open_opt.p
    original_info = dict(open_opt.info_c)
    assert not getattr(cyclic, "_pepsy_norm_includes_exponent", False)

    with pytest.raises(ValueError, match="open-boundary MPS"):
        open_opt.set_p(cyclic)

    assert open_opt.p is original_state
    assert open_opt.info_c == original_info
    assert not getattr(cyclic, "_pepsy_norm_includes_exponent", False)

    exact_opt = py.MpsOptimizer(
        cyclic,
        gates=[],
        chi=4,
        mode="exact",
    )
    with pytest.raises(ValueError, match="open-boundary MPS"):
        exact_opt.set_mode("svd")

    assert exact_opt.mode == "exact"
    assert exact_opt.info_c == {}


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
    assert event["insert"] == 2
    assert event["sites"] == (2,)
    assert opt.info_c["cur_orthog"] == (2, 2)


def test_mps_optimizer_local_normalization_keeps_singleton_center(monkeypatch):
    """Scale control should reuse FIT's endpoint instead of sweeping the span."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(
            4,
            bond_dim=2,
            phys_dim=2,
            dtype="complex128",
            seed=91,
        ),
        gates=[],
        chi=8,
        mode="dmrg2",
    )
    opt.canonize_mps(opt.p, 0)
    opt.p[0].modify(data=2.0 * opt.p[0].data)

    def fail_canonize(*_args, **_kwargs):
        raise AssertionError("normalization moved an authoritative FIT center")

    monkeypatch.setattr(qtn.MatrixProductState, "canonize", fail_canonize)
    event = opt._normalize_orthog_tensors(
        opt.p,
        (0, 3),
        step=1,
        reason="test",
        canonicalize=False,
    )

    assert event["span"] == (0, 3)
    assert event["insert"] == 0
    assert event["sites"] == (0,)
    assert opt.info_c["cur_orthog"] == (0, 0)
    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
    assert opt.p.norm() == pytest.approx(2.0)


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


def test_mps_optimizer_represented_norm_capability_check_is_cached(monkeypatch):
    """Repeated FIT results should not re-contract the full MPS norm."""
    p0 = qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128", seed=13)
    cache = mps_optimizer_module._NORM_INCLUDES_EXPONENT_CACHE  # pylint: disable=protected-access
    cache.pop(type(p0), None)
    original = py.MpsOptimizer._class_norm_includes_exponent
    calls = []

    def count_checks(state):
        calls.append(state)
        return original(state)

    monkeypatch.setattr(
        py.MpsOptimizer,
        "_class_norm_includes_exponent",
        staticmethod(count_checks),
    )
    py.MpsOptimizer._install_represented_norm(p0.copy())  # pylint: disable=protected-access
    py.MpsOptimizer._install_represented_norm(p0.copy())  # pylint: disable=protected-access

    assert len(calls) == 1


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
            # This legacy accuracy comparison intentionally requests the old
            # fixed-sweep behavior; the public default is adaptive FIT.
            fit_rtol=None,
            non_unitary=True,
            normalize_every=1,
        )

        fidelity = float(np.real(py.tn_fidelity(opt.p, target, contraction_opt="auto-hq")))
        results[mode] = {
            "fidelity": fidelity,
            "represented_norm": float(np.real(opt.p.norm())),
        }

        events = opt.get_normalizations()
        assert len(events) == len(gates)
        assert [event["step"] for event in events] == list(range(1, len(gates) + 1))
        assert [event["reason"] for event in events] == [
            "step",
            "step",
            "step",
            "step",
            "compression",
            "compression",
            "compression",
        ]
        assert all(event["sites"] == (event["insert"],) for event in events)
        _assert_event_sites_locally_normalized(opt.p, events[-1])
        assert fidelity > 0.92

    assert results["dmrg"]["fidelity"] == pytest.approx(
        results["mpo"]["fidelity"],
        abs=5e-10,
    )
    assert results["dmrg"]["represented_norm"] == pytest.approx(
        results["mpo"]["represented_norm"],
        abs=5e-10,
    )


def test_mps_optimizer_set_p_rebases_unitary_stabilization_norm():
    """A replacement state must not inherit the prior raw-norm baseline."""
    identity = np.eye(4, dtype=np.complex128)
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        [(identity, (0, 1))],
        chi=2,
        mode="dmrg2",
    )
    optimizer.run(progbar=False, n_iter=1)

    replacement = qtn.MPS_computational_state("00", dtype="complex128")
    replacement[0].modify(data=3.0 * replacement[0].data)
    optimizer.set_p(replacement)

    assert optimizer._unitary_previous_norm is None  # pylint: disable=protected-access
    optimizer.run(progbar=False, n_iter=1)

    assert _mps_data_norm(optimizer.p) == pytest.approx(3.0)


def test_mps_optimizer_normalize_rebases_raw_unitary_stabilization_norm():
    """Manual normalization preserves represented scale without restoring it twice."""
    state = qtn.MPS_computational_state("00", dtype="complex128")
    state[0].modify(data=3.0 * state[0].data)
    optimizer = py.MpsOptimizer(
        state,
        [(np.eye(4, dtype=np.complex128), (0, 1))],
        chi=2,
        mode="dmrg2",
    )
    optimizer.run(progbar=False, n_iter=1)

    optimizer.normalize(insert=0)
    optimizer.run(progbar=False, n_iter=1)

    assert _mps_data_norm(optimizer.p) == pytest.approx(1.0)
    assert optimizer.p.norm() == pytest.approx(3.0)


@pytest.mark.parametrize("mode", ["mpo", "swap", "perm", "svd"])
def test_standalone_compression_modes_honor_unitary_stabilization(mode):
    """Every compressed mode can restore norm after a lossy unitary split."""
    gates = [
        (qu.hadamard(dtype="complex64"), (0,)),
        (qu.CNOT(dtype="complex64"), (0, 2)),
    ]
    stabilized = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex64"),
        gates,
        chi=1,
        mode=mode,
    )
    stabilized.run(
        progbar=False,
        cutoff=0.0,
        stabilize_unitary=True,
        timing=True,
    )
    unstabilized = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex64"),
        gates,
        chi=1,
        mode=mode,
    )
    unstabilized.run(
        progbar=False,
        cutoff=0.0,
        stabilize_unitary=False,
    )

    assert _mps_data_norm(stabilized.p) == pytest.approx(1.0, abs=2.0e-5)
    assert _mps_data_norm(unstabilized.p) < 0.999
    assert stabilized.norm_diagnostics()["infidelity"] == pytest.approx(
        0.5, abs=2.0e-5
    )
    assert unstabilized.norm_diagnostics()["infidelity"] == pytest.approx(
        0.5, abs=2.0e-5
    )
    assert stabilized.get_norm_events()[0]["kind"] == "unitary_compression"
    timing_name = "direct" if mode == "mpo" else mode
    assert stabilized.get_run_timing()["stages"][f"{timing_name}.stabilize"]["calls"] == 1


@pytest.mark.parametrize("mode", ["dmrg1", "dmrg2", "dmrg3"])
def test_dmrg_schedules_record_automatic_norm_survival(mode):
    """All named DMRG schedules use the same automatic norm ledger."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        [(qu.hadamard(), (0,)), (qu.CNOT(), (0, 2))],
        chi=1,
        mode=mode,
    )
    opt.run(progbar=False, cutoff=0.0, n_iter=3, stabilize_unitary=True)

    diagnostics = opt.norm_diagnostics()
    assert diagnostics["events"] == 1
    assert diagnostics["infidelity"] == pytest.approx(0.5, abs=2.0e-5)
    assert _mps_data_norm(opt.p) == pytest.approx(1.0, abs=2.0e-5)


def test_mps_norm_names_and_dmrg_target_overlap_are_distinct():
    """DMRG exposes target overlap separately from retained norm fidelity."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        [(qu.hadamard(), (0,)), (qu.CNOT(), (0, 2))],
        chi=1,
        mode="dmrg",
    )
    opt.run(
        progbar=False,
        cutoff=0.0,
        n_iter=3,
        stabilize_unitary=True,
        fit_overlap_diagnostics=True,
    )

    diagnostics = opt.norm_diagnostics()
    fit = opt.get_fit_diagnostics()
    removed_metric_names = {
        "norm_fidelity_raw",
        "norm_fidelity",
        "norm_infidelity",
        "local_norm_fidelity",
        "local_norm_infidelity",
        "cumulative_norm_fidelity",
        "cumulative_norm_infidelity",
    }
    assert removed_metric_names.isdisjoint(diagnostics)
    assert removed_metric_names.isdisjoint(opt.get_norm_events()[0])
    assert diagnostics["local_fidelity"] == pytest.approx(0.5, abs=2e-5)
    assert diagnostics["cumulative_fidelity"] == pytest.approx(0.5, abs=2e-5)
    assert diagnostics["norm"] == pytest.approx(1.0, abs=2e-5)
    assert diagnostics["state_norm"] == pytest.approx(1.0, abs=2e-5)
    assert diagnostics["cumulative_norm"] == pytest.approx(
        np.sqrt(0.5), abs=2e-5,
    )
    assert fit["fit_overlap_fidelity"] == pytest.approx(0.5, abs=2e-5)
    assert fit["fit_overlap_infidelity"] == pytest.approx(0.5, abs=2e-5)

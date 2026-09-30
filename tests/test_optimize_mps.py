"""MPS replay regression tests."""


import types
import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn
import pepsy as py
import pepsy.optimizers.mps.optimizer as mps_optimizer_module


from _mps_test_helpers import (
    _mps_data_norm,
    _non_unitary_entangling_gate,
    _two_branch_flip_submpo,
)


def _perm_mps_to_logical_dense(opt):
    """Return a permuted-mode MPS dense state in logical site order."""
    physical = opt.p.to_dense().reshape((2,) * opt.p.L)
    logical_axes = [opt.qubits.index(site) for site in range(opt.p.L)]
    return np.transpose(physical, logical_axes).reshape(-1)


def test_mps_optimizer_effective_length_is_gate_active_and_cap_reduced():
    """L_eff is a lightweight stream-support ledger, not a rank probe."""
    product = qtn.MPS_computational_state("000", dtype="complex128")

    empty = py.MpsOptimizer(product, gates=[], chi=8, mode="mpo")
    assert empty.L_eff == 0
    empty.run(cutoff=0.0, progbar=False)
    assert empty.L_eff == 0

    one_site = py.MpsOptimizer(
        product,
        gates=[(qu.hadamard(), (1,))],
        chi=8,
        mode="mpo",
    )
    one_site.run(cutoff=0.0, progbar=False)
    assert one_site.L_eff == 1
    assert one_site.mps_length_diagnostics()["L_eff_history"] == (0, 1)

    nonlocal_gate = py.MpsOptimizer(
        product,
        gates=[(qu.CNOT(), (0, 2))],
        chi=8,
        mode="mpo",
    )
    nonlocal_gate.run(cutoff=0.0, progbar=False)
    assert nonlocal_gate.L_eff == 3

    capped = py.MpsOptimizer(
        product,
        gates=[
            (qu.hadamard(), (1,)),
            ("cap", 1, np.array([1.0, 1.0]), "left"),
        ],
        chi=8,
        mode="mpo",
    )
    capped.run(cutoff=0.0, progbar=False)
    assert capped.L_eff == 0
    assert capped.mps_length_diagnostics()["L_eff_history"] == (0, 1, 0)


def test_mps_optimizer_has_automatic_norm_tracking_without_legacy_api():
    """Norm-survival diagnostics are automatic, without an opt-in flag."""
    state = qtn.MPS_computational_state("00", dtype="complex128")
    opt = py.MpsOptimizer(state, gates=[], chi=2, mode="svd")

    assert callable(opt.norm_diagnostics)
    assert opt.norm_diagnostics()["tracking"] is True
    assert not hasattr(opt, "get_fidelities")
    assert not hasattr(opt, "get_true_infidelities")
    assert not hasattr(opt, "get_norm_infidelity_samples")
    assert not hasattr(opt, "track_infidelity")
    assert not hasattr(opt, "get_infidelities")
    assert not hasattr(opt, "get_infidelity_samples")
    assert not hasattr(opt, "reset_infidelity_tracking")
    with pytest.raises(TypeError, match="track_infidelity"):
        py.MpsOptimizer(state, gates=[], chi=2, mode="svd", track_infidelity=False)
    with pytest.raises(TypeError, match="fidelity_samples"):
        opt.run(progbar=False, fidelity_samples=0)


def test_mps_optimizer_run_rejects_removed_infidelity_option():
    """The removed keyword is rejected instead of silently doing extra work."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="svd",
    )
    with pytest.raises(TypeError, match="track_infidelity"):
        opt.run(progbar=False, track_infidelity=False)


def test_mps_optimizer_accepts_svd_mode():
    """SVD mode should be accepted by ``MpsOptimizer`` mode validation."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0, gates=[], chi=8, mode="svd")
    assert opt.mode == "svd"


@pytest.mark.parametrize(
    "mode", ["dmrg", "mpo", "svd", "swap", "perm", "mix", "exact"]
)
def test_mps_optimizer_rejects_mismatched_gate_stream_backend(mode):
    """Mismatched user gates must be prepared before optimizer construction."""
    torch = pytest.importorskip("torch")

    state = qtn.MPS_computational_state("00", dtype="complex128")
    state.apply_to_arrays(py.backend_torch(dtype=torch.complex128, device="cpu"))
    gate = qu.CNOT()  # ordinary NumPy gate stream
    with pytest.raises(TypeError, match="requires every gate"):
        py.MpsOptimizer(
            state,
            gates=[(gate, (0, 1))],
            chi=2,
            mode=mode,
            inplace=True,
        )


def test_mps_optimizer_rejects_mismatched_submpo_stream_backend():
    """Mismatched sub-MPO tensors must be prepared before replay."""
    torch = pytest.importorskip("torch")

    state = qtn.MPS_computational_state("00", dtype="complex128")
    state.apply_to_arrays(py.backend_torch(dtype=torch.complex128, device="cpu"))
    submpo = _two_branch_flip_submpo(L=2, sites=(0, 1), targets=(0, 1))
    optimizer = py.MpsOptimizer(
        state,
        gates=[],
        chi=2,
        mode="mpo",
        inplace=True,
    )

    with pytest.raises(TypeError, match="sub-MPO"):
        optimizer.set_gates([py.MpsOptimizer.submpo_event(submpo, (0, 1))])
    assert all(isinstance(tensor.data, np.ndarray) for tensor in submpo.tensors)
    assert all(isinstance(tensor.data, torch.Tensor) for tensor in optimizer.p.tensors)


def test_mps_optimizer_backend_diagnostics_check_every_gate():
    """Backend checks inspect every gate, not only the first stream payload."""
    torch = pytest.importorskip("torch")

    state = qtn.MPS_computational_state("00", dtype="complex128")
    to_backend = py.backend_torch(dtype=torch.complex128, device="cpu")
    state.apply_to_arrays(to_backend)
    matching = to_backend(np.eye(2, dtype=complex))
    foreign = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    optimizer = py.MpsOptimizer(
        state,
        gates=[],
        chi=2,
        mode="mpo",
        inplace=True,
    )

    assert optimizer.backend_info() == {
        "backend": "torch",
        "dtype": "complex128",
        "device": "cpu",
    }
    assert optimizer.backend == "torch"
    with pytest.raises(TypeError, match=r"stream\[1\]"):
        optimizer._validate_gate_stream_backend(
            [matching, foreign], ["gate", "gate"]
        )


def test_mps_optimizer_accepts_explicitly_prepared_gate():
    """Callers can prepare a payload explicitly before installing it."""
    torch = pytest.importorskip("torch")

    state = qtn.MPS_computational_state("00", dtype="complex128")
    state.apply_to_arrays(py.backend_torch(dtype=torch.complex128, device="cpu"))
    optimizer = py.MpsOptimizer(state, gates=[], chi=2, mode="svd")

    gate = optimizer.to_backend(qu.CNOT())
    optimizer.set_gates([(gate, (0, 1))])
    optimizer.run(progbar=False)

    assert all(isinstance(tensor.data, torch.Tensor) for tensor in optimizer.p.tensors)


def test_mps_optimizer_rejects_mixed_state_backends():
    """All live MPS tensors must agree on backend, dtype, and device."""
    torch = pytest.importorskip("torch")

    state = qtn.MPS_computational_state("00", dtype="complex128")
    state[0].modify(data=torch.as_tensor(state[0].data, dtype=torch.complex128))
    with pytest.raises(TypeError, match="one compatible backend"):
        py.MpsOptimizer(state, gates=[], chi=2, mode="mpo")


def test_mps_optimizer_reports_symmray_block_backend():
    """Symmray diagnostics retain the underlying Torch block backend."""
    pytest.importorskip("symmray")
    torch = pytest.importorskip("torch")

    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    state = py.ps_to_mps(
        3,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0)),
        seed=1,
        dtype="complex128",
    )
    state.apply_to_arrays(py.backend_torch(dtype=torch.complex128, device="cpu"))
    optimizer = py.MpsOptimizer(state, gates=[], chi=2, mode="mpo")

    assert optimizer.backend_info() == {
        "backend": "symmray",
        "dtype": "complex128",
        "device": "cpu",
        "array_backend": "torch",
    }


def test_mps_optimizer_rejects_dense_symmray_submpo_blocks():
    """Dense sub-MPO blocks cannot be promoted into a native Symmray MPS."""
    pytest.importorskip("symmray")
    torch = pytest.importorskip("torch")

    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    state = py.ps_to_mps(
        3,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0)),
        seed=1,
        dtype="complex128",
    )
    state.apply_to_arrays(py.backend_torch(dtype=torch.complex128, device="cpu"))
    submpo = qtn.MatrixProductOperator.from_dense(
        fermion.hopping_gate(0.01, t=1.0),
        dims=(4, 4),
        sites=(0, 1),
        L=3,
    )
    optimizer = py.MpsOptimizer(state, gates=[], chi=2, mode="mpo")

    with pytest.raises(TypeError, match="sub-MPO"):
        optimizer._validate_gate_stream_backend([submpo], ["submpo"])
    assert all(tensor.data.backend == "numpy" for tensor in submpo.tensors)


def test_mps_optimizer_accepts_perm_mode():
    """Perm mode should expose an identity logical-to-physical ordering initially."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0, gates=[], chi=8, mode="perm")

    assert opt.mode == "perm"
    assert opt.qubits == [0, 1, 2, 3]


def test_mps_optimizer_rejects_removed_simple_update_mode():
    """Simple-update mode is no longer part of MpsOptimizer."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")

    with pytest.raises(ValueError, match="Unknown mode: su"):
        py.MpsOptimizer(p0, gates=[], chi=2, mode="su")

    optimizer = py.MpsOptimizer(p0, gates=[], chi=2, mode="direct")
    with pytest.raises(ValueError, match="Unknown mode: su"):
        optimizer.set_mode("su")


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

    perm.run(progbar=False, cutoff=1e-12)
    reference.run(progbar=False, cutoff=1e-12)

    assert perm.qubits == [0, 2, 3, 1]
    assert perm.logical_order == perm.qubits
    assert np.allclose(_perm_mps_to_logical_dense(perm), reference.p.to_dense().reshape(-1))

    perm.restore_qubit_order()
    assert perm.qubits == [0, 1, 2, 3]
    assert perm.logical_order == perm.qubits
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

    opt.run(progbar=False)

    assert opt.qubits == [0, 3, 1, 2]
    assert opt.measurements[0][1] == (3,)
    assert opt.measurements[0][2] == 1


def test_mps_optimizer_perm_conditional_gate_maps_logical_site_once():
    """A feed-forward gate after a lazy swap must not be mapped twice."""
    flip = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000"),
        [
            (qu.hadamard(), 0),
            (qu.CNOT(), (0, 3)),
            ("measure", "Z", 0, -1),
            ("if", -1, 1, (flip, 2)),
        ],
        chi=8,
        mode="perm",
    )

    opt.run(progbar=False, cutoff=0.0)

    expected = np.zeros(16, dtype=complex)
    expected[11] = 1.0  # logical |1011>
    np.testing.assert_allclose(opt.to_dense().reshape(-1), expected, atol=1e-12)


def test_mps_optimizer_perm_is_single_svd_swap_path():
    """Perm mode should retain Quimb's no-swap-back local SVD semantics."""
    p0 = qtn.MPS_computational_state("00000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (3, 0)),
        (qu.CNOT(), (1, 4)),
        (qu.hadamard(), (2,)),
        (qu.CNOT(), (4, 1)),
    ]
    reference = py.MpsOptimizer(p0.copy(), gates=gates, chi=32, mode="exact")
    reference.run(progbar=False)
    opt = py.MpsOptimizer(
        p0,
        gates=gates,
        chi=32,
        mode="perm",
    )
    opt.run(progbar=False, cutoff=0.0)

    np.testing.assert_allclose(
        np.asarray(opt.to_dense()).reshape(-1),
        np.asarray(reference.to_dense()).reshape(-1),
        atol=1e-10,
    )
    assert opt.qubits != list(range(5))


@pytest.mark.parametrize("mode", ["perm-src", "perm-dmrg2", "dmrg2-perm"])
def test_mps_optimizer_rejects_composed_perm_mode_aliases(mode):
    """Permutation mode is not a routing prefix/suffix for another solver."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")

    with pytest.raises(ValueError, match=f"Unknown mode: {mode}"):
        py.MpsOptimizer(p0, gates=[], chi=8, mode=mode)


def test_mps_optimizer_perm_rejects_routing_keyword_and_layout():
    """Perm mode owns routing and cannot be combined with another layout."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0, gates=[], chi=8, mode="perm")

    with pytest.raises(ValueError, match="persistent layouts cannot be combined"):
        opt.apply_layout((0, 2, 3, 1), layout_report=False)
    with pytest.raises(TypeError, match="unexpected keyword argument 'routing'"):
        py.MpsOptimizer(p0, gates=[], chi=8, mode="perm", routing="perm")


def test_mps_optimizer_exact_to_perm_mode_switch_rebuilds_mps():
    """Switching a contracted exact state must seed perm bookkeeping."""
    gates = [("h", 0), ("cnot", 0, 3), ("h", 1)]
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000"), gates, chi=16, mode="exact"
    )
    opt.run(progbar=False)
    opt.set_mode("perm")

    assert opt.qubits == list(range(4))
    opt.run(progbar=False, cutoff=0.0, fit_init_strategy="guess-direct")

    reference = py.MpsOptimizer(
        qtn.MPS_computational_state("0000"), gates + gates, chi=16, mode="exact"
    )
    reference.run(progbar=False)
    np.testing.assert_allclose(
        np.asarray(opt.to_dense()).reshape(-1),
        np.asarray(reference.to_dense()).reshape(-1),
        atol=1e-10,
    )


def test_mps_optimizer_perm_stabilization_includes_route_compression():
    """Perm route truncation belongs to its unitary compression step."""
    vector = np.zeros(32, dtype=complex)
    vector[0] = vector[18] = 1.0 / np.sqrt(2.0)
    opt = py.MpsOptimizer(
        qtn.MatrixProductState.from_dense(vector, [2] * 5),
        gates=[("cnot", 0, 4)],
        chi=1,
        mode="perm",
    )

    opt.run(progbar=False, cutoff=1.0e-12, stabilize_unitary=True)

    event = opt.get_norm_events()[0]
    assert event["kind"] == "unitary_compression"
    assert event["observed_norm"] < 0.9 * event["expected_norm"]
    assert opt.p.norm() == pytest.approx(1.0)


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


def test_mps_optimizer_opt_in_run_timing_reports_replay_metrics():
    """Timing is opt-in and returns a copy-safe replay record."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="direct",
    )

    assert opt.get_run_timing() is None
    opt.run(progbar=False, timing=True)

    timing = opt.get_run_timing()
    assert timing["status"] == "complete"
    assert timing["mode"] == "quimb-direct"
    assert timing["mode_alias"] is None
    assert timing["event_count"] == 1
    assert timing["elapsed_seconds"] >= 0.0
    assert timing["final_bond"] <= 2
    assert timing["stages"]["direct.replay"]["calls"] == 1
    assert timing["stages"]["canonicalize"]["calls"] >= 1
    assert not any(name.startswith("infidelity.") for name in timing["stages"])
    timing["mode"] = "changed"
    assert opt.last_run_timing["mode"] == "quimb-direct"


def test_mps_optimizer_diagnostic_accessors_are_copy_safe():
    """Public diagnostic snapshots cannot mutate optimizer-owned state."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    opt = py.MpsOptimizer(
        p0,
        gates=[(qu.hadamard(), (0,)), (qu.CNOT(), (0, 1)), (scale, (0,))],
        chi=1,
        mode="dmrg",
    )

    opt.run(
        progbar=False,
        n_iter=2,
        fit_rtol=None,
        non_unitary=True,
        normalize_every=True,
        quality_check_every=True,
    )

    quality_checks = opt.get_quality_checks()
    normalizations = opt.get_normalizations()
    fit_diagnostics = opt.get_fit_diagnostics()

    assert quality_checks
    assert normalizations
    assert fit_diagnostics is not None

    quality_checks[0]["step"] = -1
    normalizations[0]["step"] = -1
    fit_diagnostics["iterations"] = -1

    assert opt.quality_checks[0]["step"] != -1
    assert opt.normalizations[0]["step"] != -1
    assert opt._last_dmrg_fit_diagnostics["iterations"] != -1


def test_mps_optimizer_non_unitary_rejects_unitary_stabilization():
    """Unitary working-norm restoration is invalid for non-unitary replay."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="direct",
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        opt.run(
            progbar=False,
            non_unitary=True,
            stabilize_unitary=True,
        )


def test_mps_optimizer_non_unitary_dmrg_keeps_adaptive_fit_default():
    """Non-unitary target scale should not disable adaptive DMRG stopping."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(_non_unitary_entangling_gate(), (0, 1))],
        chi=2,
        mode="dmrg",
    )
    captured = {}

    def capture_execute(*args, **kwargs):
        captured.update(kwargs)
        return opt.p

    opt._execute_mode = capture_execute
    opt.run(progbar=False, non_unitary=True, n_iter=2)

    assert captured["fit_rtol"] == pytest.approx(
        opt._resolve_fit_rtol("auto")
    )
    assert captured["fit_rtol"] is not None


def test_mps_optimizer_fit_diagnostics_is_none_before_fit():
    """The public FIT diagnostic accessor is explicit before any FIT run."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="mpo",
    )

    assert opt.get_fit_diagnostics() is None


def test_mps_optimizer_fit_overlap_diagnostic_failure_is_nonfatal(monkeypatch):
    """Optional FIT overlap failures stay diagnostics, not replay failures."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="dmrg",
    )

    def fail(*args, **kwargs):
        raise RuntimeError("diagnostic unavailable")

    monkeypatch.setattr(mps_optimizer_module, "tn_fidelity", fail)
    result = opt._fit_overlap_diagnostics(opt.p, opt.p)

    assert result["fit_overlap_fidelity"] is None
    assert result["fit_overlap_infidelity"] is None
    assert "diagnostic unavailable" in result["fit_overlap_error"]


@pytest.mark.parametrize("overlap", [float("nan"), float("inf"), -float("inf")])
def test_mps_optimizer_fit_overlap_nonfinite_is_reported(monkeypatch, overlap):
    """Non-finite optional overlaps must not become valid clipped fidelities."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="dmrg",
    )
    monkeypatch.setattr(mps_optimizer_module, "tn_fidelity", lambda *a, **k: overlap)

    result = opt._fit_overlap_diagnostics(opt.p, opt.p)

    assert result["fit_overlap_fidelity"] is None
    assert result["fit_overlap_infidelity"] is None
    assert "non-finite" in result["fit_overlap_error"]


@pytest.mark.parametrize("mode", ["dmrg", "dmrg1", "dmrg2", "dmrg3", "mix"])
@pytest.mark.parametrize("timing", [False, True])
def test_mps_optimizer_fit_overlap_diagnostics_are_opt_in(monkeypatch, mode, timing):
    """The expensive FIT-target overlap contraction is disabled by default."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(8, 2, dtype="complex128", seed=714),
        gates=[(qu.CNOT(), (2, 4))],
        chi=2,
        mode=mode,
    )

    def fail(*args, **kwargs):
        raise AssertionError("FIT overlap diagnostics should be opt-in")

    monkeypatch.setattr(mps_optimizer_module, "tn_fidelity", fail)
    optimizer.run(progbar=False, n_iter=2, fit_rtol=None, timing=timing)

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["fit_overlap_diagnostics"] is False
    assert diagnostics["fit_overlap_fidelity"] is None
    assert diagnostics["fit_overlap_infidelity"] is None
    assert diagnostics["fit_overlap_error"] is None


@pytest.mark.parametrize("mode", ["mpo", "swap", "svd", "dmrg", "mix"])
def test_mps_optimizer_modes_skip_clocks_when_timing_disabled(
    monkeypatch,
    mode,
):
    """Untimed replay must not touch the profiling clock in any mode."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=2,
        mode=mode,
    )

    def fail_clock():
        raise AssertionError("timing=False must not read the profiling clock")

    monkeypatch.setattr(
        mps_optimizer_module,
        "time",
        types.SimpleNamespace(perf_counter=fail_clock),
    )

    def fail_synchronizer(*_args, **_kwargs):
        raise AssertionError(
            "timing=False must not construct a device synchronizer"
        )

    monkeypatch.setattr(
        mps_optimizer_module.FIT,
        "_make_backend_synchronizer",
        fail_synchronizer,
    )
    monkeypatch.setattr(optimizer, "_mps_data_is_finite", fail_synchronizer)
    monkeypatch.setattr(optimizer, "_fit_overlap_diagnostics", fail_synchronizer)
    monkeypatch.setattr(optimizer, "_run_quality_check", fail_synchronizer)
    # Omit timing/diagnostic flags to guard their public defaults. Even an
    # explicit sync request must remain inert while timing is disabled.
    optimizer.run(progbar=False, timing_sync_device=True)

    assert optimizer.get_run_timing() is None


def test_mps_optimizer_timing_record_identifies_named_dmrg_mode():
    """Timed DMRG aliases retain their schedule identity and fit summary."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=213),
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(progbar=False, n_iter=2, fit_rtol=None, timing=True)

    timing = optimizer.get_run_timing()
    assert timing["mode"] == "dmrg"
    assert timing["mode_alias"] == "dmrg2"
    assert timing["fit_diagnostics"] == optimizer.get_fit_diagnostics()
    assert timing["fit_diagnostics"]["block_size"] == 2


def test_mps_optimizer_dmrg_uses_gate_window_fit(monkeypatch):
    """DMRG keeps FIT restricted to the gate window, not the full MPS."""
    called_ranges = []
    original_run_gate = mps_optimizer_module.FIT.run_gate

    def record_run_gate(self, *args, **kwargs):
        called_ranges.append(tuple(self.range_int))
        return original_run_gate(self, *args, **kwargs)

    def fail_full_chain_fit(*args, **kwargs):
        raise AssertionError("MpsOptimizer DMRG must not call FIT.run_eff")

    monkeypatch.setattr(mps_optimizer_module.FIT, "run_gate", record_run_gate)
    monkeypatch.setattr(mps_optimizer_module.FIT, "run_eff", fail_full_chain_fit)

    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(5, bond_dim=2, phys_dim=2, dtype="complex128", seed=32),
        gates=[(qu.CNOT(), (1, 3))],
        chi=2,
        mode="dmrg",
    )
    opt.run(progbar=False, n_iter=2)

    assert called_ranges == [(1, 3)]


def test_mps_optimizer_mix_timing_includes_mix_summary():
    """Mixed timing records expose the existing backend decision summary."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[(qu.hadamard(), (0,)), (qu.CNOT(), (0, 1))],
        chi=2,
        mode="mix",
    )

    opt.run(progbar=False, timing=True)

    timing = opt.get_run_timing()
    assert timing["mix_summary"] == opt.last_mix_summary
    assert timing["mix_summary"]["elapsed_seconds"] >= 0.0


def test_mps_optimizer_mix_uses_dmrg_during_growth_and_fixed_chi(monkeypatch):
    """Mixed multi-site gates use one-site FIT in both rank phases."""
    guess_start_bonds = []
    original_guess = py.MpsOptimizer._build_compression_fit_guess

    def record_direct_guess(self, p, *args, **kwargs):
        guess_start_bonds.append(int(p.max_bond()))
        return original_guess(self, p, *args, **kwargs)

    monkeypatch.setattr(
        py.MpsOptimizer,
        "_build_compression_fit_guess",
        record_direct_guess,
    )
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 1)),
        (qu.CNOT(), (1, 2)),
        (qu.CNOT(), (1, 2)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="mix")
    out = opt.run(
        progbar=False,
        cutoff=1e-12,
        n_iter=3,
    )

    assert out.max_bond() <= 2
    assert [event["backend"] for event in opt.mix_history] == [
        "mpo",
        "dmrg",
        "dmrg",
        "dmrg",
    ]
    assert [event["reason"] for event in opt.mix_history[1:]] == [
        "guess_direct_dmrg1",
        "guess_direct_dmrg1",
        "guess_direct_dmrg1",
    ]
    assert opt.mix_history[1]["start_bond"] == 1
    assert opt.mix_history[1]["end_bond"] == 2
    assert opt.mix_history[-1]["start_bond"] == 2
    assert guess_start_bonds == [1, 2, 2]
    assert opt.last_mix_summary["mpo_steps"] == 1
    assert opt.last_mix_summary["dmrg_steps"] == 3
    assert opt.last_mix_summary["fallback_steps"] == 0


def test_mps_optimizer_mix_defaults_to_guess_direct_dmrg1():
    """Mixed multi-site gates use the fixed guess-direct/DMRG1 contract."""
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 1)),
        (qu.CNOT(), (1, 2)),
        (qu.CNOT(), (0, 2)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="mix")
    out = opt.run(
        progbar=False,
        cutoff=1e-12,
        n_iter=3,
        fit_rtol=None,
        timing=True,
    )

    assert out.max_bond() <= 2
    assert [event["backend"] for event in opt.mix_history] == [
        "mpo",
        "dmrg",
        "dmrg",
        "dmrg",
    ]
    fit_steps = opt.get_run_timing()["fit_steps"]
    assert [record["block_size"] for record in fit_steps] == [1] * 9
    diagnostics = opt.get_fit_diagnostics()
    assert diagnostics["block_size"] == 1
    assert diagnostics["one_site_refinement_sweeps"] == 3
    assert diagnostics["fit_init_strategy_requested"] == "guess_direct"
    assert diagnostics["fit_init_strategy"] == "guess_direct"
    assert diagnostics["guess_method"] == "direct"
    assert diagnostics["guess_used"] is True
    assert opt.last_mix_summary["mpo_steps"] == 1
    assert opt.last_mix_summary["dmrg_steps"] == 3


def test_mps_optimizer_mix_one_site_fast_path_keeps_input_identity():
    """Mixed mode should apply one-site gates without a DMRG trial copy."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=17)
    opt = py.MpsOptimizer(
        p0,
        gates=[(qu.hadamard(), (1,))],
        chi=2,
        mode="mix",
        inplace=True,
    )

    out = opt.run(progbar=False, cutoff=1e-12)

    assert opt.mix_history[0]["backend"] == "mpo"
    assert opt.mix_history[0]["reason"] == "one_site_exact"
    assert opt.p is p0
    assert out is p0


def test_mps_optimizer_mix_targets_attainable_edge_bonds():
    """Mixed diagnostics should cap edge targets by their physical ranks."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[],
        chi=8,
        mode="mix",
    )

    assert opt._mix_target_bond_dimensions() == [2, 4, 2]


def test_mps_optimizer_mix_guess_direct_opens_short_active_bonds():
    """The disposable direct guess, not live pre-padding, grows short bonds."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 1)),
        (qu.CNOT(), (1, 2)),
        (qu.CNOT(), (2, 3)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="mix")
    out = opt.run(
        progbar=False,
        cutoff=1e-12,
        n_iter=8,
    )

    expected = np.zeros(16, dtype=complex)
    expected[[0, 15]] = 1.0 / np.sqrt(2.0)
    assert np.allclose(out.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1), expected)
    assert [entry["backend"] for entry in opt.mix_history] == [
        "mpo",
        "dmrg",
        "dmrg",
        "dmrg",
    ]
    assert all(
        entry["reason"] == "guess_direct_dmrg1"
        for entry in opt.mix_history[1:]
    )


def test_mps_optimizer_mix_history_accumulates_control_segments():
    """Mixed-mode diagnostics should cover all gate segments in one run."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00"),
        gates=[
            (qu.hadamard(), (0,)),
            ("measure", "Z", 0, +1),
            (qu.CNOT(), (0, 1)),
        ],
        chi=2,
        mode="mix",
    )

    opt.run(progbar=False, seed=7)

    assert [event["step"] for event in opt.mix_history] == [1, 2]
    assert [event["backend"] for event in opt.mix_history] == ["mpo", "dmrg"]
    assert opt.last_mix_summary["mpo_steps"] == 1
    assert opt.last_mix_summary["dmrg_steps"] == 1


def test_mps_optimizer_mix_one_site_is_exact_at_target_bond():
    """One-site gates stay on the exact fast path at the target bond."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=17)
    assert p0.max_bond() == 2
    gates = [(qu.hadamard(), (1,))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="mix")
    opt.run(progbar=False, cutoff=1e-12)

    assert opt.mix_history[0]["backend"] == "mpo"
    assert opt.mix_history[0]["reason"] == "one_site_exact"


def test_mps_optimizer_mix_falls_back_to_mpo_on_nonfinite_dmrg(monkeypatch):
    """Mix mode should restore and use MPO if DMRG leaves non-finite data."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=19)
    p0_ref = p0.copy()
    gates = [(qu.CNOT(), (0, 2))]
    original_run_dmrg = py.MpsOptimizer._run_dmrg

    def nonfinite_dmrg(self, *args, **kwargs):
        original_run_dmrg(self, *args, **kwargs)
        center = self._current_orthog(self.p)[0]
        data = np.asarray(self.p[center].data)
        self.p[center].modify(data=np.full_like(data, np.nan))

    monkeypatch.setattr(py.MpsOptimizer, "_run_dmrg", nonfinite_dmrg)
    opt = py.MpsOptimizer(p0, gates=gates, chi=2, mode="mix", inplace=True)

    opt.run(progbar=False, cutoff=1e-12, finite_check=True)

    assert opt.mix_history[0]["backend"] == "mpo"
    assert opt.mix_history[0]["reason"] == "dmrg_fallback"
    assert "non-finite" in opt.mix_history[0]["fallback_error"]
    assert opt.last_mix_summary["fallback_steps"] == 1
    assert opt.p is p0
    assert py.MpsOptimizer._mps_data_is_finite(opt.p)
    reference = py.MpsOptimizer(p0_ref, gates=gates, chi=2, mode="mpo")
    reference.run(progbar=False, cutoff=1e-12)
    assert np.allclose(opt.p.to_dense(), reference.p.to_dense())


def test_mps_optimizer_three_site_fit_uses_window_and_falls_back_short():
    """MpsOptimizer should use three-site FIT and shorten adjacent windows."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(qu.hadamard(), (0,)), (qu.CNOT(), (0, 3))],
        chi=2,
        mode="dmrg",
    )
    optimizer.run(
        progbar=False,
        n_iter=3,
        fit_rtol=None,
        fit_block_size=3,
        timing=True,
    )
    assert optimizer._last_dmrg_fit_diagnostics["block_size"] == 3
    assert optimizer._last_dmrg_fit_diagnostics["adaptive_sweeps"] == 2
    assert optimizer._last_dmrg_fit_diagnostics["one_site_refinement_sweeps"] == 1
    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [3, 3, 1]
    assert [
        record["site_count"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [2, 2, 4]

    adjacent = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="dmrg",
    )
    adjacent.run(
        progbar=False,
        n_iter=1,
        fit_rtol=None,
        fit_block_size=3,
        timing=True,
    )
    assert adjacent._last_dmrg_fit_diagnostics["block_size"] == 2
    assert adjacent._last_dmrg_fit_diagnostics["one_site_refinement_sweeps"] == 0
    assert [
        record["block_size"]
        for record in adjacent.get_run_timing()["fit_steps"]
    ] == [2]


@pytest.mark.parametrize(
    ("where", "block_size"),
    [((0, 2), 2), ((0, 3), 3)],
)
def test_mps_optimizer_boundary_long_range_uses_fixed_handoff(
    where,
    block_size,
    monkeypatch,
):
    """A window one site wider than its FIT block is not rank-adaptive."""
    adaptive_rank_flags = []
    original_run_fit_gate = py.MpsOptimizer._run_fit_gate

    def record_schedule(self, fit, **kwargs):
        adaptive_rank_flags.append(bool(kwargs["adaptive_until_rank"]))
        return original_run_fit_gate(self, fit, **kwargs)

    monkeypatch.setattr(py.MpsOptimizer, "_run_fit_gate", record_schedule)
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state(
            "0" * (max(where) + 1),
            dtype="complex128",
        ),
        gates=[(qu.CNOT(), where)],
        chi=2,
        mode="dmrg",
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        fit_rtol=None,
        fit_block_size=block_size,
    )

    assert adaptive_rank_flags == [False]


def test_mps_optimizer_batched_boundary_long_range_uses_fixed_handoff(
    monkeypatch,
):
    """The batched DMRG path applies the same inclusive-span rule."""
    adaptive_rank_flags = []
    original_run_fit_gate = py.MpsOptimizer._run_fit_gate

    def record_schedule(self, fit, **kwargs):
        adaptive_rank_flags.append(bool(kwargs["adaptive_until_rank"]))
        return original_run_fit_gate(self, fit, **kwargs)

    monkeypatch.setattr(py.MpsOptimizer, "_run_fit_gate", record_schedule)
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1)), (qu.CNOT(), (1, 2))],
        chi=2,
        mode="dmrg",
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        fit_rtol=None,
        fit_block_size=2,
        fit_layer_size=2,
    )

    assert adaptive_rank_flags == [False]


@pytest.mark.parametrize("block_size", (2, 3))
def test_mps_optimizer_adaptive_blocks_do_not_preexpand_bonds(
    block_size,
    monkeypatch,
):
    """Adaptive FIT must grow only bonds reached by its native SVD splits."""
    length = 8
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0" * length, dtype="complex128"),
        gates=[(qu.hadamard(), (0,)), (qu.CNOT(), (0, 4))],
        chi=2,
        mode="dmrg",
    )

    def fail_rank_warmup(*args, **kwargs):
        raise AssertionError(
            "adaptive FIT must not pre-expand MPS bonds before fitting"
        )

    monkeypatch.setattr(
        optimizer,
        "_prepare_one_site_dmrg_state",
        fail_rank_warmup,
    )
    optimizer.run(
        progbar=False,
        n_iter=3,
        fit_rtol=None,
        fit_block_size=block_size,
        cutoff=0.0,
        target_cutoff=0.0,
        fit_single_pair_fast_path=False,
        timing=True,
    )

    assert [
        optimizer.p.bond_size(site, site + 1) for site in range(length - 1)
    ] == [2, 2, 2, 2, 1, 1, 1]
    assert optimizer._last_dmrg_fit_diagnostics["block_size"] == block_size


def test_mps_optimizer_mix_guess_direct_does_not_preexpand_bonds(monkeypatch):
    """Mixed rank growth belongs to the disposable direct guess."""
    length = 8
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0" * length, dtype="complex128"),
        gates=[(qu.hadamard(), (0,)), (qu.CNOT(), (0, 4))],
        chi=2,
        mode="mix",
    )

    def fail_rank_padding(*args, **kwargs):
        raise AssertionError("mixed guess-direct must not pre-expand live bonds")

    monkeypatch.setattr(
        optimizer,
        "_prepare_one_site_dmrg_state",
        fail_rank_padding,
    )
    optimizer.run(
        progbar=False,
        n_iter=3,
        fit_rtol=None,
        cutoff=0.0,
        target_cutoff=0.0,
        timing=True,
    )

    assert [
        optimizer.p.bond_size(site, site + 1) for site in range(length - 1)
    ] == [2, 2, 2, 2, 1, 1, 1]
    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["block_size"] == 1
    assert diagnostics["fit_init_strategy"] == "guess_direct"
    assert diagnostics["guess_method"] == "direct"
    assert optimizer.mix_history[-1]["backend"] == "dmrg"


@pytest.mark.parametrize(
    "run_kwargs",
    [
        {"fit_block_size": 2},
        {"fit_block_size": 3},
        {"fit_init_strategy": "guess-src"},
    ],
)
def test_mps_optimizer_mix_rejects_nonfixed_fit_policy(run_kwargs):
    """Mixed mode exposes one algorithm, not block/guess submodes."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="mix",
    )

    with pytest.raises(ValueError, match="mode='mix' fixes"):
        optimizer.run(progbar=False, **run_kwargs)


def test_unitary_fit_stabilization_preserves_working_norm():
    """Unitary FIT stabilization keeps the live state at its working norm."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.hadamard(), (0,)), (qu.CNOT(), (0, 1))],
        chi=1,
        mode="fit",
    )

    out = optimizer.run(progbar=False, n_iter=2, stabilize_unitary=True)
    raw = out.copy()
    raw.exponent = 0.0

    assert float(np.real(raw.norm())) == pytest.approx(1.0, abs=1.0e-12)


def test_dmrg_fit_layer_size_and_target_cutoff_are_independent(monkeypatch):
    """Paper-style gate blocks must not reuse the output truncation cutoff."""
    calls = []
    original = py.MpsOptimizer._build_dmrg_batch_target

    def record_target(
        self,
        p,
        gates,
        where,
        target_cutoff,
        cutoff_mode="rsum2",
        *,
        target_strategy="auto",
    ):
        calls.append(
            (len(gates), float(target_cutoff), cutoff_mode, target_strategy)
        )
        return original(
            self,
            p,
            gates,
            where,
            target_cutoff,
            cutoff_mode,
            target_strategy=target_strategy,
        )

    monkeypatch.setattr(
        py.MpsOptimizer,
        "_build_dmrg_batch_target",
        record_target,
    )
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(
            4, bond_dim=2, phys_dim=2, dtype="complex128", seed=207
        ),
        gates=[
            (qu.CNOT(), (0, 2)),
            (qu.CNOT(), (1, 3)),
            (qu.CNOT(), (0, 3)),
        ],
        chi=2,
        mode="dmrg",
    )

    optimizer.run(
        progbar=False,
        n_iter=2,
        cutoff=1.0e-3,
        target_cutoff=0.0,
        fit_layer_size=2,
    )

    assert calls == [
        (2, 0.0, "rsum2", "layered"),
        (1, 0.0, "rsum2", "layered"),
    ]


def test_mps_optimizer_fit_mode_is_clear_dmrg_alias():
    """The public FIT spelling should select the maintained DMRG kernel."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="fit",
    )

    out = optimizer.run(progbar=False, n_iter=1)

    assert optimizer.mode == "dmrg"
    assert out.L == 2


@pytest.mark.parametrize(
    ("mode", "expected_blocks"),
    [
        ("dmrg1", [2, 2, 1]),
        ("dmrg2", [2, 2, 1]),
        ("dmrg3", [3, 3, 2]),
    ],
)
def test_mps_optimizer_dmrg_mode_aliases_select_block_size(mode, expected_blocks):
    """Named DMRG modes select the corresponding FIT block size."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode=mode,
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        fit_rtol=None,
        timing=True,
    )

    assert optimizer.mode == "dmrg"
    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == expected_blocks


def test_mps_optimizer_mix_uses_norm_guard_without_full_scan(monkeypatch):
    """Successful mixed FIT should avoid a full post-update data scan."""
    original_check = py.MpsOptimizer._mps_data_is_finite
    checks = 0

    def counted_check(candidate):
        nonlocal checks
        checks += 1
        return original_check(candidate)

    monkeypatch.setattr(
        py.MpsOptimizer,
        "_mps_data_is_finite",
        staticmethod(counted_check),
    )
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(
            3, bond_dim=2, phys_dim=2, dtype="complex128", seed=202
        ),
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="mix",
    )

    opt.run(progbar=False, n_iter=3, fit_rtol=None)

    assert opt.mix_history[0]["backend"] == "dmrg"
    assert checks == 0


def test_mps_optimizer_mix_guess_direct_avoids_full_data_scan(monkeypatch):
    """Mixed guess-direct/FIT should avoid a full post-update data scan."""
    original_check = py.MpsOptimizer._mps_data_is_finite
    checks = 0

    def counted_check(candidate):
        nonlocal checks
        checks += 1
        return original_check(candidate)

    monkeypatch.setattr(
        py.MpsOptimizer,
        "_mps_data_is_finite",
        staticmethod(counted_check),
    )
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[
            (qu.hadamard(), (0,)),
            (qu.CNOT(), (0, 1)),
            (qu.CNOT(), (1, 2)),
        ],
        chi=2,
        mode="mix",
    )

    opt.run(progbar=False)

    assert [entry["backend"] for entry in opt.mix_history] == [
        "mpo",
        "dmrg",
        "dmrg",
    ]
    assert checks == 0


def test_mps_optimizer_mix_stops_fit_adaptively():
    """Mixed n_iter is a maximum when the FIT norm has converged."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=21
    )
    opt = py.MpsOptimizer(
        state,
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="mix",
    )

    opt.run(
        progbar=False,
        n_iter=8,
        fit_min_iter=2,
        fit_rtol=1e9,
        fit_patience=1,
    )

    event = opt.mix_history[0]
    assert event["backend"] == "dmrg"
    # Mixed mode's default is one-site FIT, so there is no separate block
    # warm-up phase before tolerance stopping.
    assert event["fit_iterations"] == 2
    assert event["fit_converged"] is True
    assert event["fit_relative_change"] <= 1e9


def test_mps_optimizer_dmrg_stops_fit_adaptively():
    """Ordinary DMRG should use the same adaptive FIT stopping controls."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=212
    )
    opt = py.MpsOptimizer(
        state,
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="dmrg",
    )

    opt.run(
        progbar=False,
        n_iter=8,
        fit_min_iter=2,
        fit_rtol=1e9,
        fit_patience=1,
    )

    assert opt._last_dmrg_fit_diagnostics["iterations"] == 4


def test_mps_optimizer_mix_can_keep_fixed_fit_iterations():
    """Disabling mixed FIT tolerance should preserve exact n_iter behavior."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=211
    )
    opt = py.MpsOptimizer(
        state,
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="mix",
    )

    opt.run(progbar=False, n_iter=3, fit_rtol=None)

    event = opt.mix_history[0]
    assert event["fit_iterations"] == 3
    assert event["fit_converged"] is False
    assert event["fit_relative_change"] is None


def test_mps_optimizer_deprecated_fit_controls_warn_and_remain_functional():
    """Legacy mixed FIT names should delegate to the canonical controls."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=217
    )
    opt = py.MpsOptimizer(
        state,
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="dmrg",
    )

    with pytest.warns(DeprecationWarning, match="mix_fit_rtol"):
        opt.run(
            progbar=False,
            n_iter=4,
            mix_fit_rtol=None,
        )

    assert opt._last_dmrg_fit_diagnostics["iterations"] == 4


def test_mps_optimizer_rejects_conflicting_legacy_fit_controls():
    """Mixed old/new convergence policies must never be resolved silently."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000"),
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="dmrg",
    )

    with pytest.warns(DeprecationWarning, match="mix_fit_rtol"):
        with pytest.raises(ValueError, match="different values"):
            opt.run(
                progbar=False,
                fit_rtol=1.0e-7,
                mix_fit_rtol=None,
            )


def test_mix_unitary_stabilization_covers_guess_direct_dmrg1():
    """Mixed guess-direct/DMRG1 should retain the unitary working norm."""
    gates = []
    for depth in range(4):
        start = depth % 2
        for site in range(start, 7, 2):
            gates.append((qu.rand_uni(4, seed=100 + len(gates)), (site, site + 1)))

    stabilized = py.MpsOptimizer(
        qtn.MPS_computational_state("0" * 8, dtype="complex128"),
        gates=gates,
        chi=4,
        mode="mix",
    )
    stabilized.run(
        progbar=False,
        n_iter=1,
        fit_rtol=None,
        stabilize_unitary=True,
    )

    unstabilized = py.MpsOptimizer(
        qtn.MPS_computational_state("0" * 8, dtype="complex128"),
        gates=gates,
        chi=4,
        mode="mix",
    )
    unstabilized.run(
        progbar=False,
        n_iter=1,
        fit_rtol=None,
        stabilize_unitary=False,
    )

    assert _mps_data_norm(stabilized.p) == pytest.approx(1.0, abs=1.0e-12)
    assert _mps_data_norm(unstabilized.p) < 0.999


def test_mps_optimizer_mix_nonfinite_sweep_disables_later_dmrg(monkeypatch):
    """A non-finite FIT sweep should fall back once and latch to MPO."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=22
    )
    def failed_fit_sweep(self, *_args, **_kwargs):
        self.iterations_run = 1
        self.converged = False
        self.last_relative_change = None
        raise np.linalg.LinAlgError(
            "Array must not contain infs or NaNs."
        )

    monkeypatch.setattr(
        mps_optimizer_module.FIT,
        "run_gate",
        failed_fit_sweep,
    )
    opt = py.MpsOptimizer(
        state,
        gates=[(qu.CNOT(), (0, 2)), (qu.CNOT(), (0, 2))],
        chi=2,
        mode="mix",
    )

    opt.run(progbar=False, n_iter=8, finite_check=True, mix_sticky_nonfinite=True)

    first, second = opt.mix_history
    assert first["reason"] == "dmrg_fallback"
    assert first["fit_iterations"] == 1
    assert first["failed_sweep"] == 1
    assert second["reason"] == "dmrg_disabled_nonfinite"
    assert opt.last_mix_summary["dmrg_disabled"] is True
    assert opt.last_mix_summary["failed_sweep"] == 1


def test_mps_optimizer_mix_inplace_success_two_site_keeps_input_identity():
    """Successful two-site mixed DMRG updates must preserve inplace semantics."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=23)
    before = p0.to_dense().copy()
    opt = py.MpsOptimizer(
        p0,
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="mix",
        inplace=True,
    )

    out = opt.run(progbar=False, cutoff=1e-12, n_iter=3)

    assert out is p0
    assert opt.p is p0
    assert opt.mix_history[0]["backend"] == "dmrg"
    assert not np.allclose(p0.to_dense(), before)


def test_mps_optimizer_mix_inplace_guess_direct_opens_short_bond():
    """In-place mixed replay commits one-site FIT from its direct guess."""
    dense = np.zeros((2, 2, 2, 2), dtype=complex)
    dense[0, 0, 0, 0] = 1.0 / np.sqrt(2.0)
    dense[1, 1, 0, 0] = 1.0 / np.sqrt(2.0)
    p0 = qtn.MatrixProductState.from_dense(dense)
    opt = py.MpsOptimizer(
        p0,
        gates=[(qu.hadamard(), (2,)), (qu.CNOT(), (2, 3))],
        chi=2,
        mode="mix",
        inplace=True,
    )

    out = opt.run(
        progbar=False,
        cutoff=1e-12,
        n_iter=4,
    )

    assert out is p0
    assert opt.p is p0
    assert opt.mix_history[1]["backend"] == "dmrg"
    assert opt.mix_history[1]["reason"] == "guess_direct_dmrg1"
    assert p0.bond_size(2, 3) == 2


def test_mps_optimizer_mix_trial_copies_only_active_canonical_path():
    """Mixed transactions isolate the active window without copying the chain."""
    p0 = qtn.MPS_rand_state(5, bond_dim=2, phys_dim=2, dtype="complex128", seed=28)
    opt = py.MpsOptimizer(p0, gates=[], chi=2, mode="mix", inplace=True)
    left_inds = tuple(tensor.left_inds for tensor in opt.p)

    trial, sites = opt._copy_mix_trial(opt.p, (0, 2), {"cur_orthog": (0, 0)})

    assert sites == (0, 1, 2)
    assert tuple(tensor.left_inds for tensor in trial) == left_inds
    assert trial[0].data is not opt.p[0].data
    assert trial[2].data is not opt.p[2].data
    assert trial[3].data is opt.p[3].data
    before = np.array(opt.p[0].data, copy=True)
    trial[0].modify(data=np.zeros_like(trial[0].data))
    assert np.allclose(opt.p[0].data, before)


def test_mps_optimizer_mix_inplace_commit_preserves_left_inds():
    """Committing a trial must retain Quimb's canonical-isometry metadata."""
    p0 = qtn.MPS_rand_state(
        5,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=281,
    )
    opt = py.MpsOptimizer(p0, gates=[], chi=2, mode="mix", inplace=True)
    committed = opt.p
    trial, sites = opt._copy_mix_trial(
        committed,
        (0, 4),
        opt.info_c,
    )
    trial.canonize([4], cur_orthog=opt.info_c["cur_orthog"])
    expected_left_inds = tuple(tensor.left_inds for tensor in trial)

    opt._commit_mix_trial(committed, trial, sites=sites)

    assert opt.p is committed
    assert tuple(tensor.left_inds for tensor in committed) == expected_left_inds


def test_mps_optimizer_mix_fallback_restores_unitary_norm_tracking(monkeypatch):
    """A failed DMRG trial must restore state before MPO fallback."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=29)
    gates = [(qu.CNOT(), (0, 2))]
    original_run_dmrg = py.MpsOptimizer._run_dmrg

    def failed_dmrg(self, *args, **kwargs):
        original_run_dmrg(self, *args, **kwargs)
        self._unitary_previous_norm = 123.456
        raise RuntimeError("forced DMRG failure")

    monkeypatch.setattr(py.MpsOptimizer, "_run_dmrg", failed_dmrg)
    opt = py.MpsOptimizer(
        p0,
        gates=gates,
        chi=2,
        mode="mix",
        inplace=True,
    )
    opt.run(progbar=False, cutoff=1e-12)
    assert py.MpsOptimizer._mps_data_is_finite(opt.p)
    assert opt.mix_history[-1]["backend"] == "mpo"


def test_mps_optimizer_mix_strict_restores_then_reraises(monkeypatch):
    """Strict mixed mode should expose DMRG errors without corrupting state."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=30)
    before = p0.to_dense().copy()

    def failed_dmrg(self, *args, **kwargs):
        raise RuntimeError("strict DMRG failure")

    monkeypatch.setattr(py.MpsOptimizer, "_run_dmrg", failed_dmrg)
    opt = py.MpsOptimizer(
        p0,
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="mix",
        inplace=True,
    )

    with pytest.raises(RuntimeError, match="strict DMRG failure"):
        opt.run(progbar=False, mix_strict=True)
    assert opt.p is p0
    assert np.allclose(p0.to_dense(), before)
    assert opt.mix_history == []


def test_mps_optimizer_mix_interrupt_restores_trial_state(monkeypatch):
    """Interrupting a DMRG trial must leave the committed MPS usable."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=31)
    before = p0.to_dense().copy()

    def interrupt(self, *args, **kwargs):
        data = np.asarray(self.p[0].data)
        self.p[0].modify(data=np.full_like(data, np.nan))
        raise KeyboardInterrupt

    monkeypatch.setattr(py.MpsOptimizer, "_run_dmrg", interrupt)
    opt = py.MpsOptimizer(
        p0,
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="mix",
        inplace=True,
    )

    with pytest.raises(KeyboardInterrupt):
        opt.run(progbar=False)
    assert opt.p is p0
    assert py.MpsOptimizer._mps_data_is_finite(p0)
    assert np.allclose(p0.to_dense(), before)
    assert opt.mix_history == []


def test_mps_optimizer_mix_batches_two_site_transactions():
    """Mixed mode should support transactional DMRG batches."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128", seed=37),
        gates=[(qu.CNOT(), (0, 2)), (qu.CNOT(), (1, 3))],
        chi=2,
        mode="mix",
    )

    out = opt.run(progbar=False, cutoff=1e-12, n_iter=3, k_2q_batch=2)

    assert out.max_bond() <= 2
    assert [entry["backend"] for entry in opt.mix_history] == ["dmrg", "dmrg"]
    assert opt.mix_history[0]["reason"] == "guess_direct_dmrg1"
    assert opt.mix_history[1]["reason"] == "dmrg_batch"


def test_mps_optimizer_batch_collection_respects_spatial_span():
    """Span-aware batching splits disjoint gates before a wide FIT window."""
    gates = [qu.CNOT(), qu.CNOT(), qu.CNOT()]
    where = [(0, 1), (2, 3), (10, 11)]

    batch_G, batch_where, count, next_idx = py.MpsOptimizer._collect_dmrg_batch(
        gates,
        where,
        0,
        3,
        max_span=5,
    )

    assert batch_G == gates[:2]
    assert batch_where == where[:2]
    assert count == 2
    assert next_idx == 2


def test_mps_optimizer_quality_checks_report_finite_canonical_state():
    """Periodic quality checks expose finite data and gauge coverage."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 2)), (qu.CNOT(), (1, 3))],
        chi=2,
        mode="dmrg",
    )

    opt.run(progbar=False, n_iter=2, quality_check_every=True)

    assert len(opt.get_quality_checks()) == 2
    assert all(record["finite"] for record in opt.get_quality_checks())
    assert all(record["canonical_ok"] for record in opt.get_quality_checks())


def test_mps_optimizer_mix_rejects_initial_bond_above_chi():
    """Mixed mode must not silently violate its configured bond limit."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, bond_dim=4, phys_dim=2, dtype="complex128", seed=41),
        gates=[(qu.CNOT(), (0, 3))],
        chi=2,
        mode="mix",
    )

    with pytest.raises(ValueError, match="initial MPS max bond"):
        opt.run(progbar=False)


def test_mps_optimizer_mix_history_keeps_logical_layout_sites():
    """Mixed history should expose logical and execution gate locations."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=2,
        mode="mix",
    )
    opt.apply_layout((3, 2, 1, 0), layout_report=False)
    opt.run(progbar=False)

    assert opt.mix_history[0]["where"] == (0, 3)
    assert opt.mix_history[0]["execution_where"] == (3, 0)


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


def test_mps_optimizer_accepts_stabilizer_style_symbolic_gate_stream():
    """Named entries should resolve to the same matrices as Pepsy primitives."""
    theta = 0.19
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[("H", 0), ("rzz", theta, 0, 1)],
        chi=4,
        mode="svd",
    )

    assert opt.where == [(0,), (0, 1)]
    assert opt.G[0].shape == (2, 2)
    assert opt.G[1].shape == (2, 2, 2, 2)

    expected = p0.copy()
    expected.gate_(py.h(), 0, contract=True)
    expected.gate_(py.rzz(theta), (0, 1), contract="split", cutoff=0.0)
    opt.run(progbar=False, cutoff=1.0e-12)

    np.testing.assert_allclose(
        np.asarray(opt.p.to_dense()).reshape(-1),
        np.asarray(expected.to_dense()).reshape(-1),
        atol=1.0e-10,
    )


def test_mps_optimizer_symbolic_gate_stream_uses_explicit_to_backend():
    """Named gates should be converted before strict backend validation."""
    torch = pytest.importorskip("torch")

    backend = py.backend_torch(dtype=torch.complex64, device="cpu")
    p0 = qtn.MPS_computational_state("000", dtype="complex64")
    p0.apply_to_arrays(backend)
    converted = []

    def to_backend(array):
        converted.append(array)
        return backend(np.array(array, copy=True))

    opt = py.MpsOptimizer(
        p0,
        gates=[("H", 0), ("rzz", 0.19, 0, 1)],
        chi=4,
        mode="svd",
        to_backend=to_backend,
    )

    assert len(converted) == 2
    assert all(isinstance(gate, torch.Tensor) for gate in opt.G)
    assert all(gate.dtype == torch.complex64 for gate in opt.G)
    opt.run(progbar=False, cutoff=1.0e-7)
    assert opt.backend_info()["backend"] == "torch"


def test_mps_optimizer_symbolic_set_and_add_gates_resolve():
    """Queue mutation should use the same symbolic resolver as construction."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=4,
        mode="svd",
    )

    opt.set_gates([("H", 0)])
    opt.add_gates([("rzz", 0.2, 0, 1)])

    assert len(opt.G) == 2
    assert all(isinstance(gate, np.ndarray) for gate in opt.G)
    opt.run(progbar=False, cutoff=1.0e-12)


def test_mps_optimizer_compiles_gate_stream_once(monkeypatch):
    """Repeated replay reuses compiled stream metadata."""
    calls = []
    original = mps_optimizer_module._normalize_gate_queue

    def count_normalization(gates):
        calls.append(gates)
        return original(gates)

    monkeypatch.setattr(
        mps_optimizer_module,
        "_normalize_gate_queue",
        count_normalization,
    )
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(qu.hadamard(), (1,)), (qu.CNOT(), (0, 3))],
        chi=8,
        mode="svd",
    )

    assert len(calls) == 1
    opt.run(progbar=False, cutoff=1e-12)
    opt.run(progbar=False, cutoff=1e-12)
    assert len(calls) == 1


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

    out = opt.run(
        progbar=False,
        cutoff=0.0,
        non_unitary=True,
        normalize_final=False,
    )
    vec = out.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 0.7
    expected[5] = 0.3

    assert opt.event_types == ["submpo"]
    assert opt.where == [(1, 3)]
    assert np.allclose(vec, expected)
    assert out.max_bond() <= 8


def test_mps_optimizer_dmrg_mode_applies_submpo_as_layered_fit_target():
    """DMRG keeps an explicit sub-MPO lazy and fits its tagged target layer."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(1, 3), targets=(1, 3))
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (1, 3))],
        chi=2,
        mode="dmrg2",
    )

    out = opt.run(
        progbar=False,
        cutoff=1.0e-12,
        non_unitary=True,
        normalize_final=False,
        fit_overlap_diagnostics=True,
    )
    vec = out.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 0.7
    expected[5] = 0.3

    assert np.allclose(vec, expected)
    diagnostics = opt.get_fit_diagnostics()
    assert diagnostics["target_strategy"] == "layered"
    assert diagnostics["guess_method"] == "src"
    assert diagnostics["fit_overlap_fidelity"] == pytest.approx(1.0)


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

    out = opt.run(
        progbar=False,
        cutoff=0.0,
        non_unitary=True,
        normalize_final=False,
    )
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
    """Applying a reusable event MPO should not mutate its payload."""
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
        non_unitary=True,
    )
    vec = out.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 0.7
    expected[5] = 0.3

    assert np.allclose(vec, expected)

    reuse = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (1, 3))],
        chi=8,
        mode="mpo",
    ).run(
        progbar=False,
        cutoff=0.0,
        non_unitary=True,
        normalize_final=False,
    )
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


def test_mps_optimizer_submpo_stream_events_require_mpo_or_dmrg_mode():
    """SVD/swap/exact modes still reject sub-MPO stream events clearly."""
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=3, sites=(0, 2), targets=(0, 2))
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[("submpo", mpo, (0, 2))],
        chi=8,
        mode="svd",
    )

    with pytest.raises(ValueError, match="require an MPO or DMRG mode"):
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

    out = opt.run(progbar=False, cutoff=1e-12, n_iter=2)

    assert out is opt.p


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "swap", "perm", "svd"])
def test_mps_optimizer_one_site_unitary_preserves_cached_center_and_norm(mode):
    """A one-site unitary must not invent a new orthogonality center."""
    state = qtn.MPS_rand_state(
        6,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=213,
    )
    state.multiply_(2.0, spread_over="all")
    optimizer = py.MpsOptimizer(
        state,
        gates=[(qu.hadamard(), (0,))],
        chi=16,
        mode=mode,
    )

    optimizer.run(progbar=False, cutoff=0.0, n_iter=2)

    assert optimizer.info_c["cur_orthog"] == tuple(
        optimizer.p.calc_current_orthog_center()
    )

    optimizer.set_gates([(qu.CNOT(), (0, 1))])
    optimizer.run(progbar=False, cutoff=0.0, n_iter=2)

    assert optimizer.info_c["cur_orthog"] == tuple(
        optimizer.p.calc_current_orthog_center()
    )


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

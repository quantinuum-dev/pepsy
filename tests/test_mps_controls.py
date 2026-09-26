"""MPS controls regression tests."""


import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn
import pepsy as py


pytestmark = [pytest.mark.core, pytest.mark.mps]


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


def test_extracted_controls_and_norm_preserve_subclass_hooks():
    class HookedOptimizer(py.MpsOptimizer):
        def _finish_measurement_center(self, site, *, renormalize):
            self.finished_sites.append(site)
            return super()._finish_measurement_center(site, renormalize=renormalize)

        def _canonical_span_norm(self, p, where, *, fallback=True):
            self.norm_spans.append(where)
            return super()._canonical_span_norm(p, where, fallback=fallback)

    opt = HookedOptimizer(
        qtn.MPS_computational_state("00"),
        [("h", 0), ("measure", "Z", 0, -1), ("reset", 0)],
        chi=4,
    )
    opt.finished_sites = []
    opt.norm_spans = []
    opt.run(progbar=False)
    opt.normalize()
    assert opt.finished_sites
    assert opt.norm_spans
    np.testing.assert_allclose(
        np.abs(opt.to_dense().reshape(-1)), [1.0, 0.0, 0.0, 0.0], atol=1e-12
    )
    assert opt.measurements[0][:3] == ("Z", (0,), -1)
    assert float(abs(opt.p.norm())) == pytest.approx(1.0)


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
    event = opt.get_norm_events()[0]
    assert event["kind"] == "measure"
    assert event["branch_probability"] == pytest.approx(prob)
    assert event["physical_boundary"] is True
    assert event["renormalized"] is True
    assert event["local_infidelity"] == pytest.approx(0.0, abs=1e-10)


def test_mps_optimizer_measure_multisite_pauli():
    """Multi-qubit Pauli measurements should collapse onto the eigenspace."""
    m = qtn.MPS_rand_state(6, 4, seed=2)
    opt = py.MpsOptimizer(m.copy(), [("measure", "ZZ", (1, 3), -1)], chi=8, mode="mpo")
    opt.run(progbar=False)

    assert np.isclose(_dense_pauli_expectation(opt.p, "ZZ", (1, 3)), -1.0)
    assert opt.measurements[0][:3] == ("ZZ", (1, 3), -1)


@pytest.mark.parametrize(
    ("mode", "expected_method"),
    [("mpo", "direct"), ("quimb-src", "src")],
)
def test_mps_optimizer_multisite_measurement_uses_bond_two_submpo(
    monkeypatch, mode, expected_method
):
    """Dense MPS measurements should use the low-bond sub-MPO compressor."""
    calls = []
    original = qtn.MatrixProductState.gate_with_submpo_

    def recording(self, submpo, *args, **kwargs):
        calls.append(
            (
                kwargs.get("method"),
                tuple(kwargs.get("where", ())),
                submpo.max_bond(),
                kwargs.get("max_bond"),
            )
        )
        return original(self, submpo, *args, **kwargs)

    monkeypatch.setattr(
        qtn.MatrixProductState,
        "gate_with_submpo_",
        recording,
    )

    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(6, 2, seed=2, dtype="complex128"),
        [("measure", "XZY", (1, 3, 5), +1)],
        chi=8,
        mode=mode,
    )
    opt.run(progbar=False, cutoff=0.0)

    # Probability preparation is lossless; only the final physical projection
    # consumes the requested compressor and chi.
    assert calls == [
        ("direct", (5, 3), 2, None), ("direct", (3, 1), 2, None),
        (expected_method, (1, 2, 3, 4, 5), 2, 8),
    ]
    assert _dense_pauli_expectation(opt.p, "XZY", (1, 3, 5)) == pytest.approx(
        1.0
    )


def test_mps_optimizer_dmrg_measurement_uses_lazy_submpo_and_src_guess(monkeypatch):
    """DMRG measurements should use a lazy target and the normal SRC guess."""
    methods = []
    original = qtn.MatrixProductState.gate_with_submpo_

    def recording(self, submpo, *args, **kwargs):
        methods.append(kwargs.get("method"))
        return original(self, submpo, *args, **kwargs)

    monkeypatch.setattr(
        qtn.MatrixProductState,
        "gate_with_submpo_",
        recording,
    )

    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(6, 2, seed=2, dtype="complex128"),
        [("measure", "XZY", (1, 3, 5), +1)],
        chi=8,
        mode="dmrg2",
    )
    opt.run(
        progbar=False,
        n_iter=3,
        fit_min_iter=1,
        fit_patience=1,
        cutoff=0.0,
    )

    diagnostics = opt.get_fit_diagnostics()
    assert methods == ["direct", "direct", "lazy", "src"]
    assert diagnostics["target_representation"] == "lazy_submpo"
    assert diagnostics["guess_method"] == "src"
    assert diagnostics["fallback"] is False
    assert _dense_pauli_expectation(opt.p, "XZY", (1, 3, 5)) == pytest.approx(
        1.0
    )


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


def test_mps_optimizer_expectation_converts_operator_to_state_backend(monkeypatch):
    """The local expectation operator should pass through backend conversion."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, 2, seed=8), gates=[], chi=4, mode="mpo"
    )
    converted = []
    original = opt._to_state_backend

    def recording(array):
        converted.append(array)
        return original(array)

    monkeypatch.setattr(opt, "_to_state_backend", recording)
    opt._state_expectation("Z", (1,))  # pylint: disable=protected-access

    assert len(converted) == 1
    assert converted[0].shape == (2, 2)


def test_mps_optimizer_local_expectation_uses_torch_state_backend(monkeypatch):
    """Torch-backed control expectations should stay on the Torch backend."""
    torch = pytest.importorskip("torch")
    vector = torch.tensor([1.0, 1.0], dtype=torch.complex128)
    vector = vector / torch.linalg.vector_norm(vector)
    opt = py.MpsOptimizer(
        qtn.MPS_product_state([vector.clone() for _ in range(3)]),
        gates=[],
        chi=4,
        mode="mpo",
    )
    observed_operators = []
    original = qtn.MatrixProductState.local_expectation_canonical

    def recording(self, operator, *args, **kwargs):
        observed_operators.append(operator)
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(
        qtn.MatrixProductState,
        "local_expectation_canonical",
        recording,
    )
    assert opt._state_expectation("Z", (1,)) == pytest.approx(0.0)

    assert isinstance(observed_operators[0], torch.Tensor)
    assert all(isinstance(tensor.data, torch.Tensor) for tensor in opt.p.tensors)


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


def test_mps_optimizer_sync_canonicalization_repairs_external_readout():
    """External Quimb readout can be explicitly rebound to ``info_c``."""
    z = np.diag([1.0, -1.0]).astype(complex)
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(6, 4, seed=17), gates=[], chi=8, mode="dmrg2"
    )
    opt.canonize_mps(opt.p, 0)
    assert opt.info_c["cur_orthog"] == (0, 0)

    # This deliberately models a lower-level caller bypassing Pepsy's
    # tracked expectation helper.
    opt.p.local_expectation_canonical(z, (5,), normalized=True)
    assert opt.info_c["cur_orthog"] == (0, 0)
    assert tuple(opt.p.calc_current_orthog_center()) == (5, 5)

    assert opt.sync_canonicalization() == (5, 5)
    assert opt.info_c["cur_orthog"] == (5, 5)

    opt.set_gates([(np.eye(4, dtype=complex), (0, 1))])
    opt.run(progbar=False, n_iter=1, cutoff=0.0)
    assert tuple(opt.p.calc_current_orthog_center()) == opt.info_c["cur_orthog"]


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
    assert [event["kind"] for event in opt.get_norm_events()] == ["reset"]
    assert opt.norm_diagnostics()["infidelity"] == pytest.approx(0.0, abs=1e-10)


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


@pytest.mark.parametrize(
    "conditional",
    [
        ("if", -1, 1, (np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex), 1)),
        {
            "kind": "feed_forward",
            "record": -1,
            "value": 1,
            "then": (np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex), 1),
        },
    ],
)
def test_mps_optimizer_conditional_gate_follows_measurement_bit(conditional):
    """MPS feed-forward applies only the matching classical branch."""
    hadamard = np.array(
        [[1.0, 1.0], [1.0, -1.0]], dtype=complex
    ) / np.sqrt(2.0)
    flip = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    for outcome, expected_index in ((+1, 0), (-1, 3)):
        entry = conditional.copy() if isinstance(conditional, dict) else conditional
        if isinstance(entry, dict):
            entry["then"] = (flip, 1)
        else:
            entry = (entry[0], entry[1], entry[2], (flip, 1))
        optimizer = py.MpsOptimizer(
            qtn.MPS_computational_state("00", dtype="complex128"),
            [(hadamard, 0), ("measure", "Z", 0, outcome), entry],
            chi=4,
            mode="mpo",
        )
        optimizer.run(progbar=False)
        state = optimizer.to_dense().reshape(-1)
        expected = np.zeros(4, dtype=complex)
        expected[expected_index] = 1.0
        np.testing.assert_allclose(state, expected, atol=1e-10)


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


def test_mps_optimizer_cap_events_replay_through_layout_finder():
    """Direct caps replay through a transient layout and restore readout order."""
    su4 = qu.rand_uni(4, seed=1)
    stream = [
        (su4, (0, 3)),
        ("cap", 1, [1.0, 1.0]),
        (qu.CNOT(), (0, 2)),
    ]
    reference = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        stream,
        chi=8,
        mode="mpo",
    )
    reference.run(progbar=False, cutoff=0.0)
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000"),
        stream,
        chi=8,
        mode="mpo",
    )
    opt.run(
        progbar=False,
        use_layout_finder=True,
        layout_report=False,
        cutoff=0.0,
    )
    assert opt.logical_order == [0, 1, 2]
    assert opt.mps_length_diagnostics()["length_history"] == (4, 3)
    assert np.allclose(
        np.asarray(opt.to_dense()).reshape(-1),
        np.asarray(reference.to_dense()).reshape(-1),
    )


def test_mps_optimizer_conditional_cap_rejects_layouts():
    """A nested cap is still incompatible with a fixed-length layout."""
    stream = [
        ("measure", "Z", 0, +1),
        ("if", -1, 0, ("cap", 1, [1.0, 1.0])),
    ]
    initial = qtn.MPS_computational_state("0000", dtype="complex128")

    persistent = py.MpsOptimizer(initial.copy(), stream, chi=8, mode="direct")
    with pytest.raises(ValueError, match="conditional cap"):
        persistent.apply_layout((0, 2, 3, 1), layout_report=False)

    transient = py.MpsOptimizer(initial.copy(), stream, chi=8, mode="direct")
    with pytest.raises(ValueError, match="conditional cap"):
        transient.run(
            progbar=False,
            use_layout_finder=True,
            layout_report=False,
        )


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

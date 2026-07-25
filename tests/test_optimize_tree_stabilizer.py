"""Dense correctness tests for the first TreeStabOptimizer milestone."""

import numpy as np
import pytest
import quimb.tensor as qtn

stim = pytest.importorskip("stim")

import pepsy


H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=complex,
)


def _rzz(theta):
    return np.diag(
        [
            np.exp(-0.5j * theta),
            np.exp(0.5j * theta),
            np.exp(0.5j * theta),
            np.exp(-0.5j * theta),
        ]
    ).astype(complex)


def _rz(theta):
    return np.diag(
        [np.exp(-0.5j * theta), np.exp(0.5j * theta)]
    ).astype(complex)


def _apply_local(state, gate, where, n):
    where = tuple(where) if not isinstance(where, int) else (where,)
    k = len(where)
    tensor = np.asarray(state).reshape((2,) * n)
    op = np.asarray(gate).reshape((2,) * (2 * k))
    out = np.tensordot(op, tensor, axes=(tuple(range(k, 2 * k)), where))
    remaining = [q for q in range(n) if q not in where]
    order = [
        where.index(q) if q in where else k + remaining.index(q)
        for q in range(n)
    ]
    return out.transpose(order).reshape(-1)


def _assert_same_state(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    pivot = int(np.argmax(np.abs(right)))
    phase = 1.0 if abs(right[pivot]) < 1e-14 else left[pivot] / right[pivot]
    assert np.allclose(left, phase * right, atol=1e-10, rtol=1e-10)


def _tree_stab_entangled_coefficient_state():
    """Build an entangled coefficient TTN with an initially identity frame."""
    from pepsy.optimizers.tree import TreeOptimizer

    coefficient = TreeOptimizer([ (H, 0), (CNOT, (0, 1)) ], n=2, chi=64)
    coefficient.run()
    tableau = stim.TableauSimulator()
    tableau.set_num_qubits(2)
    return pepsy.TreeStabOptimizer.from_tableau_and_state(
        tableau, coefficient.tn, chi=64
    )


def test_tree_stab_is_public_and_cliffords_are_tableau_only():
    from pepsy.optimizers.tree_stabilizer import TreeStabOptimizer

    assert pepsy.TreeStabOptimizer is TreeStabOptimizer
    opt = TreeStabOptimizer(2)
    initial = opt.p.to_dense().copy()

    opt.apply([("h", 0), ("cnot", 0, 1)])

    assert opt.p.max_bond() == 1
    assert np.allclose(opt.p.to_dense(), initial)
    expected = _apply_local(_apply_local(np.array([1, 0, 0, 0]), H, (0,), 2), CNOT, (0, 1), 2)
    _assert_same_state(opt.to_statevector(), expected)


def test_tree_stab_chi_none_is_uncapped_and_sampling_is_conditional():
    opt = pepsy.TreeStabOptimizer(
        2,
        chi=None,
        gates=[("h", 0), ("cnot", 0, 1)],
    )
    opt.run()

    assert opt.tree_optimizer.chi is None
    estimate = opt.tree_optimizer.estimate_bonds()
    assert estimate["chi"] is None
    assert estimate["requires_truncation"] is False
    np.testing.assert_allclose(
        opt.probability_bits_many(["00", "01", "10", "11"]),
        [0.5, 0.0, 0.0, 0.5],
        atol=1e-10,
    )
    negative = opt.copy()
    negative.measure_pauli("Z", 0, outcome=-1)
    assert negative.expectation("Z", 0) == pytest.approx(-1.0)
    samples = opt.sample_bits(128, seed=12, shuffle=False)
    assert samples.shape == (128, 2)
    assert np.all(samples[:, 0] == samples[:, 1])


def test_tree_stab_frame_layout_uses_current_conjugated_supports():
    opt = pepsy.TreeStabOptimizer(
        4,
        gates=[("h", 0), ("cnot", 0, 3), ("rz", 0.23, 3)],
        frame_layout="auto",
    )

    assert opt.frame_layout_plan is opt.plan
    assert len(opt.frame_layout_events) == 1
    assert opt.frame_layout_events[0]["support"] == (0, 3)


def test_tree_stab_layout_kwargs_enable_cost_aware_refinement():
    opt = pepsy.TreeStabOptimizer(
        6,
        gates=[("h", 0), ("cnot", 0, 5), ("cnot", 1, 4), ("rz", 0.3, 5)],
        layout_kwargs={
            "objective": "hybrid",
            "hybrid_weights": (1.0, 2.0, 0.5),
            "refine": "greedy",
            "refine_budget": 3,
        },
    )
    assert opt.plan.n == 6
    assert opt._tree.layout_objective == "hybrid"


def test_tree_stab_from_stim_and_stream_analysis():
    opt = pepsy.TreeStabOptimizer.from_stim(stim.Circuit("H 0\nCX 0 1"))

    assert opt.stim_plan.num_qubits == 2
    assert opt.stim_sample.gate_stream
    analysis = opt.queued_stream_analysis()
    assert analysis.total_entries == 2
    opt.run()
    np.testing.assert_allclose(
        opt.probability_bits_many(["00", "01", "10", "11"]),
        [0.5, 0.0, 0.0, 0.5],
        atol=1e-10,
    )


def test_tree_stab_from_stim_lowers_record_control_to_feed_forward():
    opt = pepsy.TreeStabOptimizer.from_stim(
        stim.Circuit("X 0\nM 0\nCX rec[-1] 1")
    )
    assert any(
        isinstance(entry, tuple)
        and isinstance(entry[0], str)
        and entry[0] == "if"
        for entry in opt._queue
    )
    opt.run()
    expected = np.zeros(4, dtype=complex)
    expected[3] = 1.0
    _assert_same_state(opt.to_statevector(), expected)


def test_tree_stab_submpo_matches_mps_coefficient_frame_contract():
    operator = np.diag([1.0, 0.7, 0.4, -0.3]).reshape(2, 2, 2, 2)
    submpo = qtn.MatrixProductOperator.from_dense(
        operator,
        dims=(2, 2),
        sites=(0, 1),
        L=3,
        max_bond=None,
        cutoff=0.0,
    )
    stream = [("h", 0), ("cnot", 0, 1), ("submpo", submpo, (0, 1))]

    tree = pepsy.TreeStabOptimizer(3, chi=None, gates=stream)
    mps = pepsy.MpsStabOptimizer(3, chi=None, gates=stream)
    assert tree.current_frame_layout()["frame_events"][0]["support"] == (0, 1)
    tree.run()
    mps.run()

    _assert_same_state(tree.to_statevector(), mps.to_statevector())


def test_tree_stab_amplitude_probability_match_dense_readout():
    opt = pepsy.TreeStabOptimizer(
        2, gates=[("h", 0), ("cnot", 0, 1)]
    ).run()

    assert opt.amplitude("00") == pytest.approx(1.0 / np.sqrt(2.0))
    assert opt.amplitude([1, 1]) == pytest.approx(1.0 / np.sqrt(2.0))
    assert opt.probability("00") == pytest.approx(0.5)
    assert opt.probability("01") == pytest.approx(0.0)


def test_tree_stab_parity_advice_runner_ghz_and_rank():
    from pepsy.optimizers.tree_stabilizer import StabilizerTreeRunResult

    ghz = pepsy.TreeStabOptimizer.ghz(3)
    np.testing.assert_allclose(
        np.abs(ghz.to_statevector()),
        [1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2)],
    )
    assert pepsy.TreeStabOptimizer(1).apply(
        [("ry", 0.37, 0)]
    ).pseudo_stabilizer_rank() == 2

    advice = pepsy.TreeStabOptimizer.recommend_settings(
        [("h", 0), ("t", 0)], n_qubits=1
    )
    assert advice.settings["max_operator_qubits"] == 2
    assert advice.settings["track_truncation"] is True
    result = pepsy.TreeStabOptimizer.run_stream(
        [("h", 0), ("t", 0)], n_qubits=1, settings={"chi": None}
    )
    assert isinstance(result, StabilizerTreeRunResult)
    assert result.mode == "direct"
    assert result.norm_diagnostics["tracking"] is False


def test_tree_stab_norm_diagnostics_and_sampling_copy_contract():
    opt = pepsy.TreeStabOptimizer(
        2,
        chi=1,
        track_truncation=True,
        exact_cooling=False,
    ).apply([("h", 0), ("cnot", 0, 1), ("rz", 0.37, 1)])
    diagnostics = opt.norm_diagnostics()

    assert diagnostics["tracking"] is True
    assert diagnostics["truncation_report"]["track_truncation"] is True
    assert "projection_diagnostics" in diagnostics
    before = opt.to_statevector().copy()
    opt.sample_bits(64, seed=2)
    np.testing.assert_allclose(opt.to_statevector(), before)


def test_tree_stab_torch_backend_matches_numpy():
    torch = pytest.importorskip("torch")
    backend = pepsy.backend_torch(dtype=torch.complex128, device="cpu")
    stream = [
        ("h", 0),
        ("cnot", 0, 1),
        ("rz", 0.37, 1),
        ("measure", "Z", 1, 1),
    ]
    cpu = pepsy.TreeStabOptimizer(2).apply(stream)
    gpu = pepsy.TreeStabOptimizer(2, to_backend=backend).apply(stream)

    assert gpu.backend_info()["backend"] == "torch"
    assert "torch" in type(gpu.p[0].data).__module__
    _assert_same_state(gpu.to_statevector(), cpu.to_statevector())


def test_tree_stab_cap_matches_mps_and_rebuilds_identity_frame():
    vec = np.array([0.8, -0.3j])
    stream = [("h", 0), ("cnot", 0, 2), ("rz", 0.37, 1)]
    tree = pepsy.TreeStabOptimizer(3, chi=None, max_dense_cap_qubits=6)
    mps = pepsy.MpsStabOptimizer(3, chi=None, max_dense_cap_qubits=6)
    tree.apply(stream)
    mps.apply(stream)

    tree.cap(1, vec)
    mps.cap(1, vec)

    _assert_same_state(tree.to_statevector(), mps.to_statevector())
    assert tree.n == mps.n == 2
    assert tree.probability("00") == pytest.approx(
        mps.probability("00"), abs=1e-7
    )
    assert tree.p.max_bond() > 1


def test_tree_stab_cap_stream_remaps_later_compact_labels():
    vec = np.array([0.75, 0.25j])
    stream = [
        ("h", 0),
        ("cnot", 0, 3),
        ("cap", 1, vec),
        ("rz", 0.19, 2),
    ]
    tree = pepsy.TreeStabOptimizer(
        4, chi=None, gates=stream, max_dense_cap_qubits=6
    )
    mps = pepsy.MpsStabOptimizer(
        4, chi=None, gates=stream, max_dense_cap_qubits=6
    )
    tree.run()
    mps.run()

    _assert_same_state(tree.to_statevector(), mps.to_statevector())
    assert tree.n == 3
    assert tree._queue == []


def test_tree_stab_cap_does_not_lower_state_replacement_to_dense_operator(
    monkeypatch,
):
    from pepsy.optimizers.tree import TreeOptimizer

    opt = pepsy.TreeStabOptimizer(
        4, chi=None, max_dense_cap_qubits=6
    ).apply([
        ("h", 0),
        ("cnot", 0, 3),
        ("rz", 0.37, 1),
    ])

    def fail_dense_operator(*_args, **_kwargs):
        raise AssertionError("cap must not build a dense replacement operator")

    monkeypatch.setattr(
        TreeOptimizer, "apply_subtree_operator", fail_dense_operator
    )
    opt.cap(1, [0.8, -0.3j])
    assert opt.n == 3
    assert opt.p.max_bond() > 1


def test_tree_stab_cap_dense_guard():
    opt = pepsy.TreeStabOptimizer(3, max_dense_cap_qubits=2)
    with pytest.raises(ValueError, match="max_dense_cap_qubits=2"):
        opt.cap(0, [1.0, 0.0])


def test_tree_stab_frame_maps_pauli_rotation_to_tree_coefficient_state():
    theta = 0.37
    opt = pepsy.TreeStabOptimizer(2)
    opt.apply([("h", 0), ("cnot", 0, 1)])
    opt.apply_pauli_rotation(theta, "Z", 0)

    expected = _apply_local(
        _apply_local(np.array([1, 0, 0, 0]), H, (0,), 2), CNOT, (0, 1), 2
    )
    rz = np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])
    expected = _apply_local(expected, rz, (0,), 2)
    _assert_same_state(opt.to_statevector(), expected)
    assert opt.norm() == pytest.approx(1.0)


def test_tree_stab_fixed_measurement_matches_dense_born_probability():
    opt = pepsy.TreeStabOptimizer(1)
    opt.apply(("h", 0))

    outcome, probability, diagnostics = opt.measure_pauli(
        "Z", 0, outcome=+1, return_diagnostics=True
    )

    assert outcome == +1
    assert probability == pytest.approx(0.5)
    assert diagnostics["probability"] == pytest.approx(0.5)
    _assert_same_state(opt.to_statevector(), np.array([1.0, 0.0], dtype=complex))
    assert opt.expectation("Z", 0) == pytest.approx(1.0)


def test_tree_stab_matrix_clifford_and_long_pauli_use_supported_paths():
    opt = pepsy.TreeStabOptimizer(2, max_operator_qubits=1)
    opt.apply([(H, 0), (CNOT, (0, 1))])
    expected = _apply_local(_apply_local(np.array([1, 0, 0, 0]), H, (0,), 2), CNOT, (0, 1), 2)
    _assert_same_state(opt.to_statevector(), expected)

    long_opt = pepsy.TreeStabOptimizer(6, max_operator_qubits=1)
    long_opt.apply([("rot", 0.21, "XXXXXX", tuple(range(6)))])
    assert long_opt.norm() == pytest.approx(1.0)


def test_tree_stab_basis_updating_measurement_preserves_physical_state():
    opt = pepsy.TreeStabOptimizer(2)
    opt.apply([("h", 0), ("cnot", 0, 1)])

    before = opt.to_statevector()
    outcome = opt.measure("Z", 0, outcome=+1, absorb_basis=True)

    assert outcome == +1
    expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    _assert_same_state(opt.to_statevector(), expected)
    assert opt.p.max_bond() == 1
    assert not np.allclose(before, opt.to_statevector())
    assert opt.measurements[-1].outcome == +1
    assert opt.projection_diagnostics[-1]["basis_updated"] is True


def test_tree_stab_basis_updating_measurement_handles_tree_support_order():
    opt = pepsy.TreeStabOptimizer(5)
    opt.apply([("h", 0), ("cnot", 0, 4), ("h", 2), ("cnot", 2, 3)])
    before = opt.to_statevector()

    # Use the explicit absorb_basis path so the tree localizer is tested
    # independently of the fixed-basis projector.
    expectation = opt.expectation("YZX", (0, 2, 4))
    assert expectation == pytest.approx(0.0)
    outcome, probability, diagnostics = opt.measure_pauli(
        "YZX", (0, 2, 4), outcome=+1,
        absorb_basis=True,
        return_diagnostics=True,
    )
    assert outcome == +1
    assert probability == pytest.approx(0.5)
    assert diagnostics["basis_updated"] is True
    acted = before
    for axis, q in zip((np.array([[0, -1j], [1j, 0]]), Z, X), (0, 2, 4)):
        acted = _apply_local(acted, axis, (q,), 5)
    expected = (before + acted) / 2.0
    expected /= np.linalg.norm(expected)
    _assert_same_state(opt.to_statevector(), expected)
    assert opt.expectation("YZX", (0, 2, 4)) == pytest.approx(1.0)
    assert opt.norm() == pytest.approx(1.0)


def test_tree_stab_reset_and_measure_reset_recycle_targets():
    opt = pepsy.TreeStabOptimizer.from_bits("1")
    opt.reset(0)
    _assert_same_state(opt.to_statevector(), np.array([1.0, 0.0], dtype=complex))

    opt = pepsy.TreeStabOptimizer(2)
    opt.apply([("h", 0), ("cnot", 0, 1)])
    measured = opt.measure_reset("Z", 0, outcome=+1)
    assert measured == +1
    _assert_same_state(opt.to_statevector(), np.array([1.0, 0.0, 0.0, 0.0]))
    assert opt.p.max_bond() == 1


def test_tree_stab_basis_update_stream_events_are_supported():
    opt = pepsy.TreeStabOptimizer(
        1,
        gates=[("h", 0), ("measure", "Z", 0, +1, True), ("reset", 0)],
    )
    opt.run()
    assert opt.expectation("Z", 0) == pytest.approx(1.0)


@pytest.mark.parametrize("outcome", [+1, -1])
def test_tree_stab_inject_t_matches_dense_branch(outcome):
    opt = pepsy.TreeStabOptimizer(2)
    opt.apply(("h", 0))
    opt.prepare_magic(1)
    measured = opt.inject_t(0, 1, outcome=outcome)

    assert measured == outcome
    target = _rz(np.pi / 4.0) @ (H @ np.array([1.0, 0.0], dtype=complex))
    ancilla = np.array([1.0, 0.0], dtype=complex) if outcome > 0 else np.array([0.0, 1.0], dtype=complex)
    _assert_same_state(opt.to_statevector(), np.kron(target, ancilla))
    assert opt.norm() == pytest.approx(1.0)
    assert opt.expectation("Z", 1) == pytest.approx(float(outcome))


@pytest.mark.parametrize("phi", [np.pi / 4.0, -np.pi / 4.0, 3.0 * np.pi / 4.0])
def test_tree_stab_inject_rz_matches_dense_state(phi):
    opt = pepsy.TreeStabOptimizer(2)
    opt.apply(("h", 0))
    opt.prepare_magic(1, angle=phi)
    opt.inject_rz(0, 1, phi, outcome=+1)

    plus = H @ np.array([1.0, 0.0], dtype=complex)
    expected = np.kron(_rz(phi) @ plus, np.array([1.0, 0.0], dtype=complex))
    _assert_same_state(opt.to_statevector(), expected)


def test_tree_stab_with_injection_recycles_ancilla_and_matches_direct():
    stream = [("h", 0), ("t", 0), ("t", 0)]
    injected = pepsy.TreeStabOptimizer.with_injection(
        1, stream, n_ancilla=1, seed=7
    )
    direct = pepsy.TreeStabOptimizer(2)
    direct.apply(stream)

    _assert_same_state(injected.to_statevector(), direct.to_statevector())
    assert injected.expectation("Z", 1) == pytest.approx(1.0)
    assert len(injected.immediate_projection_events) == 2
    assert injected.last_immediate_injection_report.n_injections == 2
    assert injected.last_immediate_injection_report.projection_peak_bond >= 1


def test_tree_stab_magic_layout_includes_injection_supports_and_finder_options():
    injected = pepsy.TreeStabOptimizer.with_injection(
        2,
        [("h", 0), ("t", 0), ("cnot", 0, 1), ("tdg", 1)],
        n_ancilla=2,
        layout_kwargs={
            "objective": "hybrid",
            "weight_mode": "auto",
            "refine": "greedy",
            "refine_budget": 4,
            "seed": 13,
        },
        seed=13,
    )

    assert injected._tree.layout_objective == "hybrid"
    assert injected._tree.layout_weight_mode == "auto"
    assert injected._tree.layout_report()["max_arity"] >= 2


def test_tree_stab_injection_protects_reserved_ancillas():
    opt = pepsy.TreeStabOptimizer(2)
    with pytest.raises(ValueError, match="reserved ancilla"):
        opt.run_with_injection([("h", 1), ("t", 0)], ancillas=[1])

    with pytest.raises(RuntimeError, match="pool exhausted"):
        opt.run_with_injection(
            [("t", 0), ("t", 0)], ancillas=[1], recycle=False
        )


@pytest.mark.parametrize("projection_order", ["input", "middle_out", "min_span"])
def test_tree_stab_deferred_injection_matches_direct_circuit(projection_order):
    stream = [
        ("h", 0), ("cnot", 0, 1), ("t", 2), ("tdg", 0),
        ("rz", np.pi / 4.0, 1), ("cnot", 1, 2), ("t", 1),
    ]
    outcomes = [+1, -1, +1, -1]
    direct = pepsy.TreeStabOptimizer(7).apply(stream)
    deferred = pepsy.TreeStabOptimizer.with_deferred_injection(
        3,
        stream,
        outcomes=outcomes,
        projection_order=projection_order,
        seed=11,
    )

    _assert_same_state(deferred.to_statevector(), direct.to_statevector())
    assert deferred.norm() == pytest.approx(1.0)
    assert deferred.last_deferred_injection_report.n_injections == 4
    assert len(deferred.deferred_projection_events) == 4
    assert [
        event.outcome
        for event in sorted(deferred.deferred_projection_events, key=lambda event: event.index)
    ] == outcomes
    assert all(event.mps_span >= 0 for event in deferred.deferred_projection_events)
    assert all(deferred.expectation("Z", ancilla) == pytest.approx(1.0) for ancilla in range(3, 7))


def test_tree_stab_deferred_injection_accepts_explicit_projection_order():
    stream = [("h", 0), ("t", 0), ("tdg", 1), ("t", 1)]
    ancillas = (2, 3, 4)
    deferred = pepsy.TreeStabOptimizer(5)
    deferred.run_with_deferred_injection(
        stream,
        ancillas=ancillas,
        outcomes=[-1, +1, -1],
        projection_order=tuple(reversed(ancillas)),
    )

    assert [event.ancilla for event in deferred.deferred_projection_events] == list(
        reversed(ancillas)
    )
    assert [event.order for event in deferred.deferred_projection_events] == [0, 1, 2]
    assert deferred.last_deferred_injection_report.projection_peak_bond >= 1


def test_tree_stab_deferred_injection_requires_fresh_ancillas():
    with pytest.raises(ValueError, match="one ancilla per injectable gate"):
        pepsy.TreeStabOptimizer.with_deferred_injection(
            2, [("t", 0), ("t", 1)], n_ancilla=1
        )

    with pytest.raises(ValueError, match="reserved ancilla"):
        pepsy.TreeStabOptimizer.with_deferred_injection(
            2, [("h", 2), ("t", 0)], n_ancilla=1
        )


def test_tree_stab_tableau_state_constructor_validates_qubit_count():
    sim = stim.TableauSimulator()
    sim.set_num_qubits(3)
    opt = pepsy.TreeStabOptimizer(2)
    with pytest.raises(ValueError, match="same number of qubits"):
        pepsy.TreeStabOptimizer.from_tableau_and_state(sim, opt.p)


def test_tree_stab_mps_compatibility_aliases_and_tree_diagnostics():
    import quimb.tensor as qtn

    product = qtn.MPS_computational_state("10", dtype="complex128")
    opt = pepsy.TreeStabOptimizer.from_mps(product, track_truncation=True)

    assert (
        pepsy.TreeStabOptimizer.from_tableau_and_nu.__func__
        is pepsy.TreeStabOptimizer.from_tableau_and_state.__func__
    )
    assert opt.max_pauli_decomposition_qubits == opt.max_operator_qubits
    assert opt.expectation_pauli_sum([(1.0, "Z", 0), (0.5, "Z", 1)]) == pytest.approx(
        -0.5
    )
    assert opt.get_infidelities() is opt.infidelities
    assert opt.get_infidelity_samples() == []
    assert opt.truncation_report()["track_truncation"] is True


def test_tree_stab_nonclifford_one_qubit_matrix_matches_dense():
    theta = 0.37
    prep = [("h", 0), (CNOT, (0, 1))]
    opt = pepsy.TreeStabOptimizer(2).apply(prep + [(_rz(theta), 1)])

    expected = _apply_local(
        _apply_local(np.array([1, 0, 0, 0]), H, (0,), 2), CNOT, (0, 1), 2
    )
    expected = _apply_local(expected, _rz(theta), (1,), 2)
    _assert_same_state(opt.to_statevector(), expected)
    assert opt.norm() == pytest.approx(1.0)


def test_tree_stab_nonclifford_two_qubit_matrix_matches_dense():
    prep = [(H, 0), (CNOT, (0, 2)), ("t", 1)]
    opt = pepsy.TreeStabOptimizer(3).apply(prep)
    before = opt.to_statevector()
    opt.apply([(_rzz(0.5), (0, 1))])

    expected = _apply_local(before, _rzz(0.5), (0, 1), 3)
    _assert_same_state(opt.to_statevector(), expected)
    assert opt.norm() == pytest.approx(np.linalg.norm(expected))


def test_tree_stab_three_qubit_dense_operator_matches_mps_and_dense():
    """A larger generic matrix uses the bounded coefficient-frame Pauli sum."""
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    unitary, _ = np.linalg.qr(raw)
    prep = [(H, 0), (CNOT, (0, 1)), (H, 2)]
    tree = pepsy.TreeStabOptimizer(
        3,
        chi=None,
        max_operator_qubits=3,
        max_pauli_terms=64,
    ).apply(prep + [(unitary, (0, 1, 2))])
    mps = pepsy.MpsStabOptimizer(
        3,
        chi=None,
        max_pauli_decomposition_qubits=3,
        max_pauli_terms=64,
    ).apply(prep + [(unitary, (0, 1, 2))])
    before = _apply_local(
        _apply_local(
            _apply_local(np.array([1, 0, 0, 0, 0, 0, 0, 0]), H, (0,), 3),
            CNOT,
            (0, 1),
            3,
        ),
        H,
        (2,),
        3,
    )
    expected = _apply_local(before, unitary, (0, 1, 2), 3)
    _assert_same_state(tree.to_statevector(), expected)
    _assert_same_state(mps.to_statevector(), expected)
    _assert_same_state(tree.to_statevector(), mps.to_statevector())


def test_tree_stab_large_dense_operator_term_budget_fails_before_replay():
    rng = np.random.default_rng(18)
    raw = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    unitary, _ = np.linalg.qr(raw)
    opt = pepsy.TreeStabOptimizer(
        3, max_operator_qubits=3, max_pauli_terms=8
    )
    before = opt.to_statevector().copy()
    with pytest.raises(ValueError, match="max_pauli_terms=8"):
        opt.apply([(unitary, (0, 1, 2))])
    _assert_same_state(opt.to_statevector(), before)
    assert opt._queue == [(unitary, (0, 1, 2))]


@pytest.mark.parametrize("backend_cls", [pepsy.MpsStabOptimizer, pepsy.TreeStabOptimizer])
def test_stabilizer_tree_and_mps_feed_forward_uses_measurement_record(backend_cls):
    opt = backend_cls(2, seed=3)
    opt.apply([
        ("h", 0),
        ("measure", "Z", 0, -1),
        ("if", -1, 1, ("x", 1)),
    ])
    expected = np.zeros(4, dtype=complex)
    expected[3] = 1.0
    _assert_same_state(opt.to_statevector(), expected)


def test_tree_stab_nonunitary_matrix_matches_dense_without_clifford_coercion():
    probability = 0.2
    gate = (1.0 - probability) * np.eye(2, dtype=complex) + probability * X
    opt = pepsy.TreeStabOptimizer(2).apply([("h", 0), (gate, 0)])

    expected = _apply_local(
        _apply_local(np.array([1, 0, 0, 0]), H, (0,), 2), gate, (0,), 2
    )
    _assert_same_state(opt.to_statevector(), expected)
    assert opt.norm() == pytest.approx(np.linalg.norm(expected))


def test_tree_stab_dense_matrix_budget_is_checked_before_decomposition():
    gate = np.eye(8, dtype=complex)
    gate[0, 0] = 0.5
    opt = pepsy.TreeStabOptimizer(3, max_operator_qubits=2)
    before = opt.to_statevector()

    with pytest.raises(ValueError, match="max_operator_qubits=2"):
        opt.apply([(gate, tuple(range(3)))])

    _assert_same_state(opt.to_statevector(), before)
    assert opt._queue == [(gate, tuple(range(3)))]


def test_tree_stab_accepts_mps_decomposition_budget_alias():
    opt = pepsy.TreeStabOptimizer(2, max_pauli_decomposition_qubits=1)
    assert opt.max_operator_qubits == 1
    with pytest.raises(ValueError, match="only one of"):
        pepsy.TreeStabOptimizer(
            2,
            max_operator_qubits=1,
            max_pauli_decomposition_qubits=1,
        )


def test_tree_stab_exact_cooling_preserves_state_and_avoids_bond_growth():
    import quimb.tensor as qtn

    pivot = np.array([1.0, 0.0], dtype=complex)
    magic = np.array([np.cos(0.19), -1j * np.sin(0.19)], dtype=complex)
    state = qtn.MPS_product_state([pivot, magic])
    stream = [("h", 0), ("cnot", 0, 1), ("rz", 0.37, 1)]

    cooled = pepsy.TreeStabOptimizer.from_mps(
        state.copy(), chi=64, exact_cooling=True
    ).apply(stream)
    plain = pepsy.TreeStabOptimizer.from_mps(
        state.copy(), chi=64, exact_cooling=False
    ).apply(stream)

    assert len(cooled.exact_cooling_events) == 1
    assert cooled.p.max_bond() == 1
    assert plain.p.max_bond() == 2
    _assert_same_state(cooled.to_statevector(), plain.to_statevector())


def test_tree_stab_exact_cooling_falls_back_without_a_stabilizer_pivot():
    import quimb.tensor as qtn

    magic_a = np.array([np.cos(0.19), -1j * np.sin(0.19)], dtype=complex)
    magic_b = np.array([np.cos(0.31), -1j * np.sin(0.31)], dtype=complex)
    state = qtn.MPS_product_state([magic_a, magic_b])
    stream = [("h", 0), ("cnot", 0, 1), ("rz", 0.37, 1)]

    cooled = pepsy.TreeStabOptimizer.from_mps(state.copy()).apply(stream)
    plain = pepsy.TreeStabOptimizer.from_mps(
        state.copy(), exact_cooling=False
    ).apply(stream)

    assert cooled.exact_cooling_events == []
    assert cooled.p.max_bond() == plain.p.max_bond() == 2
    _assert_same_state(cooled.to_statevector(), plain.to_statevector())


def test_tree_stab_greedy_cliffords_reduce_tree_bonds_without_physical_change():
    sim = _tree_stab_entangled_coefficient_state()
    before = sim.to_statevector()

    moves = sim.disentangle_cliffords(bonds=(0, 1))

    assert len(moves) == 1
    assert moves[0]["bond"] == (0, 1)
    assert moves[0]["tree_path"] == (0, 2, 1)
    assert moves[0]["score_after"] < moves[0]["score_before"]
    assert sim.p.max_bond() == 1
    assert sim.bond_history == [2, 1]
    _assert_same_state(sim.to_statevector(), before)


def test_tree_stab_disentangle_stream_event_is_caller_scheduled():
    theta = 0.37
    sim = _tree_stab_entangled_coefficient_state()
    before = sim.to_statevector()
    expected = _apply_local(before, _rz(theta), (0,), 2)

    sim.apply([
        ("disentangle", {"sweeps": 1, "bonds": ((0, 1),)}),
        ("rz", theta, 0),
    ])

    assert len(sim.disentangle_events) == 1
    assert sim.p.max_bond() == 1
    _assert_same_state(sim.to_statevector(), expected)

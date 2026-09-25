"""Tree measurement and control regression tests."""


import numpy as np
import pytest
import quimb.tensor as qtn
from pepsy.optimizers.tree import SubTreeMPO
from pepsy.optimizers.tree import TreeOptimizer
from pepsy.optimizers.tree import TreePlan
from pepsy.optimizers.tree import TreeTensorNetwork


from _tree_test_helpers import (
    _exact_state,
    _fidelity,
    _rand_unitary,
    _random_stream,
    _sv_apply_kq,
    _two_branch_flip_submpo,
)


pytestmark = [pytest.mark.integration, pytest.mark.optional, pytest.mark.tree]


def test_measure_born_statistics_and_collapse():
    """Measurement samples the Born rule and collapses to a unit-norm state."""
    theta = 0.7
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    ry = np.array([[c, -s], [s, c]], dtype=complex)
    base = TreeOptimizer([(ry, 0)], n=3, chi=4, seed=0)

    n_shots = 3000
    ones = sum(base.copy().measure(0) for _ in range(n_shots))
    assert abs(ones / n_shots - s**2) < 0.03  # p(1) = sin^2(theta/2)

    forced = base.copy()
    assert forced.measure(0, outcome=0) == 0
    assert abs(forced.norm() - 1.0) < 1e-9


def test_reset_forces_ground_state():
    """reset() returns a qubit to |0> regardless of its prior value."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt = TreeOptimizer([(x, 1)], n=3, chi=4, seed=1)  # qubit 1 in |1>
    opt.reset(1)
    assert _fidelity(opt.to_dense(), np.array([1.0] + [0.0] * 7)) > 1 - 1e-9
    assert abs(opt.norm() - 1.0) < 1e-9


def test_measure_is_seed_reproducible():
    """Two optimizers with the same seed measure the same outcome."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2)
    a = TreeOptimizer([(h, 0)], n=2, seed=42).measure(0)
    b = TreeOptimizer([(h, 0)], n=2, seed=42).measure(0)
    assert a == b


def test_tree_stream_measure_and_reset_match_mps_event_contract():
    """TTN streams accept Pauli measurement/reset events and record outcomes."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt = TreeOptimizer(
        [(x, 0), ("measure", "Z", 0, -1), ("reset", 0)],
        n=2,
        chi=4,
        seed=12,
    )

    assert len(opt.measurements) == 1
    pauli, where, outcome, probability = opt.measurements[0]
    assert (pauli, where, outcome) == ("Z", (0,), -1)
    assert probability == pytest.approx(1.0)
    assert _fidelity(opt.to_dense(), np.array([1.0, 0.0, 0.0, 0.0])) > 1 - 1e-12
    assert opt.event_types == ["gate", "measure", "reset"]


def test_tree_stream_measure_reset_records_then_prepares_pauli_state():
    """measure_reset records the result and leaves the + Pauli eigenstate."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    opt = TreeOptimizer(
        [(h, 0), TreeOptimizer.measure_reset_event("X", 0, +1)],
        n=1,
        chi=4,
    )

    assert opt.measurements == [("X", (0,), +1, pytest.approx(1.0))]
    assert _fidelity(
        opt.to_dense(), np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    ) > 1 - 1e-12
    assert opt.norm() == pytest.approx(1.0)


def test_tree_stream_multisite_pauli_measurement():
    """A product-Pauli event can collapse a multi-qubit tree subtree."""
    opt = TreeOptimizer([("measure", "ZZ", (0, 1), +1)], n=2, chi=4)

    assert opt.measurements[0][:3] == ("ZZ", (0, 1), +1)
    assert opt.measurements[0][3] == pytest.approx(1.0)
    assert opt.norm() == pytest.approx(1.0)


def test_multisite_pauli_measurement_preserves_parity_sector_coherence():
    """Parity projection must not collapse the individual Pauli outcomes."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer([(h, 0), (cnot, (0, 1))], n=2, chi=8)

    opt._measure_pauli("ZZ", (0, 1), +1)
    assert _fidelity(
        opt.to_dense(), np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
    ) > 1 - 1e-12


def test_wide_pauli_measurement_avoids_dense_projector():
    """A wide product-Pauli event remains factorized past the dense limit."""
    n = 9
    opt = TreeOptimizer(
        [("measure", "Z" * n, tuple(range(n)), +1)], n=n, chi=4
    )
    assert opt.measurements[0][2] == +1
    assert opt.norm() == pytest.approx(1.0)


def test_default_dense_operator_guard():
    """General dense operators have a finite default support limit."""
    opt = TreeOptimizer(None, n=9, run=False)
    assert opt.max_operator_qubits == 8
    with pytest.raises(MemoryError, match="max_operator_qubits"):
        opt.apply_gate(np.eye(2**9, dtype=complex), tuple(range(9)))


def test_tree_stream_control_mapping_and_cap_event():
    """Mapping controls and MPS-compatible cap events work on a tree."""
    opt = TreeOptimizer(
        [{"kind": "measure", "pauli": "Z", "where": 0, "outcome": +1}],
        n=1,
    )
    assert opt.measurements[0][:3] == ("Z", (0,), +1)

    capped = TreeOptimizer(
        [TreeOptimizer.cap_event(0, [1.0, 0.0])], n=2, chi=4
    )
    assert capped.n == capped.plan.n == capped.tn.nqubits == 1
    assert np.allclose(capped.to_dense(), [1.0, 0.0])

    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    shifted = TreeOptimizer(
        [TreeOptimizer.cap_event(1, [1.0, 0.0]), (x, 1)],
        n=3,
        chi=4,
    )
    assert shifted.n == 2
    assert np.argmax(np.abs(shifted.to_dense())) == 1


def test_tree_cap_matches_dense_contraction_and_compacts_plan():
    """Capping an entangled leaf matches dense contraction and keeps a valid TTN."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        [(h, 0), (cnot, (0, 1))],
        n=4,
        tree=TreePlan.from_order(range(4), structure="balanced"),
        chi=8,
    )
    before = opt.to_dense().reshape((2,) * 4)
    vec = np.array([1.0, 2.0], dtype=complex)
    expected = np.tensordot(vec, before, axes=(0, 1)).reshape(-1)

    opt.cap(1, vec)

    assert opt.n == opt.plan.n == opt.tn.nqubits == 3
    assert np.allclose(opt.to_dense(), expected)
    assert opt.tn.validate(check_canonical=True) is opt.tn
    assert set(opt.plan.leaf_of_qubit) == {0, 1, 2}


def test_tree_public_submpo_and_pauli_backend_operations():
    """Public tree operator primitives cover native MPO and Pauli paths."""
    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    mpo = _two_branch_flip_submpo(L=n, sites=(0, 3), targets=(0, 3))
    opt = TreeOptimizer(None, n=n, tree=plan, chi=16, run=False)

    opt.apply_submpo(mpo, (0, 3))
    assert np.allclose(
        opt.to_dense(),
        0.7 * np.eye(16, dtype=complex)[:, 0]
        + 0.3 * _sv_apply_kq(
            np.eye(16, dtype=complex)[:, 0],
            np.kron(np.array([[0, 1], [1, 0]], dtype=complex),
                    np.array([[0, 1], [1, 0]], dtype=complex)),
            (0, 3),
            n,
        ),
    )

    rotated = TreeOptimizer(None, n=n, tree=plan, chi=16, run=False)
    theta = 0.37
    rotated.apply_pauli_rotation(theta, "XZ", (0, 3))
    pauli = np.kron(
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    )
    expected = (
        np.cos(theta / 2.0) * np.eye(4, dtype=complex)
        - 1j * np.sin(theta / 2.0) * pauli
    )
    assert np.allclose(
        rotated.to_dense(),
        _sv_apply_kq(np.eye(16, dtype=complex)[:, 0], expected, (0, 3), n),
    )

    summed = TreeOptimizer(None, n=n, tree=plan, chi=16, run=False)
    summed.apply_pauli_sum([(1.0, {0: "X", 3: "X"})])
    assert np.allclose(
        summed.to_dense(),
        _sv_apply_kq(
            np.eye(16, dtype=complex)[:, 0],
            np.kron(
                np.array([[0, 1], [1, 0]], dtype=complex),
                np.array([[0, 1], [1, 0]], dtype=complex),
            ),
            (0, 3),
            n,
        ),
    )


def test_tree_pauli_sum_routes_only_active_support_over_steiner_subtree():
    """Sparse Pauli TreeMPOs must not turn support into a chain window."""
    plan = TreePlan.from_order(range(8), structure="balanced")
    active = (0, 2, 7)
    opt = TreeOptimizer(
        None, n=8, tree=plan, chi=8, cutoff=0.0, profile=True, run=False
    )

    opt.apply_pauli_sum([
        (0.7, {0: "X", 2: "Y", 7: "Z"}),
        (0.2, {0: "Z", 7: "X"}),
    ])

    assert opt.update_history[-1]["support"] == active
    routes = [
        event for event in opt.profile_events
        if event.get("route") == "subtreempo"
        and event.get("kind") == "metadata_path"
    ]
    assert routes
    assert routes[-1]["support"] == active
    assert routes[-1]["subtree_nodes"] == len(
        opt._steiner_nodes([opt.plan.node_of_qubit[q] for q in active])
    )
    assert opt.tn.validate(check_canonical=True) is opt.tn


@pytest.mark.parametrize("operation", ["sum", "rotation", "projector"])
def test_tree_pauli_operator_helpers_use_compact_subtreempo(monkeypatch, operation):
    """Dense Pauli helpers lower to the same compact tree operator route."""
    plan = TreePlan.from_order(range(8), structure="balanced")
    opt = TreeOptimizer(
        None, n=8, tree=plan, chi=8, cutoff=0.0, mode="direct", run=False,
    )
    seen = []
    apply = opt.apply_sub_mpotree

    def traced(operator, *args, **kwargs):
        seen.append(operator)
        return apply(operator, *args, **kwargs)

    monkeypatch.setattr(opt, "apply_sub_mpotree", traced)
    if operation == "sum":
        opt.apply_pauli_sum([
            (0.7, {0: "X", 2: "Y", 7: "Z"}),
            (0.2, {0: "Z", 7: "X"}),
        ])
    elif operation == "rotation":
        opt.apply_pauli_rotation(0.37, "XYZ", (0, 2, 7))
    else:
        opt.project_pauli("XYZ", (0, 2, 7), +1)

    assert seen
    assert all(isinstance(operator, SubTreeMPO) for operator in seen)
    assert all(operator.num_tensors < len(plan.nodes()) for operator in seen)
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_tree_two_site_numpy_mpo_is_coerced_to_cupy_state_backend():
    """Two-site MPO factors follow a CuPy TTN without mutating the MPO."""
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed without a CUDA device.")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy CUDA runtime unavailable: {exc}")

    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(
        lambda array: cupy.asarray(array, dtype=cupy.complex64)
    )
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    mpo = qtn.MatrixProductOperator.from_dense(
        np.kron(x, x), dims=(2, 2), sites=(0, 1), L=n,
    )
    before = [tensor.data.copy() for tensor in mpo.tensors]
    opt = TreeOptimizer(
        None, state=state, tree=plan, chi=8, cutoff=0.0, run=False,
    )

    with pytest.warns(UserWarning, match="converting a gate/operator payload"):
        opt.apply_submpo(mpo, (0, 1))

    assert opt.backend_info() == {
        "backend": "cupy",
        "dtype": "complex64",
        "device": str(cupy.cuda.Device()),
    }
    expected = np.zeros(2**n, dtype=np.complex64)
    expected[12] = 1.0  # X_0 X_1 |0000> = |1100>
    np.testing.assert_allclose(opt.to_dense(), expected, atol=1e-5)
    assert opt.tn.validate() is opt.tn
    for tensor, original in zip(mpo.tensors, before):
        np.testing.assert_array_equal(tensor.data, original)


def test_tree_two_site_cupy_gate_does_not_unwrap_memory_pointer():
    """A CuPy dense gate reaches TreeMPO factorization as an array."""
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed without a CUDA device.")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy CUDA runtime unavailable: {exc}")

    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(
        lambda array: cupy.asarray(array, dtype=cupy.complex128)
    )
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    gate = cupy.asarray(np.kron(x, x), dtype=cupy.complex128)
    opt = TreeOptimizer(
        [(gate, (0, 1))],
        n=n,
        tree=plan,
        state=state,
        chi=8,
        cutoff=0.0,
        run=False,
    )

    opt.run(progbar=False)

    expected = np.zeros(2**n, dtype=np.complex128)
    expected[12] = 1.0
    np.testing.assert_allclose(opt.to_dense(), expected, atol=1e-12)


def test_tree_expectation_mpo_is_batched_and_non_mutating():
    """A structured MPO expectation uses one tree pass and preserves state."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    zz = np.diag([1.0, -1.0, -1.0, 1.0])
    mpo = qtn.MatrixProductOperator.from_dense(
        zz, dims=(2, 2), sites=(0, 1), L=4,
    )
    opt = TreeOptimizer([(h, 0), (cnot, (0, 1))], n=4, chi=16)
    before = opt.to_dense().copy()

    value = opt.expectation_mpo(mpo, (0, 1), max_bond=16)

    assert value == pytest.approx(1.0)
    assert np.allclose(opt.to_dense(), before)
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_tree_expectation_mpo_reports_private_ket_truncation():
    """Expectation diagnostics expose accidental finite-cap truncation."""
    identity = np.eye(4, dtype=complex)
    mpo = qtn.MatrixProductOperator.from_dense(
        identity, dims=(2, 2), sites=(0, 1), L=4,
    )
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer([(h, 0), (cnot, (0, 1))], n=4, chi=16)

    with pytest.warns(UserWarning, match="private transformed ket"):
        value, diagnostics = opt.expectation_mpo(
            mpo, (0, 1), max_bond=1, return_diagnostics=True,
        )

    assert diagnostics["truncated"] is True
    assert diagnostics["n_truncated"] >= 1
    assert diagnostics["max_bond"] == 1
    assert value != pytest.approx(1.0)

    exact_value, exact_diagnostics = opt.expectation_mpo(
        mpo, (0, 1), max_bond=16, return_diagnostics=True,
    )
    assert exact_value == pytest.approx(1.0)
    assert exact_diagnostics["truncated"] is False


def test_tree_exact_mpo_expectation_keeps_mpo_separate(monkeypatch):
    """Exact MPO readout does not lower or compress the tree state."""
    identity = np.eye(4, dtype=complex)
    mpo = qtn.MatrixProductOperator.from_dense(
        identity, dims=(2, 2), sites=(0, 1), L=4,
    )
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer([(h, 0), (cnot, (0, 1))], n=4, chi=1)
    before = opt.to_dense().copy()
    before_bond = opt.tn.max_bond()

    def forbid_dense(*args, **kwargs):
        raise AssertionError("exact MPO readout must not call MPO.to_dense()")

    monkeypatch.setattr(qtn.MatrixProductOperator, "to_dense", forbid_dense)
    value = opt.expectation_mpo_exact(mpo, (0, 1))

    assert value == pytest.approx(1.0)
    assert opt.tn.max_bond() == before_bond
    assert np.allclose(opt.to_dense(), before)


def test_tree_rejects_native_mpo_on_dense_state():
    """Native and ordinary tensor backends cannot be mixed silently."""
    class NativeMarker:
        pepsy_tree_native = True

    opt = TreeOptimizer(None, n=2, run=False)
    with pytest.raises(TypeError, match="native Symmray MPO"):
        opt.apply_submpo(NativeMarker(), (0, 1))


def test_tree_subtree_route_batches_sibling_messages(monkeypatch):
    """Independent leaf messages landing at one node use one contraction."""
    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    opt = TreeOptimizer(
        None, n=n, tree=plan, chi=16, cutoff=0.0,
        subtree_workers=2, run=False,
    )
    identity = np.eye(2**n, dtype=complex)
    before = opt.to_dense().copy()
    original_contract = qtn.tensor_contract
    grouped_calls = []

    def traced_contract(*tensors, **kwargs):
        if len(tensors) >= 3:
            grouped_calls.append(len(tensors))
        return original_contract(*tensors, **kwargs)

    monkeypatch.setattr(qtn, "tensor_contract", traced_contract)
    opt.apply_subtree_operator(identity, tuple(range(n)))

    assert grouped_calls
    assert np.allclose(opt.to_dense(), before)
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_tree_profile_report_is_opt_in():
    """Kernel timings are empty by default and available when requested."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    quiet = TreeOptimizer([(x, 0)], n=4, chi=4)
    quiet_report = quiet.profile_report()
    assert quiet_report["enabled"] is False
    assert quiet_report["events"] == []
    assert quiet_report["by_kind"] == {}
    assert quiet_report["native_compression_routes"] == {}
    assert quiet_report["update_seconds"] == 0.0
    assert quiet_report["total_seconds"] == 0.0
    assert quiet_report["timing_semantics"][
        "total_seconds_is_sum_of_events_not_wall_time"
    ] is True

    profiled = TreeOptimizer(
        [(x, 0), (cnot, (0, 3))], n=4, chi=4, profile=True,
    )
    report = profiled.profile_report()
    assert report["enabled"] is True
    assert report["events"]
    assert report["by_kind"]["update"]["count"] == 2
    assert report["native_compression_routes"] == {}
    assert report["by_kind"]["gate_factorization"]["count"] == 1
    assert report["by_kind"]["tensor_absorption"]["count"] >= 2
    assert report["by_kind"]["metadata_path"]["count"] == 1
    assert report["by_kind"]["center_movement"]["count"] >= 1
    assert report["update_seconds"] == report["by_kind"]["update"]["seconds"]
    assert report["total_seconds"] > 0.0


def test_tree_profile_separates_qr_thread_hops_from_compression():
    """Profiled direct routing exposes exact QR hops as separate events."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        [(cnot, (0, 3))], n=4, chi=2, cutoff=0.0,
        profile=True, track_bond_diagnostics=True,
    )

    report = opt.profile_report()
    hops = [event for event in report["events"] if event["kind"] == "thread_hop"]
    assert hops
    assert all(event["seconds"] >= 0.0 for event in hops)
    assert report["by_kind"]["thread_hop"]["count"] == len(hops)
    assert report["by_kind"]["edge_canonize"]["count"] >= 1


def test_tree_bond_diagnostics_distinguish_transient_qr_growth():
    """Temporary gate/QR growth may exceed chi while live bonds do not."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        [(cnot, (0, 3))], n=4, chi=1, cutoff=0.0,
        record_history=False, track_bond_diagnostics=True,
    )

    report = opt.bond_diagnostic_report()
    assert report["enabled"] is True
    assert report["max_transient_bond"] >= 2
    assert report["max_live_bond_after"] <= 1
    assert report["n_transient_exceeds_chi"] >= 1
    update = report["updates"][0]
    assert update["transient_max_bond"] > update["live_max_bond_after"]
    assert update["transient_exceeds_chi"] is True
    assert update["bond_trace"]


def test_tree_norm_and_fidelity_check_is_deterministic_without_network_fidelity():
    """Small exact replay uses local norm plus a deterministic statevector oracle."""
    rng = np.random.default_rng(417)
    stream = _random_stream(4, 10, rng, two_qubit_frac=0.8)
    exact = _exact_state(stream, 4)

    first = TreeOptimizer(
        stream, n=4, chi=64, cutoff=0.0, threads=1,
    )
    second = TreeOptimizer(
        stream, n=4, chi=64, cutoff=0.0, threads=2,
    )
    first_dense = first.to_dense()
    second_dense = second.to_dense()

    assert first.norm() == pytest.approx(np.linalg.norm(first_dense))
    assert second.norm() == pytest.approx(np.linalg.norm(second_dense))
    assert _fidelity(exact, first_dense) > 1.0 - 1e-10
    assert _fidelity(exact, second_dense) > 1.0 - 1e-10
    assert _fidelity(first_dense, second_dense) > 1.0 - 1e-12


def test_tree_pauli_expectation_and_projection_are_public():
    """Pauli expectation/projection share the measurement backend semantics."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer([(h, 0), (cnot, (0, 1))], n=2, chi=8)
    assert opt.expectation_pauli("ZZ", (0, 1)) == pytest.approx(1.0)
    assert opt.expectation_pauli("XX", (0, 1)) == pytest.approx(1.0)
    opt.project_pauli("ZZ", (0, 1), +1)
    assert opt.expectation_pauli("ZZ", (0, 1)) == pytest.approx(1.0)


def test_tree_sync_canonicalization_rebuilds_state_owned_center():
    """Tree recovery clears stale lower-level canonical-region metadata."""
    opt = TreeOptimizer([], n=4, chi=8, run=False)
    opt.tn.canonize_around_qubits_((0, 3))

    center = opt.sync_canonicalization()

    assert center == opt.plan.root
    assert opt.center == opt.tn.orthogonality_center == center
    assert opt.is_canonical_form(center)


def test_tree_public_pauli_measurement_returns_probability_and_diagnostics():
    """The public Pauli measurement API exposes Born probability diagnostics."""
    theta = 0.8
    ry = np.array([
        [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
        [np.sin(theta / 2.0), np.cos(theta / 2.0)],
    ], dtype=complex)
    opt = TreeOptimizer([(ry, 0)], n=5, chi=8)

    outcome, probability, diagnostics = opt.measure_pauli(
        "Z", 0, outcome=+1, return_diagnostics=True
    )

    assert outcome == +1
    assert probability == pytest.approx(np.cos(theta / 2.0) ** 2)
    assert diagnostics["probability"] == pytest.approx(probability)
    assert diagnostics["norm_before"] == pytest.approx(1.0)
    assert diagnostics["norm_after"] == pytest.approx(1.0)
    assert diagnostics["support"] == (0,)
    assert diagnostics["span_before"] == diagnostics["span_after"]
    assert diagnostics["bonds_before"] == diagnostics["bonds_after"]
    assert opt.expectation_pauli("Z", 0) == pytest.approx(1.0)


def test_tree_pauli_projection_can_preserve_branch_norm():
    """A non-normalizing Pauli projection retains its physical survival norm."""
    theta = 0.9
    ry = np.array([
        [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
        [np.sin(theta / 2.0), np.cos(theta / 2.0)],
    ], dtype=complex)
    opt = TreeOptimizer([(ry, 0)], n=4, chi=8)

    diagnostics = opt.project_pauli(
        "Z", 0, +1, renormalize=False, return_diagnostics=True
    )
    expected_probability = np.cos(theta / 2.0) ** 2

    assert diagnostics["renormalized"] is False
    assert diagnostics["norm_after"] == pytest.approx(
        np.sqrt(expected_probability)
    )
    assert diagnostics["norm_ratio"] == pytest.approx(
        np.sqrt(expected_probability)
    )
    assert opt.expectation_pauli("Z", 0) == pytest.approx(1.0)
    assert opt.get_projection_diagnostics()[-1] is diagnostics


def test_tree_sparse_long_pauli_avoids_dense_operator_limit():
    """A long sparse Pauli measurement uses the factorized tree path."""
    n = 17
    where = (0, 3, 7, 11, 16)
    opt = TreeOptimizer(None, n=n, chi=8, max_operator_qubits=2, run=False)

    outcome, probability, diagnostics = opt.measure_pauli(
        "ZZZZZ", where, outcome=+1, return_diagnostics=True
    )

    assert outcome == +1
    assert probability == pytest.approx(1.0)
    assert diagnostics["support"] == where
    assert diagnostics["max_bond_after"] <= 8
    assert opt.norm() == pytest.approx(1.0)


def test_tree_cap_can_preserve_stable_logical_labels():
    """Stable-label caps compact storage but preserve caller-facing IDs."""
    opt = TreeOptimizer(None, n=4, chi=8, run=False)

    opt.cap(1, [1.0, 0.0], stable_labels=True)

    assert opt.n == 3
    assert opt.qubits == [0, 2, 3]
    assert opt.logical_order == [0, 2, 3]
    assert opt.position(2) == 1
    assert opt.logical_site(1) == 2

    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt.apply_1q(x, 3)
    assert np.argmax(np.abs(opt.to_dense())) == 1


def test_tree_stable_labels_work_for_pauli_projection_paths():
    """Pauli projection resolves stable labels once, then uses compact sites."""
    opt = TreeOptimizer(None, n=4, chi=8, run=False)
    opt.cap(1, [1.0, 0.0], stable_labels=True)

    diagnostics = opt.project_pauli(
        "Z", 2, +1, renormalize=False, return_diagnostics=True
    )

    assert diagnostics["support"] == (2,)
    assert diagnostics["norm_after"] == pytest.approx(1.0)
    assert opt.expectation_pauli("Z", 2) == pytest.approx(1.0)


def test_tree_stable_cap_event_supports_later_logical_events():
    """A stable-label cap event keeps later stream labels addressable."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt = TreeOptimizer(
        [
            TreeOptimizer.cap_event(
                1, [1.0, 0.0], compact_labels=False
            ),
            (x, 3),
        ],
        n=4,
        chi=8,
    )

    assert opt.qubits == [0, 2, 3]
    assert np.argmax(np.abs(opt.to_dense())) == 1


def test_tree_reset_reuses_an_ancilla_without_disturbing_data():
    """Reset and repeated use of an ancilla leave the data Bell pair intact."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt = TreeOptimizer([(h, 0), (cnot, (0, 1)), (x, 2)], n=3, chi=8)

    opt.reset(2)
    assert opt.expectation_pauli("Z", 2) == pytest.approx(1.0)
    opt.apply_1q(x, 2)
    opt.reset(2)

    assert opt.expectation_pauli("ZZ", (0, 1)) == pytest.approx(1.0)
    assert opt.expectation_pauli("XX", (0, 1)) == pytest.approx(1.0)
    assert opt.expectation_pauli("Z", 2) == pytest.approx(1.0)


def test_tree_stream_submpo_markers_use_recursive_operator_path():
    """Tuple and mapping sub-MPO markers match their dense support operator."""
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    dense = np.asarray(mpo.to_dense())

    event = TreeOptimizer.submpo_event(mpo, (0, 3))
    tuple_opt = TreeOptimizer([event], n=4, chi=8)
    mapping_opt = TreeOptimizer(
        [{"kind": "submpo", "mpo": mpo, "where": [0, 3]}],
        n=4,
        chi=8,
    )
    expected = _sv_apply_kq(np.eye(16, dtype=complex)[:, 0], dense, (0, 3), 4)

    assert tuple_opt.event_types == ["submpo"]
    assert TreeOptimizer.is_submpo_event(event)
    assert TreeOptimizer.submpo_event_parts(event)[1] == (0, 3)
    assert tuple_opt.update_history[0]["kind"] == "submpo"
    assert _fidelity(expected, tuple_opt.to_dense()) > 1 - 1e-10
    assert _fidelity(tuple_opt.to_dense(), mapping_opt.to_dense()) > 1 - 1e-10


def test_tree_stream_submpo_does_not_require_dense_materialization(monkeypatch):
    """Native MPO replay and bond estimation avoid ``to_dense`` allocation."""
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    expected = np.asarray(mpo.to_dense())

    def fail_to_dense():
        raise AssertionError("sub-MPO was unexpectedly materialized")

    monkeypatch.setattr(mpo, "to_dense", fail_to_dense)
    event = TreeOptimizer.submpo_event(mpo, (0, 3))
    opt = TreeOptimizer([event], n=4, chi=8)
    report = opt.estimate_bonds([event])

    reference = _sv_apply_kq(
        np.eye(16, dtype=complex)[:, 0], expected, (0, 3), 4
    )
    assert _fidelity(reference, opt.to_dense()) > 1 - 1e-10
    assert report["events"][0]["crossing_edges"]


def test_tree_native_submpo_keeps_unacted_physical_root_structured(monkeypatch):
    """A physical root on the Steiner subtree need not be an MPO site."""
    where = (1, 2, 3)
    plan = TreePlan.from_order(
        range(1, 5), structure="balanced", root_qubit=0
    )
    gate = _rand_unitary(len(where), np.random.default_rng(53))
    mpo = qtn.MatrixProductOperator.from_dense(
        gate.reshape((2,) * (2 * len(where))),
        dims=(2,) * len(where),
        sites=where,
        L=5,
        max_bond=None,
        cutoff=0.0,
    )
    expected = _sv_apply_kq(
        np.eye(2**5, dtype=complex)[:, 0], gate, where, 5
    )

    def fail_to_dense():
        raise AssertionError("sub-MPO was unexpectedly materialized")

    monkeypatch.setattr(mpo, "to_dense", fail_to_dense)
    opt = TreeOptimizer(
        None, n=5, tree=plan, chi=64, cutoff=0.0, run=False
    )
    opt.apply_submpo(mpo, where)

    assert _fidelity(expected, opt.to_dense()) > 1 - 1e-10


def test_tree_submpo_mode_declares_and_validates_mpo_streams():
    """The explicit sub-MPO mode accepts MPO events and rejects dense gates."""
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    opt = TreeOptimizer(None, n=4, chi=8, mode="submpo", run=False)
    opt.run([TreeOptimizer.submpo_event(mpo, (0, 3))])
    assert opt.mode == "submpo"

    dense = np.asarray(mpo.to_dense())
    ordinary = TreeOptimizer(None, n=4, chi=8, mode="submpo", run=False)
    with pytest.raises(ValueError, match="requires explicit sub-MPO"):
        ordinary.run([(dense, (0, 3))])


def test_tree_estimate_bonds_includes_submpo_operator_schmidt_rank():
    """Sub-MPO markers participate in the same conservative bond estimate."""
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    plan = TreePlan.from_order(range(4), structure="balanced")
    opt = TreeOptimizer(None, n=4, tree=plan, chi=1, run=False)

    report = opt.estimate_bonds([("submpo", mpo, (0, 3))])

    assert report["events"][0]["kind"] == "submpo"
    assert set(report["events"][0]["crossing_edges"].values()) == {2}
    assert report["max_bond"] == 2
    assert report["requires_truncation"]
    with pytest.raises(MemoryError, match="max_operator_qubits"):
        opt.preflight(
            [("submpo", mpo, (0, 3))],
            max_operator_qubits=1,
        )


def test_measurement_enforces_max_subtree_nodes_on_streamed_product_pauli():
    """Control-event execution applies the same subtree guard as dense gates."""
    opt = TreeOptimizer(None, n=8, max_subtree_nodes=1, run=False)
    with pytest.raises(MemoryError, match="max_subtree_nodes"):
        opt.run([{
            "kind": "measure",
            "pauli": "ZZ",
            "where": [0, 7],
            "outcome": +1,
        }])

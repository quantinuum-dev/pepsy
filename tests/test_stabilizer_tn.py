"""Tests for the stabilizer-tensor-network state container and Clifford update.

Covers Phase 1 (state container + statevector reconstruction) and Phase 2
(Clifford update: basis-only, |nu> unchanged) of the STN build
(arXiv:2403.08724).
"""

import sys
import types

import numpy as np
import pytest

stim = pytest.importorskip("stim")
qtn = pytest.importorskip("quimb.tensor")

from pepsy.optimizers.stabilizer_tn import STNState


def _fidelity(a, b):
    """State overlap magnitude (1.0 iff equal up to global phase)."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def _stim_reference_state(n, gates):
    """Reference statevector from an independent stim TableauSimulator."""
    sim = stim.TableauSimulator()
    sim.set_num_qubits(n)
    method = {
        "h": "h", "x": "x", "y": "y", "z": "z", "s": "s", "sdg": "s_dag",
        "cnot": "cnot", "cx": "cnot", "cy": "cy", "cz": "cz", "swap": "swap",
    }
    for name, *targets in gates:
        getattr(sim, method[name])(*targets)
    return np.asarray(sim.state_vector(endian="big")).reshape(-1)


def _random_clifford_circuit(n, depth, seed):
    rng = np.random.default_rng(seed)
    one_q = ["h", "s", "sdg", "x", "y", "z"]
    two_q = ["cnot", "cz", "swap"]
    gates = []
    for _ in range(depth):
        if n >= 2 and rng.random() < 0.4:
            a, b = rng.choice(n, size=2, replace=False)
            gates.append((rng.choice(two_q), int(a), int(b)))
        else:
            q = int(rng.integers(n))
            gates.append((rng.choice(one_q), q))
    return gates


def test_initial_state_is_all_zero():
    st = STNState(3)
    assert st.num_qubits == 3
    assert st.max_bond() == 1
    # coefficient MPS p = |000> = e0
    np.testing.assert_allclose(st.p_dense(), np.eye(8)[0])
    # |psi> = C p = |000>
    assert _fidelity(st.to_statevector(), np.eye(8)[0]) == pytest.approx(1.0)
    assert st.pseudo_stabilizer_rank() == 1


def test_bell_state_ground_truth():
    """H(0), CNOT(0,1) prepares (|00> + |11>)/sqrt(2), independent of stim."""
    st = STNState(2).h(0).cnot(0, 1)
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert _fidelity(st.to_statevector(), expected) == pytest.approx(1.0)
    # Clifford gates do not touch the coefficient MPS p.
    assert st.max_bond() == 1
    np.testing.assert_allclose(st.p_dense(), np.eye(4)[0])


def test_ghz_state_ground_truth():
    st = STNState(3).h(0).cnot(0, 1).cnot(1, 2)
    expected = np.zeros(8, dtype=complex)
    expected[0] = expected[7] = 1 / np.sqrt(2)
    assert _fidelity(st.to_statevector(), expected) == pytest.approx(1.0)
    assert st.max_bond() == 1


def test_apply_clifford_circuit_matches_stim_reference():
    gates = [("h", 0), ("s", 1), ("cnot", 0, 1), ("z", 2), ("cz", 1, 2),
             ("swap", 0, 2), ("sdg", 0)]
    st = STNState(3).apply_clifford_circuit(gates)
    ref = _stim_reference_state(3, gates)
    assert _fidelity(st.to_statevector(), ref) == pytest.approx(1.0)
    # basis-only updates keep |nu> trivial
    assert st.max_bond() == 1
    assert st.pseudo_stabilizer_rank() == 1


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_random_clifford_matches_stim_and_keeps_chi_one(seed):
    n, depth = 4, 30
    gates = _random_clifford_circuit(n, depth, seed)
    st = STNState(n).apply_clifford_circuit(gates)
    ref = _stim_reference_state(n, gates)
    assert _fidelity(st.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)
    # Clifford-only circuits never grow the coefficient MPS.
    assert st.max_bond() == 1


def test_invalid_gate_and_qubit_raise():
    st = STNState(2)
    with pytest.raises(ValueError):
        st.apply_clifford("t", 0)          # non-Clifford / unknown
    with pytest.raises(ValueError):
        st.apply_clifford("h", 5)          # out of range
    with pytest.raises(ValueError):
        st.apply_clifford("cnot", 0)       # wrong arity


def test_zero_qubit_rejected():
    with pytest.raises(ValueError):
        STNState(0)


# --------------------------------------------------------------------------- #
# MpsStabOptimizer simulator (Clifford + non-Clifford rotations, matrices, sub-MPO)
# --------------------------------------------------------------------------- #
from pepsy.optimizers.stabilizer_tn import (  # noqa: E402
    DeferredInjectionReport,
    DeferredProjectionRecord,
    ImmediateInjectionReport,
    ImmediateProjectionRecord,
    MeasurementRecord,
    MpsStabOptimizer,
    NormEventRecord,
    StabilizerMpsSettingsAdvice,
    StabilizerMpsRunResult,
    StabilizerMpsSimulator,
    StreamAnalysisRecord,
    pauli_rotation_mpo,
    run_stabilizer_mps_stream,
)

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex)


def _rot(axis, theta):
    p = {"X": _X, "Y": _Y, "Z": _Z}[axis]
    return np.cos(theta / 2) * _I - 1j * np.sin(theta / 2) * p


def _rzz(theta):
    zz = np.kron(_Z, _Z)
    return np.cos(theta / 2) * np.eye(4) - 1j * np.sin(theta / 2) * zz


def _apply_gate_dense(psi, u, where, n):
    """Apply gate ``u`` on qubits ``where`` (big-endian) to statevector ``psi``."""
    where = list(where)
    k = len(where)
    t = psi.reshape([2] * n)
    u = u.reshape([2] * k + [2] * k)
    t = np.tensordot(u, t, axes=(list(range(k, 2 * k)), where))
    t = np.moveaxis(t, list(range(k)), where)
    return t.reshape(-1)


def _cap_dense(psi, vec, where, n):
    """Contract a qubit leg with ``vec`` and remove it from a dense state."""
    tensor = psi.reshape([2] * n)
    return np.tensordot(tensor, np.asarray(vec, dtype=complex), axes=([where], [0])).reshape(-1)


def _dense_reference(n, stream):
    """Dense statevector after applying a named gate stream to |0...0>."""
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    clifford = {"h": _H, "x": _X, "y": _Y, "z": _Z,
                "s": np.diag([1, 1j]).astype(complex),
                "sdg": np.diag([1, -1j]).astype(complex)}
    for entry in stream:
        name = entry[0]
        if name in clifford:
            psi = _apply_gate_dense(psi, clifford[name], (entry[1],), n)
        elif name == "cnot" or name == "cx":
            cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)
            psi = _apply_gate_dense(psi, cnot, (entry[1], entry[2]), n)
        elif name == "cz":
            cz = np.diag([1, 1, 1, -1]).astype(complex)
            psi = _apply_gate_dense(psi, cz, (entry[1], entry[2]), n)
        elif name in ("rx", "ry", "rz"):
            axis = {"rx": "X", "ry": "Y", "rz": "Z"}[name]
            psi = _apply_gate_dense(psi, _rot(axis, entry[1]), (entry[2],), n)
        elif name == "rzz":
            psi = _apply_gate_dense(psi, _rzz(entry[1]), (entry[2], entry[3]), n)
        elif name == "t":
            psi = _apply_gate_dense(psi, _T, (entry[1],), n)
        else:
            raise AssertionError(f"unhandled {name}")
    return psi


def _coefficient_bell_optimizer():
    """A Bell state stored in ``p`` rather than in the free tableau frame."""
    p = qtn.MatrixProductState.from_dense(
        np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2), dims=[2, 2]
    )
    tableau = stim.TableauSimulator()
    tableau.set_num_qubits(2)
    return MpsStabOptimizer.from_tableau_and_state(tableau, p, chi=2)


def test_clifford_disentangling_moves_bell_entanglement_into_tableau():
    sim = _coefficient_bell_optimizer()
    before = sim.to_statevector()

    moves = sim.disentangle_cliffords()

    assert len(moves) == 1
    assert moves[0]["bond"] == 0
    assert moves[0]["score_before"][0] == 2
    assert moves[0]["score_after"][0] == 1
    assert sim.state.max_bond() == 1
    assert _fidelity(sim.to_statevector(), before) == pytest.approx(1.0, abs=1e-12)
    # A gauge update is not physical compressed unitary evolution.
    assert sim.infidelities == []
    assert sim.bond_history == [2, 1]


def test_disentangle_stream_event_preserves_following_physical_gate_order():
    theta = 0.37
    sim = _coefficient_bell_optimizer()
    before = sim.to_statevector()
    expected = _apply_gate_dense(before, _rot("Z", theta), (0,), 2)

    sim.apply([("disentangle", {"sweeps": 1}), ("rz", theta, 0)])

    assert sim.state.max_bond() == 1
    assert _fidelity(sim.to_statevector(), expected) == pytest.approx(1.0, abs=1e-12)
    assert sim.infidelities == []
    assert len(sim.bond_history) == 3


def test_static_frame_layout_uses_dynamic_tableau_support():
    sim = MpsStabOptimizer(3)
    sim.set_gates([("cnot", 0, 2), ("rz", 0.4, 2)])

    plan = sim.current_frame_layout(order="auto")

    assert plan["kind"] == "stn_frame_layout"
    assert plan["frame_events"][0]["support"] == (0, 2)
    pos = plan["site_map"]
    assert abs(pos[0] - pos[2]) == 1
    assert plan["input_stats"]["max_event_span"] == 2
    assert plan["stats"]["max_event_span"] == 1


def test_static_frame_layout_run_matches_unlaid_reference():
    stream = [
        ("h", 0),
        ("cnot", 0, 2),
        ("rz", 0.37, 2),
        ("rx", -0.22, 1),
        ("rzz", 0.19, 0, 1),
    ]
    ref = MpsStabOptimizer(3).apply(stream)
    laid = MpsStabOptimizer(3, stream, layout="auto", layout_report=False).run()

    assert laid.logical_order != [0, 1, 2]
    assert _fidelity(laid.to_statevector(), ref.to_statevector()) == pytest.approx(
        1.0, abs=1e-12
    )


def test_static_frame_layout_absorb_measure_matches_reference():
    stream = [("h", 0), ("cnot", 0, 2), ("rz", 0.37, 2), ("ry", 0.41, 1)]
    ref = MpsStabOptimizer(3).apply(stream)
    laid = (
        MpsStabOptimizer(3)
        .set_gates(stream)
        .apply_layout([2, 0, 1], layout_report=False)
        .run()
    )

    m_ref = ref.measure("Z", 2, outcome=+1, absorb_basis=True)
    m_laid = laid.measure("Z", 2, outcome=+1, absorb_basis=True)

    assert m_laid == m_ref
    assert _fidelity(laid.to_statevector(), ref.to_statevector()) == pytest.approx(
        1.0, abs=1e-12
    )


def test_static_frame_layout_rejects_entangled_coefficient_mps():
    sim = _coefficient_bell_optimizer()
    sim.set_gates([("rz", 0.2, 0)])

    with pytest.raises(ValueError, match="product coefficient MPS"):
        sim.apply_layout([1, 0], layout_report=False)


def test_simulator_single_nonclifford_rotations_match_dense():
    for axis in ("rx", "ry", "rz"):
        for theta in (0.3, 1.1, -0.7):
            for prep in ([], [("h", 0)], [("h", 0), ("cnot", 0, 1)]):
                stream = prep + [(axis, theta, 0)]
                sim = MpsStabOptimizer(2).apply(stream)
                ref = _dense_reference(2, stream)
                assert _fidelity(sim.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


def test_simulator_full_circuit_matches_dense():
    stream = [
        ("h", 0), ("cnot", 0, 1), ("rz", 0.6, 1), ("ry", -0.4, 2),
        ("cz", 1, 2), ("rzz", 0.9, 0, 2), ("s", 0), ("rx", 0.5, 1), ("t", 2),
    ]
    sim = MpsStabOptimizer(3).apply(stream)
    ref = _dense_reference(3, stream)
    assert _fidelity(sim.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    ("pivot", "frame"),
    [
        (np.array([1, 0], dtype=complex), [("h", 0), ("cnot", 0, 1)]),
        (np.array([0, 1], dtype=complex), [("h", 0), ("cnot", 0, 1)]),
        (np.array([1, 1], dtype=complex) / np.sqrt(2), [("cnot", 0, 1)]),
        (np.array([1, 1j], dtype=complex) / np.sqrt(2), [("h", 0), ("cnot", 0, 1)]),
        (np.array([1, 0], dtype=complex), [("z", 0), ("h", 0), ("cnot", 0, 1)]),
    ],
)
def test_exact_cooling_moves_controlled_pauli_into_tableau(pivot, frame):
    # The trailing Rz maps through ``frame`` to a two-site Pauli. Site 0 is a
    # stabilizer pivot, whereas site 1 is magic so the ordinary MPO update
    # would entangle the coefficient MPS.
    theta = 0.37
    magic = np.array([np.cos(0.19), -1j * np.sin(0.19)], dtype=complex)
    p = qtn.MPS_product_state([pivot, magic])
    stream = [*frame, ("rz", theta, 1)]

    cooled = MpsStabOptimizer.from_mps(p.copy(), chi=None).apply(stream)
    plain = MpsStabOptimizer.from_mps(
        p.copy(), chi=None, exact_cooling=False
    ).apply(stream)

    assert len(cooled.exact_cooling_events) == 1
    assert cooled.state.max_bond() == 1
    assert plain.state.max_bond() == 2
    assert _fidelity(cooled.to_statevector(), plain.to_statevector()) == pytest.approx(
        1.0, abs=1e-9
    )


def test_exact_cooling_falls_back_when_no_stabilizer_pivot_exists():
    theta = 0.37
    magic_a = np.array([np.cos(0.19), -1j * np.sin(0.19)], dtype=complex)
    magic_b = np.array([np.cos(0.31), -1j * np.sin(0.31)], dtype=complex)
    p = qtn.MPS_product_state([magic_a, magic_b])
    stream = [("h", 0), ("cnot", 0, 1), ("rz", theta, 1)]

    cooled = MpsStabOptimizer.from_mps(p.copy(), chi=None).apply(stream)
    plain = MpsStabOptimizer.from_mps(
        p.copy(), chi=None, exact_cooling=False
    ).apply(stream)

    assert cooled.exact_cooling_events == []
    assert cooled.state.max_bond() == plain.state.max_bond() == 2
    assert _fidelity(cooled.to_statevector(), plain.to_statevector()) == pytest.approx(
        1.0, abs=1e-9
    )


def test_simulator_t_layer_stays_chi_one():
    n = 5
    stream = [("h", q) for q in range(n)] + [("t", q) for q in range(n)]
    sim = MpsStabOptimizer(n).apply(stream)
    assert sim.state.max_bond() == 1  # Corollary 2.1: free non-Clifford ops


def test_simulator_clifford_angle_rotations_are_free():
    # Rotations with angle a multiple of pi/2 are Clifford -> routed to the
    # tableau, so |nu> stays chi=1, and the result matches the dense reference.
    import math
    n = 3
    stream = [
        ("h", 0), ("cnot", 0, 1),
        ("rz", math.pi / 2, 1), ("rx", math.pi, 0), ("ry", -math.pi / 2, 2),
        ("rzz", math.pi, 0, 2), ("rz", math.pi / 2, 2),
    ]
    sim = MpsStabOptimizer(n).apply(stream)
    assert sim.state.max_bond() == 1  # all Clifford -> free
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    for e in stream:
        if e[0] == "h":
            psi = _apply_gate_dense(psi, _H, (e[1],), n)
        elif e[0] == "cnot":
            cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)
            psi = _apply_gate_dense(psi, cnot, (e[1], e[2]), n)
        elif e[0] in ("rz", "rx", "ry"):
            axis = {"rz": "Z", "rx": "X", "ry": "Y"}[e[0]]
            psi = _apply_gate_dense(psi, _rot(axis, e[1]), (e[2],), n)
        elif e[0] == "rzz":
            psi = _apply_gate_dense(psi, _rzz(e[1]), (e[2], e[3]), n)
    assert _fidelity(sim.to_statevector(), psi) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    ("axes", "theta"),
    [
        ("X", np.pi / 2),
        ("Y", -np.pi / 2),
        ("IZX", np.pi),
        ("XYZI", 3 * np.pi / 2),
        ("IIII", np.pi / 2),
        ("YZX", 5 * np.pi / 2),
    ],
)
def test_clifford_pauli_rotation_tableau_matches_dense(axes, theta):
    n = len(axes)
    sim = MpsStabOptimizer(n).apply(
        [("rot", theta, axes, tuple(range(n)))]
    )

    matrices = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
    pmat = matrices[axes[0]]
    for axis in axes[1:]:
        pmat = np.kron(pmat, matrices[axis])
    unitary = (
        np.cos(theta / 2) * np.eye(2**n)
        - 1j * np.sin(theta / 2) * pmat
    )
    expected = stim.Tableau.from_unitary_matrix(unitary, endian="big")
    actual = sim.state._sim.current_inverse_tableau().inverse()

    assert actual == expected
    np.testing.assert_allclose(sim.state.p_dense(), np.eye(2**n)[0])
    assert sim.state.max_bond() in (None, 1)


def test_large_clifford_pauli_rotation_avoids_dense_construction(monkeypatch):
    import pepsy.optimizers.stabilizer_tn.mps_stab_optimizer as optimizer_module

    n = 128
    sim = MpsStabOptimizer(n)

    def forbid_dense_path(*args, **kwargs):
        raise AssertionError("Clifford Pauli rotation used a dense conversion")

    monkeypatch.setattr(optimizer_module.np, "kron", forbid_dense_path)
    monkeypatch.setattr(stim.Tableau, "from_unitary_matrix", forbid_dense_path)

    entry = ("rot", np.pi / 2, "IXYZ" * (n // 4), tuple(range(n)))
    sim.apply([entry, entry])

    assert sim.state.max_bond() == 1
    assert len(sim._clifford_rot_cache) == 1


def test_simulator_matrix_entries_clifford_and_nonclifford():
    # Clifford matrix (H) -> tableau; non-Clifford matrix (T) -> |nu> ZYZ path.
    stream = [(_H, 0), (_T, 0), (_H, 1), (_T, 1)]
    sim = MpsStabOptimizer(2).apply(stream)
    ref_stream = [("h", 0), ("t", 0), ("h", 1), ("t", 1)]
    ref = _dense_reference(2, ref_stream)
    assert _fidelity(sim.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


def test_simulator_random_1q_matrix_matches_dense():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(a)
    u = q @ np.diag(np.exp(1j * np.angle(np.diag(r))))  # Haar-ish U(2)
    prep = [("h", 0), ("cnot", 0, 1)]
    sim = MpsStabOptimizer(2).apply(prep + [(u, 1)])
    psi = _dense_reference(2, prep)
    psi = _apply_gate_dense(psi, u, (1,), 2)
    assert _fidelity(sim.to_statevector(), psi) == pytest.approx(1.0, abs=1e-6)


def test_stabilizer_mps_matches_shared_stream_and_diagnostic_api():
    """The STN facade exposes the same stream/ledger surface as ordinary MPS."""
    sim = MpsStabOptimizer(1)

    assert sim.norm_diagnostics()["tracking"] is True
    assert sim.gate_stream() == ()
    assert sim.has_trajectory_events is False
    assert MpsStabOptimizer.measure_event("Z", 0) == ("measure", "Z", (0,))
    assert MpsStabOptimizer.reset_event(0) == ("reset", (0,))
    assert MpsStabOptimizer.measure_reset_event("Z", 0) == (
        "measure_reset", "Z", (0,)
    )
    assert sim.get_fit_diagnostics() is None
    assert sim.get_quality_checks() == []


def test_stabilizer_mps_run_replays_noisy_stream_through_shared_runner():
    sim = MpsStabOptimizer(1, gates=[("x_error", 0.5, 0)])

    result = sim.run(shots=8, seed=3)

    assert type(result).__name__ == "NoisyResult"
    assert result.shots == 8
    assert result.branches == 2
    assert all(
        optimizer.norm_diagnostics()["tracking"]
        for optimizer in result.optimizers
    )


def test_stabilizer_mps_transactional_run_restores_state_and_queue():
    sim = MpsStabOptimizer(1, gates=[("h", 0), ("not-a-gate", 0)])

    with pytest.raises(ValueError, match="Unknown gate"):
        sim.run(transactional=True)

    assert len(sim._queue) == 2
    np.testing.assert_allclose(sim.to_statevector(), [1.0, 0.0])
    assert sim.norm_diagnostics()["infidelity"] is None


def test_simulator_submpo_event_in_nu_frame():
    # A sub-MPO event acts directly on the coefficient MPS p; from |0...0> the
    # basis is identity so it also equals the physical operator.
    sim = MpsStabOptimizer(3, track_infidelity=True)
    mpo = pauli_rotation_mpo(0.8, ["X", "I", "Z"], sign=1.0)
    exact = mpo.apply(sim.state.p).to_dense().reshape(-1)
    sim.apply([("submpo", mpo, (0, 1, 2))])
    got = np.asarray(sim.state.p.to_dense()).reshape(-1)
    assert _fidelity(got, exact) == pytest.approx(1.0, abs=1e-9)
    assert sim.infidelities == []


def test_simulator_truncation_caps_bond_and_tracks_infidelity():
    # Build entanglement in |nu> with spread rotations, then cap chi.
    rng = np.random.default_rng(1)
    n = 6
    stream = []
    for q in range(n):
        stream.append(("h", q))
    for _ in range(20):
        a, b = rng.choice(n, size=2, replace=False)
        stream.append(("cnot", int(a), int(b)))
        stream.append(("rz", float(rng.uniform(0.2, 1.2)), int(rng.integers(n))))
    sim = MpsStabOptimizer(n, chi=4, track_infidelity=True).apply(stream)
    assert sim.state.max_bond() <= 4
    assert all(0.0 <= inf <= 1.0 for inf in sim.infidelities)
    assert np.all(np.diff(sim.infidelities) >= -1e-10)
    assert max(sim.infidelities) > 0.0  # truncation actually occurred
    assert sim.infidelities[-1] == pytest.approx(
        1.0 - sim.norm() ** 2, abs=1e-10
    )


def test_nonunitary_dense_gate_reports_gdagger_g_infidelity():
    """Non-unitary compression is normalized by the exact G-dagger-G norm."""
    gate = np.diag([1.0, 1.0, 1.0, 0.2]).astype(complex)
    sim = MpsStabOptimizer(
        2, chi=1, track_infidelity=True, exact_cooling=False
    )
    sim.apply([("h", 0), ("h", 1)])
    target_norm = np.linalg.norm(gate @ sim.to_statevector())
    assert sim._dense_gate_target_norm(gate, (0, 1)) == pytest.approx(
        target_norm, abs=1e-10
    )
    sim.apply([(gate, (0, 1))])

    expected = 1.0 - (sim.norm() / target_norm) ** 2
    assert len(sim.infidelities) == 1
    assert sim.infidelities[-1] == pytest.approx(expected, abs=1e-10)
    assert sim.norm_diagnostics()["infidelity"] == pytest.approx(expected, abs=1e-10)
    assert sim.norm_diagnostics()["current_valid"] is False

    # Measurement restores a normalized physical state without losing the
    # preceding non-unitary compression diagnostic.
    sim.measure("Z", 0, outcome=+1)
    assert sim.norm() == pytest.approx(1.0, abs=1e-10)
    assert sim.norm_diagnostics()["infidelity"] == pytest.approx(expected, abs=1e-10)


def test_norm_events_close_segment_before_measurement_normalizes_after():
    rng = np.random.default_rng(2)
    n = 6
    stream = []
    for q in range(n):
        stream.append(("h", q))
    for _ in range(20):
        a, b = rng.choice(n, size=2, replace=False)
        stream.append(("cnot", int(a), int(b)))
        stream.append(("rz", float(rng.uniform(0.2, 1.2)), int(rng.integers(n))))

    sim = MpsStabOptimizer(n, chi=4, track_infidelity=True).apply(stream)
    pre_loss = sim.infidelities[-1]
    pre_norm = sim.norm()

    sim.measure("Z", 0)

    assert sim.infidelities[-1] == pytest.approx(pre_loss)
    assert sim.norm() == pytest.approx(1.0, abs=1e-10)
    event = sim.norm_events[-1]
    assert isinstance(event, NormEventRecord)
    assert event["kind"] == "measure"
    assert event.kind == "measure"
    assert event["valid"] is True
    assert event["pre_norm"] == pytest.approx(pre_norm, abs=1e-10)
    assert event["pre_norm_sq"] == pytest.approx(pre_norm ** 2, abs=1e-10)
    assert event["segment_infidelity"] == pytest.approx(pre_loss, abs=1e-10)
    assert 0.0 <= event["branch_probability"] <= 1.0
    assert event["expected_projected_norm_sq"] == pytest.approx(
        event["pre_norm_sq"] * event["branch_probability"], abs=1e-10
    )
    assert event["projected_norm_sq"] == pytest.approx(
        event["projected_norm"] ** 2, abs=1e-10
    )
    assert event["projector_survival"] == pytest.approx(
        min(
            1.0,
            event["projected_norm_sq"] / event["expected_projected_norm_sq"],
        ),
        abs=1e-10,
    )
    assert event["post_norm"] == pytest.approx(1.0, abs=1e-10)

    diagnostics = sim.norm_diagnostics()
    assert diagnostics["completed_segments"] == 1
    assert diagnostics["segments_including_current"] == 1
    expected_total_survival = pre_norm ** 2 * event["projector_survival"]
    assert diagnostics["total_survival_proxy"] == pytest.approx(
        expected_total_survival
    )
    assert diagnostics["total_infidelity_proxy"] == pytest.approx(
        1.0 - expected_total_survival
    )
    assert diagnostics["total_norm_proxy"] == pytest.approx(
        expected_total_survival ** 0.5
    )
    assert diagnostics["norm_survival"] == pytest.approx(expected_total_survival)
    assert diagnostics["norm_infidelity"] == pytest.approx(
        1.0 - expected_total_survival
    )
    assert diagnostics["infidelity"] == pytest.approx(
        1.0 - expected_total_survival
    )
    assert diagnostics["fidelity"] == pytest.approx(expected_total_survival)
    assert sim.get_infidelities() is sim.infidelities
    assert diagnostics["norm"] == pytest.approx(expected_total_survival ** 0.5)
    assert diagnostics["geometric_mean_norm"] == pytest.approx(
        expected_total_survival ** 0.5
    )


def test_norm_events_mark_reset_boundaries():
    sim = MpsStabOptimizer(
        2, chi=1, track_infidelity=True, exact_cooling=False
    ).apply([("rxx", 0.8, 0, 1)])
    pre_loss = sim.infidelities[-1]

    sim.reset(0)

    assert sim.norm() == pytest.approx(1.0, abs=1e-10)
    event = sim.norm_events[-1]
    assert isinstance(event, NormEventRecord)
    assert event["kind"] == "reset"
    assert event["valid"] is True
    assert event["segment_infidelity"] >= pre_loss - 1e-10
    assert event["projector_survival"] is not None
    assert event["post_norm"] == pytest.approx(1.0, abs=1e-10)


def test_norm_events_track_projector_compression_loss_separately():
    sim = MpsStabOptimizer(2, chi=1, track_infidelity=True)

    sim.measure("XX", (0, 1), outcome=+1)

    event = sim.norm_events[-1]
    assert event["kind"] == "measure"
    assert event["segment_infidelity"] == pytest.approx(0.0, abs=1e-12)
    assert event["branch_probability"] == pytest.approx(0.5, abs=1e-12)
    assert event["expected_projected_norm_sq"] == pytest.approx(0.5, abs=1e-12)
    assert 0.0 < event["projected_norm_sq"] < event["expected_projected_norm_sq"]
    assert 0.0 < event["projector_survival"] < 1.0
    assert event["projector_infidelity"] == pytest.approx(
        1.0 - event["projector_survival"]
    )
    assert sim.norm() == pytest.approx(1.0, abs=1e-10)

    diagnostics = sim.norm_diagnostics()
    assert diagnostics["completed_segment_infidelities"] == pytest.approx([0.0])
    assert diagnostics["completed_projector_infidelities"] == pytest.approx(
        [event["projector_infidelity"]]
    )
    assert diagnostics["total_survival_proxy"] == pytest.approx(
        event["projector_survival"]
    )
    assert diagnostics["total_infidelity_proxy"] == pytest.approx(
        event["projector_infidelity"]
    )
    assert diagnostics["total_norm_proxy"] == pytest.approx(
        event["projector_survival"] ** 0.5
    )


def test_norm_progress_reports_entry_part_and_norm_infidelity(monkeypatch):
    progress_instances = []

    class _FakeTqdm:
        def __init__(self, **kwargs):
            self.total = kwargs["total"]
            self.n = 0
            self.postfix_calls = []
            progress_instances.append(self)

        def set_postfix(self, **kwargs):
            self.postfix_calls.append(dict(kwargs))

        def update(self, amount):
            self.n += amount

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "tqdm", types.SimpleNamespace(tqdm=_FakeTqdm))
    sim = MpsStabOptimizer(1, chi=1, track_infidelity=True, seed=7)
    sim.apply([("h", 0), ("t", 0), ("measure", "Z", 0)], progbar=True)

    progress = progress_instances[-1]
    assert progress.n == 3
    assert [call["part"] for call in progress.postfix_calls] == [
        "clifford",
        "T",
        "measurement",
    ]
    last = progress.postfix_calls[-1]
    assert sorted(last) == ["infidelity", "norm_infidelity", "part"]
    expected = sim._format_progress_infidelity(
        sim.norm_diagnostics()["infidelity"]
    )
    assert last["infidelity"] == expected
    assert last["norm_infidelity"] == expected


def test_simulator_two_qubit_nonclifford_matrix_supported():
    # A dense non-Clifford 2q matrix is now applied via Pauli decomposition.
    sim = MpsStabOptimizer(3).apply([("h", 0), ("t", 2)])
    psi = sim.to_statevector()
    u = _rzz(0.5)  # non-Clifford 2q matrix
    sim.apply([(u, (0, 1))])
    ref = _apply_gate_dense(psi, u, (0, 1), 3)
    assert _fidelity(sim.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


def test_sparse_dense_matrix_uses_submpo_instead_of_branch_sum(monkeypatch):
    import pepsy.optimizers.stabilizer_tn.mps_stab_optimizer as optimizer_module

    xx = np.kron(_X, _X)
    yy = np.kron(_Y, _Y)
    zz = np.kron(_Z, _Z)
    gate = (
        0.41 * np.eye(4, dtype=complex)
        - 0.23j * xx
        + 0.17 * yy
        + 0.11j * zz
    )
    sim = MpsStabOptimizer(4).apply([("h", 0), ("cnot", 0, 2), ("t", 3)])
    before = sim.to_statevector()
    calls = []
    original = optimizer_module.pauli_sum_submpo

    def spy_pauli_sum_submpo(branches, *args, **kwargs):
        calls.append(tuple(branches))
        return original(branches, *args, **kwargs)

    def forbid_branch_sum(self, branches, *, unitary):
        raise AssertionError("sparse Pauli sum should use the sub-MPO fast path")

    monkeypatch.setattr(optimizer_module, "pauli_sum_submpo", spy_pauli_sum_submpo)
    monkeypatch.setattr(
        optimizer_module.MpsStabOptimizer,
        "_apply_operator_sum",
        forbid_branch_sum,
    )

    sim.apply([(gate, (0, 2))])

    expected = _apply_gate_dense(before, gate, (0, 2), 4)
    actual = sim.to_statevector()
    assert calls and len(calls[0]) == 4
    assert _fidelity(actual, expected) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(actual) == pytest.approx(
        np.linalg.norm(expected), rel=1e-8
    )


def test_three_qubit_dense_matrix_explicit_opt_in_balanced_sum_matches_dense():
    matrices = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
    weighted_paulis = [
        (0.31, "III"),
        (-0.17j, "XII"),
        (0.23, "IYI"),
        (-0.11, "IIZ"),
        (0.07j, "XXI"),
        (0.19, "XIZ"),
        (-0.13j, "ZYX"),
        (0.29, "YYY"),
    ]
    gate = np.zeros((8, 8), dtype=complex)
    for weight, labels in weighted_paulis:
        pmat = matrices[labels[0]]
        for label in labels[1:]:
            pmat = np.kron(pmat, matrices[label])
        gate += weight * pmat

    sim = MpsStabOptimizer(
        3, max_pauli_decomposition_qubits=3
    ).apply(
        [("h", 0), ("cnot", 0, 1), ("t", 2)]
    )
    before = sim.to_statevector()

    sim.apply([(gate, (2, 0, 1))])
    expected = _apply_gate_dense(before, gate, (2, 0, 1), 3)
    actual = sim.to_statevector()

    assert _fidelity(actual, expected) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(actual) == pytest.approx(
        np.linalg.norm(expected), rel=1e-8
    )


def test_dense_gate_budget_rejects_before_decomposition_or_state_mutation(
    monkeypatch,
):
    import pepsy.optimizers.stabilizer_tn.mps_stab_optimizer as optimizer_module

    sim = MpsStabOptimizer(3)
    gate = np.eye(8, dtype=complex)
    gate[0, 0] = 0.5
    before = sim.to_statevector()
    before_history = (tuple(sim.infidelities), tuple(sim.bond_history))

    def forbid_decomposition(*args, **kwargs):
        raise AssertionError("Pauli decomposition ran before its budget check")

    monkeypatch.setattr(
        optimizer_module, "pauli_decomposition", forbid_decomposition
    )
    with pytest.raises(
        ValueError,
        match=(
            r"3-qubit dense gate would enumerate 64 candidate terms.*"
            r"max_pauli_decomposition_qubits=2"
        ),
    ):
        sim.apply([(gate, tuple(range(3)))])

    np.testing.assert_allclose(sim.to_statevector(), before)
    assert (tuple(sim.infidelities), tuple(sim.bond_history)) == before_history


def test_dense_gate_budget_does_not_limit_clifford_matrix_dispatch(monkeypatch):
    import pepsy.optimizers.stabilizer_tn.mps_stab_optimizer as optimizer_module

    gate = np.kron(np.kron(_H, _X), _Z)
    sim = MpsStabOptimizer(3)

    def forbid_decomposition(*args, **kwargs):
        raise AssertionError("Clifford matrix reached Pauli decomposition")

    monkeypatch.setattr(
        optimizer_module, "pauli_decomposition", forbid_decomposition
    )
    sim.apply([(gate, tuple(range(3)))])

    expected = gate @ np.eye(8)[0]
    assert _fidelity(sim.to_statevector(), expected) == pytest.approx(
        1.0, abs=1e-6
    )


def test_near_clifford_nonunitary_matrix_not_misrouted_to_tableau():
    # (1-p)I + pX is NON-unitary but close to the identity Clifford; because
    # stim.Tableau.from_unitary_matrix does not verify unitarity, it must not be
    # accepted as a tableau and applied as a no-op. (Regression: DEM "coin".)
    p = 0.1
    coin = (1 - p) * _I + p * _X
    for n, q in [(1, 0), (3, 1)]:
        sim = MpsStabOptimizer(n).apply([(coin, q)])
        psi = sim.to_statevector()
        zero = np.zeros(2 ** n, dtype=complex); zero[0] = 1.0
        ref = _apply_gate_dense(zero, coin, (q,), n)
        assert _fidelity(psi, ref) == pytest.approx(1.0, abs=1e-9)
        assert not np.allclose(psi, zero)  # state actually changed


@pytest.mark.parametrize(
    "gate_dtype, small_coeff, atol",
    [(np.complex64, 1e-5, 1e-7), (np.complex128, 1e-6, 1e-12)],
)
def test_operator_tolerance_is_independent_of_svd_cutoff(
    gate_dtype, small_coeff, atol
):
    gate = np.eye(2, dtype=gate_dtype) + small_coeff * _X.astype(gate_dtype)
    sim = MpsStabOptimizer(1, cutoff=1e-2).apply([(gate, 0)])

    np.testing.assert_allclose(
        sim.to_statevector(), np.array([1.0, small_coeff]), atol=atol
    )


def test_explicit_operator_tolerance_can_prune_small_pauli_terms():
    gate = _I + 1e-3 * _X
    sim = MpsStabOptimizer(1, cutoff=0.0, operator_tol=1e-2).apply([(gate, 0)])

    np.testing.assert_allclose(sim.to_statevector(), np.array([1.0, 0.0]))


@pytest.mark.parametrize("operator_tol", [-1.0, np.nan, np.inf])
def test_operator_tolerance_must_be_finite_and_nonnegative(operator_tol):
    with pytest.raises(ValueError, match="operator_tol"):
        MpsStabOptimizer(1, operator_tol=operator_tol)


@pytest.mark.parametrize(
    ("value", "error"),
    [(-1, ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_pauli_decomposition_budget_validation(value, error):
    with pytest.raises(error, match="max_pauli_decomposition_qubits"):
        MpsStabOptimizer(1, max_pauli_decomposition_qubits=value)


def test_copy_preserves_pauli_decomposition_budget():
    sim = MpsStabOptimizer(
        2,
        max_pauli_decomposition_qubits=None,
        max_dense_cap_qubits=None,
    )
    copied = sim.copy()

    assert copied.max_pauli_decomposition_qubits is None
    assert copied.max_dense_cap_qubits is None
    assert "max_pauli_decomposition_qubits=None" in repr(copied)
    assert "max_dense_cap_qubits=None" in repr(copied)


@pytest.mark.parametrize(
    "mode",
    ("dmrg", "dmrg1", "dmrg2", "dmrg3", "mpo", "svd", "swap", "perm", "exact"),
)
def test_coefficient_compression_modes_preserve_stn_state(mode):
    stream = [
        ("h", 0),
        ("cnot", 0, 1),
        ("t", 2),
        ("rxx", 0.37, 0, 2),
    ]
    reference = MpsStabOptimizer(
        3, mode="exact", exact_cooling=False
    ).apply(stream)
    optimizer = MpsStabOptimizer(
        3,
        chi=4,
        mode=mode,
        exact_cooling=False,
    ).apply(stream)

    assert optimizer.mode == mode
    if mode == "exact":
        assert optimizer.chi is None
    else:
        assert optimizer.state.max_bond() <= 4
    assert _fidelity(optimizer.to_statevector(), reference.to_statevector()) == pytest.approx(
        1.0, abs=1e-10
    )


def test_stn_mode_validation_and_copy_preservation():
    with pytest.raises(ValueError, match="Unknown MpsStabOptimizer mode"):
        MpsStabOptimizer(2, mode="not-a-mode")

    copied = MpsStabOptimizer(2, mode="dmrg3", chi=2).copy()
    assert copied.mode == "dmrg3"
    assert "mode='dmrg3'" in repr(copied)


@pytest.mark.parametrize(
    ("value", "error"),
    [(-1, ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_dense_cap_budget_validation(value, error):
    with pytest.raises(error, match="max_dense_cap_qubits"):
        MpsStabOptimizer(1, max_dense_cap_qubits=value)


def test_zero_operator_produces_valid_zero_mps():
    sim = MpsStabOptimizer(2, chi=1, track_infidelity=True).apply(
        [(np.zeros((2, 2), dtype=complex), 0)]
    )

    assert sim.state.p is not None
    assert sim.state.max_bond() == 1
    assert sim.norm() == pytest.approx(0.0)
    np.testing.assert_allclose(sim.to_statevector(), np.zeros(4))
    assert sim.amplitude("00") == pytest.approx(0.0)
    assert sim.probability("00") == pytest.approx(0.0)
    assert sim.infidelities == []

    # Further linear evolution keeps the valid zero state instead of crashing.
    sim.apply([("h", 1), (_I + 0.2 * _X, 0)])
    assert sim.norm() == pytest.approx(0.0)


def test_normalized_observables_reject_zero_norm_state_without_mutation():
    sim = MpsStabOptimizer(2).apply([(np.zeros((2, 2), dtype=complex), 0)])
    before = sim.to_statevector()
    before_history = (len(sim.infidelities), len(sim.bond_history), len(sim.measurements))

    calls = [
        lambda: sim.expectation("Z", 0),
        lambda: sim.expectation_pauli_sum([(1.0, "Z", 0)]),
        lambda: sim.sample("Z", 0),
        lambda: sim.probability_bits("00"),
        lambda: sim.sample_bits(1),
        lambda: sim.measure("Z", 0),
        lambda: sim.measure("Z", 0, absorb_basis=True),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="zero-norm state"):
            call()

    np.testing.assert_allclose(sim.to_statevector(), before)
    assert (len(sim.infidelities), len(sim.bond_history), len(sim.measurements)) == before_history


def test_weighted_xor_matrix_matches_dense():
    # (1-p)I + p X^{⊗k} weighted-XOR gate (the capped DEM mechanism), non-unitary
    # and near-Clifford, on non-adjacent qubits.
    p = 0.2
    xx = np.kron(_X, _X)
    wxor = (1 - p) * np.eye(4, dtype=complex) + p * xx
    prep = MpsStabOptimizer(3).apply([("h", 0), ("cnot", 0, 1)])
    psi0 = prep.to_statevector()
    prep.apply([(wxor, (0, 2))])
    ref = _apply_gate_dense(psi0, wxor, (0, 2), 3)
    assert _fidelity(prep.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Measurement (Lemma 3): expectation, forced collapse, Born sampling
# --------------------------------------------------------------------------- #
def _expectation_dense(psi, axis, q, n):
    o = _apply_gate_dense(psi.copy(), {"X": _X, "Y": _Y, "Z": _Z}[axis], (q,), n)
    return float(np.real(np.vdot(psi, o)))


def test_measure_expectation_matches_dense():
    stream = [("h", 0), ("cnot", 0, 1), ("rz", 0.7, 1), ("ry", 0.9, 2),
              ("rx", 0.5, 0)]
    n = 3
    sim = MpsStabOptimizer(n).apply(stream)
    psi = sim.to_statevector()
    for axis in ("X", "Y", "Z"):
        for q in range(n):
            assert sim.expectation(axis, q) == pytest.approx(
                _expectation_dense(psi, axis, q, n), abs=1e-6
            )


def test_measure_forced_outcome_collapses_to_dense():
    n = 3
    stream = [("h", 0), ("cnot", 0, 1), ("rz", 0.7, 1), ("ry", 0.9, 2)]
    for axis in ("X", "Y", "Z"):
        for q in range(n):
            for m in (+1, -1):
                sim = MpsStabOptimizer(n).apply(stream)
                psi = sim.to_statevector()
                # skip impossible outcomes (probability ~0)
                p = 0.5 * (1 + m * _expectation_dense(psi, axis, q, n))
                if p < 1e-6:
                    continue
                sim.measure(axis, q, outcome=m)
                o = {"X": _X, "Y": _Y, "Z": _Z}[axis]
                proj = _apply_gate_dense(psi.copy(), (np.eye(2) + m * o) / 2, (q,), n)
                proj = proj / np.linalg.norm(proj)
                assert _fidelity(sim.to_statevector(), proj) == pytest.approx(1.0, abs=1e-6)


def test_measure_is_repeatable():
    # Measuring the same observable twice returns the same outcome deterministically.
    sim = MpsStabOptimizer(3).apply([("h", 0), ("cnot", 0, 1), ("ry", 0.6, 2)])
    first = sim.measure("Z", 0)
    for _ in range(5):
        assert sim.measure("Z", 0) == first
    # post-measurement expectation is +-1
    assert abs(sim.expectation("Z", 0)) == pytest.approx(1.0, abs=1e-9)


def test_measure_born_statistics():
    # <Z> on qubit 0 of Rx(theta)|0> is cos(theta); sampled frequency should match.
    theta = 0.9
    exp_ref = np.cos(theta)
    plus = 0
    shots = 400
    for s in range(shots):
        sim = MpsStabOptimizer(1, seed=s).apply([("rx", theta, 0)])
        if sim.measure("Z", 0) == 1:
            plus += 1
    freq_exp = (2 * plus / shots) - 1  # <Z> = p+ - p-
    assert freq_exp == pytest.approx(exp_ref, abs=0.12)


def test_measure_multiqubit_pauli_and_stream_entry():
    n = 3
    stream = [("h", 0), ("cnot", 0, 1), ("rz", 0.5, 2)]
    sim = MpsStabOptimizer(n).apply(stream)
    psi = sim.to_statevector()
    zz = np.kron(_Z, _Z)
    ref = float(np.real(np.vdot(psi, _apply_gate_dense(psi.copy(), zz, (0, 1), n))))
    assert sim.expectation("ZZ", (0, 1)) == pytest.approx(ref, abs=1e-6)
    # measure via stream entry records the outcome
    sim.apply([("measure", "ZZ", (0, 1))])
    assert len(sim.measurements) == 1
    assert sim.measurements[0][2] in (+1, -1)


# --------------------------------------------------------------------------- #
# Hardening: general gate coverage, copy/inplace, incremental queue, diagnostics
# --------------------------------------------------------------------------- #
def _pauli_kron(paulis):
    m = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
    out = m[paulis[0]]
    for p in paulis[1:]:
        out = np.kron(out, m[p])
    return out


def _named_gate(entry):
    """Return ``(matrix, where)`` for a named gate-stream entry (dense reference)."""
    name = entry[0]
    clifford = {"h": _H, "x": _X, "y": _Y, "z": _Z,
                "s": np.diag([1, 1j]).astype(complex),
                "sdg": np.diag([1, -1j]).astype(complex)}
    if name in clifford:
        return clifford[name], (entry[1],)
    if name in ("cnot", "cx"):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex), (entry[1], entry[2])
    if name == "cz":
        return np.diag([1, 1, 1, -1]).astype(complex), (entry[1], entry[2])
    if name == "swap":
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], complex), (entry[1], entry[2])
    if name in ("rx", "ry", "rz"):
        return _rot({"rx": "X", "ry": "Y", "rz": "Z"}[name], entry[1]), (entry[2],)
    if name in ("rxx", "ryy", "rzz"):
        ax = {"rxx": "X", "ryy": "Y", "rzz": "Z"}[name]
        pp = _pauli_kron([ax, ax])
        u = np.cos(entry[1] / 2) * np.eye(4) - 1j * np.sin(entry[1] / 2) * pp
        return u, (entry[2], entry[3])
    if name == "t":
        return _T, (entry[1],)
    if name == "tdg":
        return _T.conj().T, (entry[1],)
    if name == "rot":
        theta, paulis, where = entry[1], entry[2], entry[3]
        where = (where,) if isinstance(where, int) else tuple(where)
        pp = _pauli_kron(list(paulis))
        u = np.cos(theta / 2) * np.eye(2 ** len(where)) - 1j * np.sin(theta / 2) * pp
        return u, where
    raise AssertionError(f"unhandled {name}")


def _dense_stream(n, stream):
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    for entry in stream:
        u, where = _named_gate(entry)
        psi = _apply_gate_dense(psi, u, where, n)
    return psi


def test_simulator_tdg_rot_rxx_ryy_match_dense():
    n = 3
    stream = [
        ("h", 0), ("cnot", 0, 1), ("tdg", 1), ("rxx", 0.8, 0, 2),
        ("ryy", -0.6, 1, 2), ("rot", 1.1, "XZ", (0, 2)), ("rot", 0.4, "Y", 1),
    ]
    sim = MpsStabOptimizer(n).apply(stream)
    ref = _dense_stream(n, stream)
    assert _fidelity(sim.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_simulator_random_exact_circuit_matches_dense(seed):
    rng = np.random.default_rng(seed)
    n = 4
    one_c = ["h", "s", "sdg", "x", "y", "z"]
    two_c = ["cnot", "cz", "swap"]
    stream = []
    for _ in range(40):
        r = rng.random()
        if r < 0.3:
            a, b = rng.choice(n, size=2, replace=False)
            stream.append((rng.choice(two_c), int(a), int(b)))
        elif r < 0.55:
            stream.append((rng.choice(one_c), int(rng.integers(n))))
        elif r < 0.8:
            axis = rng.choice(["rx", "ry", "rz"])
            stream.append((axis, float(rng.uniform(0.1, 1.3)), int(rng.integers(n))))
        else:
            a, b = rng.choice(n, size=2, replace=False)
            stream.append((rng.choice(["rxx", "rzz"]), float(rng.uniform(0.1, 1.3)), int(a), int(b)))
    sim = MpsStabOptimizer(n).apply(stream)  # exact (chi=None)
    ref = _dense_stream(n, stream)
    assert _fidelity(sim.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


def test_simulator_inplace_false_preserves_original():
    base = STNState(3)
    sim = MpsStabOptimizer(base, inplace=False).apply([("h", 0), ("rz", 0.7, 0), ("cnot", 0, 1)])
    # original untouched
    np.testing.assert_allclose(base.p_dense(), np.eye(8)[0])
    assert base.max_bond() == 1
    # the simulator evolved its own copy
    assert sim.state is not base


def test_simulator_incremental_add_gates_equivalent_to_single_run():
    stream = [("h", 0), ("cnot", 0, 1), ("rz", 0.6, 1), ("ry", -0.4, 2), ("t", 2)]
    full = MpsStabOptimizer(3).apply(stream)
    inc = MpsStabOptimizer(3)
    inc.add_gates(stream[:2]).run()
    inc.add_gates(stream[2:]).run()
    assert _fidelity(inc.to_statevector(), full.to_statevector()) == pytest.approx(1.0, abs=1e-6)


def test_run_failure_consumes_only_successful_queue_prefix():
    sim = MpsStabOptimizer(1)
    sim.add_gates([("x", 0), ("not-a-gate", 0)])

    with pytest.raises(ValueError, match="Unknown gate"):
        sim.run()

    assert sim.expectation("Z", 0) == pytest.approx(-1.0)
    assert sim._queue == [("not-a-gate", 0)]

    # Retrying the remaining failed entry must not replay the successful X.
    with pytest.raises(ValueError, match="Unknown gate"):
        sim.run()
    assert sim.expectation("Z", 0) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "entry, message",
    [
        (("rot", 0.3, "XZ", (0,)), "different lengths"),
        (("rot", 0.3, "XZ", (0, 0)), "distinct"),
        (("rot", 0.3, "XA", (0, 1)), "Invalid Pauli axis"),
        (("rot", 0.3, "XZ", (0, 2)), "out of range"),
    ],
)
def test_general_rotation_validates_pauli_support(entry, message):
    with pytest.raises(ValueError, match=message):
        MpsStabOptimizer(2).apply([entry])


def test_expectation_of_stabilizer_is_deterministic():
    # After a Clifford circuit, Z-basis stabilizers have expectation +-1.
    sim = MpsStabOptimizer(3).apply([("h", 0), ("cnot", 0, 1), ("cnot", 1, 2)])
    # Z0 Z1 and Z1 Z2 are stabilizers of the GHZ state -> expectation +1.
    assert sim.expectation("ZZ", (0, 1)) == pytest.approx(1.0, abs=1e-9)
    assert sim.expectation("ZZ", (1, 2)) == pytest.approx(1.0, abs=1e-9)


def test_pseudo_stabilizer_rank_t_state_is_maximal():
    # |T>^n = prod T prod H |0> has maximal pseudo-stabilizer rank 2^n (paper Eq. 11-12).
    for n in (2, 3):
        stream = [("h", q) for q in range(n)] + [("t", q) for q in range(n)]
        sim = MpsStabOptimizer(n).apply(stream)
        assert sim.state.max_bond() == 1
        assert sim.pseudo_stabilizer_rank() == 2 ** n


def test_submpo_truncation_caps_bond():
    # Apply a spread rotation MPO as a sub-MPO event with a chi cap.
    n = 6
    sim = MpsStabOptimizer(n, chi=4)
    sim.apply([("h", q) for q in range(n)])
    for a in range(n - 1):
        sim.apply([("cnot", a, a + 1)])
    mpo = pauli_rotation_mpo(0.7, ["X"] + ["I"] * (n - 2) + ["Z"], sign=1.0)
    sim.apply([("submpo", mpo, tuple(range(n)))])
    assert sim.state.max_bond() <= 4


# --------------------------------------------------------------------------- #
# Initial states, norm, progress bar, alias
# --------------------------------------------------------------------------- #
def test_initial_state_ghz_is_stabilizer():
    n = 4
    sim = MpsStabOptimizer.ghz(n)
    assert sim.state.max_bond() == 1  # GHZ is a stabilizer state -> chi=1
    exp = np.zeros(2 ** n, dtype=complex)
    exp[0] = exp[-1] = 1 / np.sqrt(2)
    assert _fidelity(sim.to_statevector(), exp) == pytest.approx(1.0, abs=1e-6)
    assert sim.norm() == pytest.approx(1.0, abs=1e-9)


def test_initial_state_from_bits():
    sim = MpsStabOptimizer.from_bits("1010")
    exp = np.zeros(16, dtype=complex)
    exp[int("1010", 2)] = 1.0
    assert _fidelity(sim.to_statevector(), exp) == pytest.approx(1.0, abs=1e-6)
    assert sim.state.max_bond() == 1


def test_constructor_accepts_computational_basis_mps_directly():
    rng = np.random.default_rng(12)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi = psi / np.linalg.norm(psi)
    p = qtn.MatrixProductState.from_dense(psi, dims=[2, 2, 2])

    sim = MpsStabOptimizer(p, chi=None, inplace=False)

    assert sim.state.p is not p
    assert _fidelity(sim.to_statevector(), psi) == pytest.approx(1.0, abs=1e-12)


def test_direct_mps_constructor_uses_identity_tableau_for_later_cliffords():
    rng = np.random.default_rng(13)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi = psi / np.linalg.norm(psi)
    p = qtn.MatrixProductState.from_dense(psi, dims=[2, 2])
    expected = _apply_gate_dense(psi, _H, (0,), 2)

    sim = MpsStabOptimizer.from_mps(p, chi=None).apply([("h", 0)])

    assert _fidelity(sim.to_statevector(), expected) == pytest.approx(1.0, abs=1e-6)
    assert sim.state.p is p


def test_direct_mps_constructor_rejects_non_qubit_physical_dim():
    p = qtn.MatrixProductState.from_dense(np.ones(3, dtype=complex), dims=[3])

    with pytest.raises(ValueError, match="physical dimension 2"):
        MpsStabOptimizer(p)


@pytest.mark.parametrize("bits", ["102", [0, 2], [0.0, 1.0]])
def test_bit_inputs_must_be_binary_integers(bits):
    with pytest.raises(ValueError, match="0 or 1"):
        MpsStabOptimizer.from_bits(bits)


def test_from_tableau_and_nu_roundtrip():
    src = MpsStabOptimizer(3).apply([("h", 0), ("cnot", 0, 1), ("rz", 0.6, 2), ("t", 1)])
    # Primary API is from_tableau_and_state / .p; the *_nu names remain aliases.
    rebuilt = MpsStabOptimizer.from_tableau_and_state(src.state._sim.copy(), src.state.p.copy())
    assert _fidelity(rebuilt.to_statevector(), src.to_statevector()) == pytest.approx(1.0, abs=1e-6)
    alias = MpsStabOptimizer.from_tableau_and_nu(src.state._sim.copy(), src.state.nu.copy())
    assert _fidelity(alias.to_statevector(), src.to_statevector()) == pytest.approx(1.0, abs=1e-6)


def test_from_tableau_and_state_rejects_size_mismatch_without_mutation():
    tableau_state = STNState(1)
    coefficient_state = STNState(2).p

    with pytest.raises(ValueError, match="same number of qubits"):
        STNState.from_tableau_and_state(tableau_state._sim, coefficient_state)

    assert tableau_state._sim.num_qubits == 1


def test_norm_preserved_after_circuit():
    sim = MpsStabOptimizer(3).apply(
        [("h", 0), ("cnot", 0, 1), ("rz", 0.7, 1), ("ry", 0.9, 2), ("t", 0)]
    )
    assert sim.norm() == pytest.approx(1.0, abs=1e-9)
    # and after a measurement, still normalized
    sim.measure("Z", 0)
    assert sim.norm() == pytest.approx(1.0, abs=1e-9)


def test_norm_expectation_and_measurement_respect_mps_exponent():
    sim = MpsStabOptimizer(2)
    sim.state.p.exponent = 2.0
    sim.state.info["cur_orthog"] = (0, 0)

    assert sim.norm() == pytest.approx(100.0)
    assert sim.expectation("Z", 0) == pytest.approx(1.0)

    sim.measure("Z", 0, outcome=+1)
    assert sim.norm() == pytest.approx(1.0)
    assert sim.state.p.exponent == pytest.approx(0.0)
    assert np.linalg.norm(sim.to_statevector()) == pytest.approx(1.0)


def test_run_progbar_smoke():
    pytest.importorskip("tqdm")
    sim = MpsStabOptimizer(3)
    sim.set_gates([("h", 0), ("cnot", 0, 1), ("rz", 0.5, 2), ("t", 1)]).run(progbar=True)
    assert sim.state.max_bond() >= 1


def test_stabilizermps_backward_alias():
    from pepsy.optimizers.stabilizer_tn import StabilizerMps
    assert StabilizerMps is MpsStabOptimizer
    assert StabilizerMpsSimulator is MpsStabOptimizer


def test_optimizers_namespace_exports():
    from pepsy.optimizers import (
        MpsStabOptimizer as M,
        STNState as S,
        StabilizerMpsSimulator as SMS,
    )
    assert M is MpsStabOptimizer
    assert S is STNState
    assert SMS is MpsStabOptimizer


def test_clean_class_api_runner_aliases():
    stream = [("h", 0), ("cnot", 0, 1), ("t", 0)]

    result = MpsStabOptimizer.run_stream(stream, n_qubits=2, mode="direct")
    alias = StabilizerMpsSimulator.simulate(stream, n_qubits=2, mode="direct")

    assert isinstance(result, StabilizerMpsRunResult)
    assert alias.mode == result.mode == "direct"
    assert _fidelity(
        alias.simulator.to_statevector(),
        result.simulator.to_statevector(),
    ) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Amplitude / probability / observable API
# --------------------------------------------------------------------------- #
def _pauli_op(pauli, where, n):
    mats = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
    axes = list(pauli)
    if where is None:
        where = tuple(range(n))
    elif isinstance(where, int):
        where = (where,)
    else:
        where = tuple(where)
    full = [np.eye(2, dtype=complex) for _ in range(n)]
    for ax, q in zip(axes, where):
        full[q] = mats[ax]
    out = full[0]
    for m in full[1:]:
        out = np.kron(out, m)
    return out


def test_amplitude_and_probability_match_dense():
    n = 3
    sim = MpsStabOptimizer(n).apply([("h", 0), ("cnot", 0, 1), ("rz", 0.7, 2), ("t", 1)])
    psi = sim.to_statevector()
    total = 0.0
    for k in range(2 ** n):
        bits = format(k, f"0{n}b")
        assert sim.amplitude(bits) == pytest.approx(psi[k], abs=1e-6)
        assert sim.probability(bits) == pytest.approx(abs(psi[k]) ** 2, abs=1e-6)
        total += sim.probability(bits)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_amplitude_ghz_ground_truth():
    sim = MpsStabOptimizer.ghz(3)
    assert sim.probability("000") == pytest.approx(0.5, abs=1e-6)
    assert sim.probability("111") == pytest.approx(0.5, abs=1e-6)
    assert sim.probability("010") == pytest.approx(0.0, abs=1e-9)


def test_expectation_full_register_string():
    n = 3
    sim = MpsStabOptimizer(n).apply([("h", 0), ("cnot", 0, 1), ("rz", 0.5, 2)])
    psi = sim.to_statevector()
    ref = float(np.real(np.vdot(psi, _pauli_op("ZIZ", None, n) @ psi)))
    assert sim.expectation("ZIZ") == pytest.approx(ref, abs=1e-6)
    # equivalent to the (pauli, where) form with identities dropped
    assert sim.expectation("ZIZ") == pytest.approx(sim.expectation("ZZ", (0, 2)), abs=1e-9)


def test_expectation_pauli_sum_matches_dense():
    n = 3
    sim = MpsStabOptimizer(n).apply([("h", 0), ("cnot", 0, 1), ("rz", 0.7, 1), ("ry", 0.4, 2)])
    psi = sim.to_statevector()
    terms = [(0.5, "Z", 0), (1.0, "ZZ", (0, 1)), (-0.3, "XIX")]
    ref = 0.0
    for c, p, *rest in terms:
        w = rest[0] if rest else None
        ref += c * np.real(np.vdot(psi, _pauli_op(p, w, n) @ psi))
    assert sim.expectation_pauli_sum(terms) == pytest.approx(ref, abs=1e-6)


def test_sample_statistics_no_collapse():
    theta = 0.9
    sim = MpsStabOptimizer(1, seed=0).apply([("rx", theta, 0)])
    outs = sim.sample("Z", 0, shots=2000)
    assert set(np.unique(outs)).issubset({-1, 1})
    assert outs.mean() == pytest.approx(np.cos(theta), abs=0.08)
    # sampling did not collapse the state
    assert sim.expectation("Z", 0) == pytest.approx(np.cos(theta), abs=1e-6)


# --------------------------------------------------------------------------- #
# Basis-updating (absorb) measurement + R1 magic-state injection
# --------------------------------------------------------------------------- #
_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)
_S = np.diag([1, 1j]).astype(complex)
_MAGIC = np.array([1.0, np.exp(1j * np.pi / 4)], complex) / np.sqrt(2)


def _absorb_stream(n, seed, depth=6):
    """A random Clifford + non-Clifford stream that gives a nontrivial |nu>."""
    rng = np.random.default_rng(seed)
    cliff = ["h", "s", "sdg", "x", "y", "z"]
    stream = []
    for _ in range(depth):
        for q in range(n):
            stream.append((rng.choice(cliff), q))
        for q in range(n - 1):
            if rng.random() < 0.5:
                stream.append(("cnot", q, q + 1))
        stream.append(("rz", float(rng.uniform(0.2, 1.2)), int(rng.integers(n))))
        stream.append(("rx", float(rng.uniform(0.2, 1.2)), int(rng.integers(n))))
    return stream


@pytest.mark.parametrize("seed", [0, 1, 4, 7])
@pytest.mark.parametrize("pauli,where", [("Z", 2), ("X", 1), ("Y", 0), ("ZZ", (1, 3))])
@pytest.mark.parametrize("outcome", [+1, -1])
def test_measure_absorb_matches_fixed_basis(seed, pauli, where, outcome):
    n = 4
    stream = _absorb_stream(n, seed)
    ref = MpsStabOptimizer(n).apply(stream)
    # skip (near) impossible forced outcomes
    p_plus = 0.5 * (1 + ref.expectation(pauli, where))
    if (outcome > 0 and p_plus < 1e-6) or (outcome < 0 and (1 - p_plus) < 1e-6):
        pytest.skip("outcome has ~0 probability")
    m_ref = ref.measure(pauli, where, outcome=outcome)          # fixed-basis
    a = MpsStabOptimizer(n).apply(stream)
    m_abs = a.measure(pauli, where, outcome=outcome, absorb_basis=True)  # basis-updating
    assert m_abs == m_ref
    assert _fidelity(a.to_statevector(), ref.to_statevector()) == pytest.approx(1.0, abs=1e-6)


def test_measure_absorb_forced_outcome_matches_dense_projector():
    n = 3
    stream = [("h", 0), ("cnot", 0, 1), ("rz", 0.7, 1), ("ry", 0.9, 2)]
    for axis in ("X", "Y", "Z"):
        for q in range(n):
            for m in (+1, -1):
                sim = MpsStabOptimizer(n).apply(stream)
                psi = sim.to_statevector()
                o = {"X": _X, "Y": _Y, "Z": _Z}[axis]
                p = 0.5 * (1 + m * float(np.real(np.vdot(psi, _apply_gate_dense(psi.copy(), o, (q,), n)))))
                if p < 1e-6:
                    continue
                sim.measure(axis, q, outcome=m, absorb_basis=True)
                proj = _apply_gate_dense(psi.copy(), (np.eye(2) + m * o) / 2, (q,), n)
                proj = proj / np.linalg.norm(proj)
                assert _fidelity(sim.to_statevector(), proj) == pytest.approx(1.0, abs=1e-6)


def test_measure_absorb_disentangles_product_ancilla():
    # Entangled data (GHZ chain) tensor a lone rotated ancilla; measuring the
    # ancilla out with absorb_basis must not blow up the bond.
    sim = MpsStabOptimizer(4, chi=None)
    sim.apply([("h", 0), ("cnot", 0, 1), ("cnot", 1, 2), ("rz", 0.7, 3)])
    sim.measure("Z", 3, absorb_basis=True, outcome=+1)
    assert sim.state.max_bond() == 1  # GHZ stays in the basis, ancilla removed
    assert sim.norm() == pytest.approx(1.0, abs=1e-9)


def test_prepare_magic_is_product_state():
    n = 3
    sim = MpsStabOptimizer(n)
    for q in range(n):
        sim.prepare_magic(q)
    assert sim.state.max_bond() == 1  # |A>^n is a product state
    ref = _MAGIC
    for _ in range(n - 1):
        ref = np.kron(ref, _MAGIC)
    assert _fidelity(sim.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("outcome", [+1, -1])
def test_inject_t_reproduces_t_gate(outcome):
    # data = |+> on qubit 0, magic ancilla on qubit 1; inject_t must realize T.
    sim = MpsStabOptimizer(2)
    sim.state.h(0)          # data -> |+>
    sim.prepare_magic(1)    # ancilla -> |A>
    m = sim.inject_t(0, 1, outcome=outcome)
    assert m == outcome
    # dense reference of the same gadget with the same outcome
    psi = np.kron(_H @ np.array([1, 0], complex), _MAGIC)   # |+>|A>
    psi = _apply_gate_dense(psi, _CNOT, (0, 1), 2)
    bit = 0 if m > 0 else 1
    proj = np.zeros((2, 2), complex); proj[bit, bit] = 1.0
    psi = _apply_gate_dense(psi, proj, (1,), 2)
    psi = psi / np.linalg.norm(psi)
    if m < 0:
        psi = _apply_gate_dense(psi, _S, (0,), 2)
    assert _fidelity(sim.to_statevector(), psi) == pytest.approx(1.0, abs=1e-6)
    # data qubit is T|+> regardless of branch: full state = (T|+>)_0 (x) |bit>_1
    tplus = _T @ (_H @ np.array([1, 0], complex))
    ref = np.kron(tplus, np.array([1, 0], complex) if bit == 0 else np.array([0, 1], complex))
    assert _fidelity(sim.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("outcome", [+1, -1])
def test_inject_t_on_entangled_data_matches_direct_t(outcome):
    # T injected on one qubit of a Clifford-entangled register equals a direct T,
    # and the coefficient MPS bond stays tiny (magic confined to the ancilla).
    sim = MpsStabOptimizer(4, chi=None)
    sim.apply([("h", 0), ("cnot", 0, 1), ("cnot", 1, 2)])  # GHZ on data 0,1,2
    sim.prepare_magic(3)
    m = sim.inject_t(0, 3, outcome=outcome)
    # dense reference: same gadget on |GHZ>_012 (x) |A>_3
    base = _apply_gate_dense(_apply_gate_dense(_apply_gate_dense(np.eye(8)[0], _H, (0,), 3), _CNOT, (0, 1), 3), _CNOT, (1, 2), 3)
    psi = np.kron(base, _MAGIC)
    psi = _apply_gate_dense(psi, _CNOT, (0, 3), 4)
    bit = 0 if m > 0 else 1
    proj = np.zeros((2, 2), complex); proj[bit, bit] = 1.0
    psi = _apply_gate_dense(psi, proj, (3,), 4)
    psi = psi / np.linalg.norm(psi)
    if m < 0:
        psi = _apply_gate_dense(psi, _S, (0,), 4)
    assert _fidelity(sim.to_statevector(), psi) == pytest.approx(1.0, abs=1e-6)
    assert sim.state.max_bond() <= 2  # magic stays confined; |nu> bond bounded


def test_reset_stream_entry_resets_to_zero():
    n = 3
    # entangle + rotate, then reset qubit 2 to |0> via the stream entry.
    sim = MpsStabOptimizer(n).apply([("h", 0), ("cnot", 0, 1), ("ry", 0.8, 2)])
    sim.apply([("reset", 2)])
    assert sim.expectation("Z", 2) == pytest.approx(1.0, abs=1e-9)  # |0> -> <Z> = +1
    assert sim.probability("000") + sim.probability("110") == pytest.approx(1.0, abs=1e-6)
    # reset is not recorded as a measurement
    assert sim.measurements == []


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_reset_stream_entry_supports_pauli_bases(axis):
    sim = MpsStabOptimizer(1).apply([("h", 0), ("reset", 0, axis)])

    assert sim.expectation(axis, 0) == pytest.approx(1.0, abs=1e-9)
    assert sim.measurements == []


@pytest.mark.parametrize(
    ("axis", "bits", "outcome"),
    [("Z", "1", -1), ("X", "0", -1), ("Y", "0", -1)],
)
def test_measure_reset_stream_entry_records_then_resets(axis, bits, outcome):
    sim = MpsStabOptimizer.from_bits(bits).apply(
        [("measure_reset", axis, 0, outcome)]
    )

    assert sim.measurements == [(axis, 0, outcome)]
    assert isinstance(sim.measurements[0], MeasurementRecord)
    assert sim.measurements[0].pauli == axis
    assert sim.expectation(axis, 0) == pytest.approx(1.0, abs=1e-9)


def test_cap_stream_entry_contracts_physical_qubit_and_shortens():
    n = 3
    sim = MpsStabOptimizer(n, chi=None).apply(
        [("h", 0), ("cnot", 0, 1), ("ry", 0.4, 2)]
    )
    before = sim.to_statevector()
    vec = np.array([1.0, 1.0], dtype=complex)
    expected = _cap_dense(before, vec, 1, n)

    sim.apply([("cap", 1, vec)])

    assert sim.n == 2
    np.testing.assert_allclose(sim.to_statevector(), expected, atol=1e-10)
    assert sim.bond_history[-1] == sim.state.max_bond()


def test_cap_stream_entry_obeys_dense_qubit_guard():
    sim = MpsStabOptimizer(3, max_dense_cap_qubits=2)

    with pytest.raises(ValueError, match="max_dense_cap_qubits"):
        sim.apply([("cap", 0, [1.0, 1.0])])


def test_reset_multiqubit_and_disentangles():
    # GHZ chain; reset the whole register back to |000..0>.
    n = 4
    sim = MpsStabOptimizer(n, chi=None).apply(
        [("h", 0), ("cnot", 0, 1), ("cnot", 1, 2), ("cnot", 2, 3), ("rz", 0.5, 3)]
    )
    sim.reset(range(n))
    ref = np.zeros(2 ** n, dtype=complex); ref[0] = 1.0
    assert _fidelity(sim.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)
    assert sim.state.max_bond() == 1


def test_reset_then_reuse_ancilla_for_injection():
    # Reset a used qubit, then re-prepare magic and inject again (ancilla reuse).
    sim = MpsStabOptimizer(2)
    sim.state.h(0)
    sim.prepare_magic(1)
    sim.inject_t(0, 1, outcome=+1)   # data qubit -> T|+>, ancilla consumed
    sim.reset(1)                      # free the ancilla back to |0>
    sim.prepare_magic(1)              # reload magic on the same site
    m = sim.inject_t(0, 1, outcome=+1)  # apply another T -> T^2|+>
    assert m == +1
    tt_plus = _T @ (_T @ (_H @ np.array([1, 0], complex)))
    full = sim.to_statevector().reshape(2, 2)
    data_vec = full[:, 0] if np.linalg.norm(full[:, 0]) > np.linalg.norm(full[:, 1]) else full[:, 1]
    assert _fidelity(data_vec, tt_plus) == pytest.approx(1.0, abs=1e-6)


def test_measure_absorb_via_stream_entry():
    n = 3
    stream = [("h", 0), ("cnot", 0, 1), ("rz", 0.7, 2)]
    sim = MpsStabOptimizer(n).apply(stream)
    # ("measure", pauli, where, outcome, absorb_basis)
    sim.apply([("measure", "Z", 0, +1, True)])
    assert len(sim.measurements) == 1 and sim.measurements[0][2] == +1
    assert sim.expectation("Z", 0) == pytest.approx(1.0, abs=1e-9)


def test_measurement_localizer_cache_keys_on_layout_and_terms():
    sim = MpsStabOptimizer(3)
    terms = {0: "X", 2: "Z"}

    first = sim._localizing_clifford_cached(terms)
    second = sim._localizing_clifford_cached(dict(reversed(tuple(terms.items()))))

    assert first is second
    assert len(sim._localizer_cache) == 1
    sim.apply_layout([2, 1, 0], layout_report=False)
    assert sim._localizer_cache == {}


# --------------------------------------------------------------------------- #
# R4: scalable computational-basis sampling / probabilities (no 2**n)
# --------------------------------------------------------------------------- #
def test_probability_bits_matches_dense():
    n = 4
    sim = MpsStabOptimizer(n, seed=0).apply(
        [("h", 0), ("cnot", 0, 1), ("rz", 0.7, 2), ("t", 1), ("ry", 0.5, 3)]
    )
    psi = sim.to_statevector()
    total = 0.0
    for k in range(2 ** n):
        b = format(k, f"0{n}b")
        assert sim.probability_bits(b) == pytest.approx(abs(psi[k]) ** 2, abs=1e-6)
        total += sim.probability_bits(b)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_probability_bits_accepts_mps_order_with_layout():
    n = 4
    sim = MpsStabOptimizer(n, seed=0, layout=[2, 0, 3, 1], layout_report=False).apply(
        [("h", 0), ("cnot", 0, 1), ("rz", 0.7, 2), ("t", 1), ("ry", 0.5, 3)]
    )
    psi = sim.to_statevector()
    for k in (0, 3, 9, 15):
        b = format(k, f"0{n}b")
        ref = abs(psi[k]) ** 2
        assert sim.probability_bits(b, order="physical") == pytest.approx(ref, abs=1e-6)
        assert sim.probability_bits(b, order="mps") == pytest.approx(ref, abs=1e-6)
        assert sim.probability_bits(b, order="auto") == pytest.approx(ref, abs=1e-6)


def test_probability_bits_many_matches_scalar_and_dense():
    n = 4
    sim = MpsStabOptimizer(n, seed=1).apply(
        [("h", 0), ("cnot", 0, 1), ("rz", 0.4, 2), ("t", 3), ("ry", 0.6, 1)]
    )
    bitstrings = ["0000", "0011", "0011", "1010", "1111"]
    got = sim.probability_bits_many(bitstrings)
    scalar = np.array([sim.probability_bits(bits) for bits in bitstrings])
    psi = sim.to_statevector()
    dense = np.array([abs(psi[int(bits, 2)]) ** 2 for bits in bitstrings])
    np.testing.assert_allclose(got, scalar, atol=1e-9)
    np.testing.assert_allclose(got, dense, atol=1e-6)


def test_probability_bits_many_empty_and_single_bitstring():
    sim = MpsStabOptimizer(2).apply([("h", 0)])
    assert sim.probability_bits_many([]).shape == (0,)
    got = sim.probability_bits_many("00")
    assert got.shape == (1,)
    assert got[0] == pytest.approx(sim.probability_bits("00"), abs=1e-12)


def test_probability_bits_does_not_mutate_state():
    sim = MpsStabOptimizer(3).apply([("h", 0), ("cnot", 0, 1), ("t", 2)])
    before = sim.to_statevector()
    sim.probability_bits("010")
    sim.probability_bits_many(["010", "111"])
    assert _fidelity(sim.to_statevector(), before) == pytest.approx(1.0, abs=1e-9)


def test_probability_bits_rejects_non_binary_values():
    with pytest.raises(ValueError, match="0 or 1"):
        MpsStabOptimizer(2).probability_bits([0, 2])
    with pytest.raises(ValueError, match="0 or 1"):
        MpsStabOptimizer(2).probability_bits_many([[0, 1], [0, 2]])


def test_sample_bits_frequencies_match_dense():
    n = 3
    sim = MpsStabOptimizer(n, seed=3).apply([("h", 0), ("cnot", 0, 1), ("ry", 0.9, 2)])
    probs = np.abs(sim.to_statevector()) ** 2
    shots = 4000
    s = sim.sample_bits(shots, seed=7)
    assert s.shape == (shots, n) and set(np.unique(s)).issubset({0, 1})
    idx = (s.astype(int) * (1 << np.arange(n - 1, -1, -1))).sum(1)
    freq = np.bincount(idx, minlength=2 ** n) / shots
    assert 0.5 * np.abs(freq - probs).sum() < 0.04  # total-variation distance


def test_sample_bits_accepts_mps_order_with_layout():
    n = 3
    sim = MpsStabOptimizer(n, seed=3, layout=[2, 0, 1], layout_report=False).apply(
        [("h", 0), ("cnot", 0, 1), ("ry", 0.9, 2)]
    )
    probs = np.abs(sim.to_statevector()) ** 2
    shots = 4000
    s = sim.sample_bits(shots, seed=7, order="mps")
    assert s.shape == (shots, n) and set(np.unique(s)).issubset({0, 1})
    idx = (s.astype(int) * (1 << np.arange(n - 1, -1, -1))).sum(1)
    freq = np.bincount(idx, minlength=2 ** n) / shots
    assert 0.5 * np.abs(freq - probs).sum() < 0.04


def test_sample_bits_packed_matches_unpacked_samples():
    n = 5
    sim = MpsStabOptimizer(n, seed=3).apply(
        [("h", 0), ("cnot", 0, 1), ("ry", 0.9, 2), ("t", 4)]
    )
    raw = sim.sample_bits(64, seed=7, shuffle=False)
    packed = sim.sample_bits(64, seed=7, shuffle=False, packed=True)
    assert packed.shape == (64, 1)
    assert packed.dtype == np.uint8
    unpacked = np.unpackbits(packed, axis=1, bitorder="big")[:, :n]
    np.testing.assert_array_equal(unpacked, raw)


def test_iter_sample_bits_chunks_and_packed_output():
    n = 3
    sim = MpsStabOptimizer(n).apply([("h", 0), ("cnot", 0, 1), ("ry", 0.4, 2)])
    chunks = list(sim.iter_sample_bits(17, chunk_size=6, seed=11, packed=True))
    assert [chunk.shape for chunk in chunks] == [(6, 1), (6, 1), (5, 1)]
    unpacked = np.vstack([
        np.unpackbits(chunk, axis=1, bitorder="big")[:, :n]
        for chunk in chunks
    ])
    assert unpacked.shape == (17, n)
    assert set(np.unique(unpacked)).issubset({0, 1})


def test_bitstring_api_aliases_match_existing_methods():
    sim = MpsStabOptimizer(3).apply([("h", 0), ("cnot", 0, 1), ("t", 2)])

    np.testing.assert_array_equal(
        sim.sample_bitstrings(16, seed=5, shuffle=False),
        sim.sample_bits(16, seed=5, shuffle=False),
    )
    assert sim.bitstring_probability("010") == pytest.approx(
        sim.probability_bits("010"),
        abs=1e-12,
    )
    np.testing.assert_allclose(
        sim.bitstring_probabilities(["000", "010", "111"]),
        sim.probability_bits_many(["000", "010", "111"]),
        atol=1e-12,
    )
    alias_chunks = list(sim.iter_sample_bitstrings(7, chunk_size=3, seed=9))
    original_chunks = list(sim.iter_sample_bits(7, chunk_size=3, seed=9))
    assert [chunk.shape for chunk in alias_chunks] == [(3, 3), (3, 3), (1, 3)]
    for alias_chunk, original_chunk in zip(alias_chunks, original_chunks):
        np.testing.assert_array_equal(alias_chunk, original_chunk)


def test_sample_bits_shuffle_false_keeps_prefix_grouping():
    sim = MpsStabOptimizer(1).apply([("h", 0)])
    s = sim.sample_bits(40, seed=5, shuffle=False)[:, 0]
    assert np.array_equal(s, np.sort(s))


def test_sample_bits_rows_are_exchangeable():
    sim = MpsStabOptimizer(1).apply([("h", 0)])
    draws = np.array([
        sim.sample_bits(2, seed=seed)[:, 0]
        for seed in range(300)
    ])

    assert draws[:, 0].mean() == pytest.approx(0.5, abs=0.1)
    assert draws[:, 1].mean() == pytest.approx(0.5, abs=0.1)


def test_sample_bits_stabilizer_ghz_support():
    # GHZ only has support on 000 and 111.
    sim = MpsStabOptimizer.ghz(3)
    s = sim.sample_bits(500, seed=1)
    rows = {tuple(r) for r in s.tolist()}
    assert rows.issubset({(0, 0, 0), (1, 1, 1)})


def test_sample_bits_deterministic_product_state():
    # A computational-basis product state samples that bitstring with certainty.
    sim = MpsStabOptimizer.from_bits("1011")
    s = sim.sample_bits(64, seed=0)
    assert np.all(s == np.array([1, 0, 1, 1], dtype=np.int8))


def test_copy_is_independent():
    sim = MpsStabOptimizer(3).apply([("h", 0), ("cnot", 0, 1), ("t", 2)])
    clone = sim.copy()
    clone.apply([("x", 0), ("rz", 0.5, 2)])
    # mutating the clone leaves the original untouched
    assert _fidelity(sim.to_statevector(),
                     MpsStabOptimizer(3).apply([("h", 0), ("cnot", 0, 1), ("t", 2)]).to_statevector()) == pytest.approx(1.0, abs=1e-6)
    assert clone.state is not sim.state


# --------------------------------------------------------------------------- #
# Robust: general Rz(phi) injection (pi/4 multiples) + edge guards
# --------------------------------------------------------------------------- #
def _rz(theta):
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]).astype(complex)


@pytest.mark.parametrize("phi", [np.pi / 4, -np.pi / 4, np.pi / 2, 3 * np.pi / 4])
@pytest.mark.parametrize("outcome", [+1, -1])
def test_inject_rz_matches_dense(phi, outcome):
    sim = MpsStabOptimizer(2)
    sim.state.h(0)                       # data -> |+>
    sim.prepare_magic(1, angle=phi)      # ancilla -> Rz(phi)|+>
    m = sim.inject_rz(0, 1, phi, outcome=outcome)
    assert m == outcome
    # dense reference of the same gadget with the same outcome
    plus = _H @ np.array([1, 0], complex)
    psi = np.kron(plus, _rz(phi) @ plus)
    psi = _apply_gate_dense(psi, _CNOT, (0, 1), 2)
    bit = 0 if m > 0 else 1
    proj = np.zeros((2, 2), complex); proj[bit, bit] = 1.0
    psi = _apply_gate_dense(psi, proj, (1,), 2)
    psi = psi / np.linalg.norm(psi)
    if m < 0:
        psi = _apply_gate_dense(psi, _rz(2 * phi), (0,), 2)
    assert _fidelity(sim.to_statevector(), psi) == pytest.approx(1.0, abs=1e-6)
    # data qubit is Rz(phi)|+> regardless of branch
    ref = np.kron(_rz(phi) @ plus, np.array([1, 0], complex) if bit == 0 else np.array([0, 1], complex))
    assert _fidelity(sim.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("outcome", [+1, -1])
def test_inject_tdg_matches_tdg(outcome):
    sim = MpsStabOptimizer(2)
    sim.state.h(0)
    sim.prepare_magic(1, angle=-np.pi / 4)
    sim.inject_tdg(0, 1, outcome=outcome)
    plus = _H @ np.array([1, 0], complex)
    tdg_plus = _T.conj().T @ plus
    full = sim.to_statevector().reshape(2, 2)
    data_vec = full[:, 0] if np.linalg.norm(full[:, 0]) > np.linalg.norm(full[:, 1]) else full[:, 1]
    assert _fidelity(data_vec, tdg_plus) == pytest.approx(1.0, abs=1e-6)


def test_inject_rz_rejects_non_pi4_angle():
    sim = MpsStabOptimizer(2)
    sim.state.h(0)
    sim.prepare_magic(1, angle=0.3)
    with pytest.raises(ValueError, match="multiple of pi/4"):
        sim.inject_rz(0, 1, 0.3)


def test_absorb_measure_forced_impossible_raises():
    # GHZ: forcing Z0 = -1 while Z1 = +1 is impossible (they are perfectly correlated).
    sim = MpsStabOptimizer(2).apply([("h", 0), ("cnot", 0, 1)])
    sim.measure("Z", 0, outcome=+1, absorb_basis=True)   # collapse to |00>
    before_p = sim.state.p_dense()
    before_tableau = sim.state._sim.current_inverse_tableau()
    before_history = (len(sim.infidelities), len(sim.bond_history), len(sim.measurements))
    with pytest.raises(ValueError, match="0 probability"):
        sim.measure("Z", 1, outcome=-1, absorb_basis=True)
    np.testing.assert_allclose(sim.state.p_dense(), before_p)
    assert sim.state._sim.current_inverse_tableau() == before_tableau
    assert (len(sim.infidelities), len(sim.bond_history), len(sim.measurements)) == before_history


def test_fixed_basis_forced_impossible_raises():
    # Same impossible post-selection via the default fixed-basis path: it must
    # raise on the ~0-norm collapse rather than silently keep a garbage state.
    sim = MpsStabOptimizer(2).apply([("h", 0), ("cnot", 0, 1)])
    sim.measure("Z", 0, outcome=+1)   # collapse to |00>
    before = sim.to_statevector()
    before_history = (len(sim.infidelities), len(sim.bond_history), len(sim.measurements))
    with pytest.raises(ValueError, match="0 probability"):
        sim.measure("Z", 1, outcome=-1)
    assert _fidelity(sim.to_statevector(), before) == pytest.approx(1.0, abs=1e-9)
    assert (len(sim.infidelities), len(sim.bond_history), len(sim.measurements)) == before_history


@pytest.mark.parametrize("outcome", [0, 2, -2, 0.5])
def test_measure_rejects_invalid_forced_outcome_without_mutation(outcome):
    sim = MpsStabOptimizer(1).apply([("h", 0)])
    before = sim.to_statevector()
    with pytest.raises(ValueError, match=r"exactly \+1 or -1"):
        sim.measure("Z", 0, outcome=outcome)
    assert _fidelity(sim.to_statevector(), before) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Mature: circuit-rewrite front end (auto-inject Z-rotations)
# --------------------------------------------------------------------------- #
def _data_marginal_ref(direct_sim, n_ancilla):
    """Reference full statevector for injection: (direct result) (x) |0>^n_ancilla."""
    e0 = np.zeros(2 ** n_ancilla, dtype=complex); e0[0] = 1.0
    return np.kron(direct_sim.to_statevector(), e0)


@pytest.mark.parametrize("n_ancilla", [1, 2])
def test_run_with_injection_matches_direct(n_ancilla):
    nd = 3
    stream = [("h", 0), ("cnot", 0, 1), ("t", 2), ("tdg", 0),
              ("rz", np.pi / 4, 1), ("cnot", 1, 2), ("t", 1), ("rz", np.pi / 2, 0)]
    direct = MpsStabOptimizer(nd).apply(stream)
    inj = MpsStabOptimizer.with_injection(nd, stream, n_ancilla=n_ancilla)
    assert inj.n == nd + n_ancilla
    ref = _data_marginal_ref(direct, n_ancilla)
    assert _fidelity(inj.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


def test_with_injection_auto_layout_matches_direct():
    nd = 3
    stream = [("h", 0), ("cnot", 0, 1), ("t", 2), ("tdg", 0),
              ("rz", np.pi / 4, 1), ("cnot", 1, 2), ("t", 1)]
    direct = MpsStabOptimizer(nd).apply(stream)
    inj = MpsStabOptimizer.with_injection(
        nd,
        stream,
        n_ancilla=1,
        layout="auto",
        layout_report=False,
    )

    assert inj.layout_plan is not None
    assert inj.layout_plan["source"] == "queued_frame_supports"
    assert _fidelity(inj.to_statevector(), _data_marginal_ref(direct, 1)) == pytest.approx(
        1.0, abs=1e-6
    )


def test_with_injection_pool_one_recycles_many_t():
    n = 5
    stream = ([("h", i) for i in range(n)]
              + [("cnot", i, i + 1) for i in range(n - 1)]
              + [("t", i) for i in range(n)] * 2)  # 10 T-gates
    direct = MpsStabOptimizer(n).apply(stream)
    inj = MpsStabOptimizer.with_injection(n, stream, n_ancilla=1)  # single recycled ancilla
    ref = _data_marginal_ref(direct, 1)
    assert _fidelity(inj.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


def test_run_with_injection_non_pi4_rz_applied_normally():
    # rz(0.3) is not a pi/4 multiple -> must fall through to the normal path (still correct).
    nd = 2
    stream = [("h", 0), ("cnot", 0, 1), ("rz", 0.3, 1), ("t", 0)]
    direct = MpsStabOptimizer(nd).apply(stream)
    inj = MpsStabOptimizer.with_injection(nd, stream, n_ancilla=1)
    assert _fidelity(inj.to_statevector(), _data_marginal_ref(direct, 1)) == pytest.approx(1.0, abs=1e-6)


def test_run_with_injection_rejects_target_in_pool():
    sim = MpsStabOptimizer(3)
    with pytest.raises(ValueError, match="ancilla pool"):
        sim.run_with_injection([("t", 2)], ancillas=[2])


@pytest.mark.parametrize("ancillas", [[2, 2], [3], [-1]])
def test_run_with_injection_rejects_invalid_ancilla_pool(ancillas):
    sim = MpsStabOptimizer(3)
    with pytest.raises(ValueError, match="ancilla"):
        sim.run_with_injection([("t", 0)], ancillas=ancillas)


def test_run_with_injection_rejects_dirty_ancilla_before_mutation():
    sim = MpsStabOptimizer(2).apply([("x", 1)])
    before = sim.to_statevector()

    with pytest.raises(ValueError, match="must start clean"):
        sim.run_with_injection([("t", 0)], ancillas=[1])

    assert _fidelity(sim.to_statevector(), before) == pytest.approx(1.0, abs=1e-9)


def test_run_with_injection_rejects_ordinary_entry_touching_pool_before_mutation():
    sim = MpsStabOptimizer(2)
    before = sim.to_statevector()

    with pytest.raises(ValueError, match="ordinary stream entry"):
        sim.run_with_injection([("h", 1), ("t", 0)], ancillas=[1])

    assert _fidelity(sim.to_statevector(), before) == pytest.approx(1.0, abs=1e-9)


def test_run_with_injection_no_recycle_exhausts():
    # two T-gates, single ancilla, recycle disabled -> exhaustion error.
    sim = MpsStabOptimizer(3)
    with pytest.raises(RuntimeError, match="exhausted"):
        sim.run_with_injection([("t", 0), ("t", 1)], ancillas=[2], recycle=False)


def test_run_with_injection_spread_pool_matches_direct():
    # Ancillas interspersed among data qubits (nearest-pick locality path).
    # Injecting the T gates must equal applying them directly on the same register.
    n = 5
    pool = [1, 3]
    stream = [("h", 0), ("cnot", 0, 2), ("t", 4), ("rz", np.pi / 4, 0),
              ("cnot", 2, 4), ("t", 2), ("tdg", 0)]
    direct = MpsStabOptimizer(n).apply(stream)   # T applied directly; qubits 1,3 stay |0>
    inj = MpsStabOptimizer(n)
    inj.run_with_injection(stream, ancillas=pool)  # teleport via spread ancillas, reset at end
    assert _fidelity(inj.to_statevector(), direct.to_statevector()) == pytest.approx(1.0, abs=1e-6)
    for a in pool:  # ancillas returned to |0>
        assert inj.expectation("Z", a) == pytest.approx(1.0, abs=1e-9)


def test_run_with_injection_reset_ancillas_leaves_zero():
    sim = MpsStabOptimizer(2)
    sim.state.h(0)
    sim.run_with_injection([("t", 0)], ancillas=[1], reset_ancillas=True)
    # ancilla qubit 1 is back to |0> -> <Z_1> = +1
    assert sim.expectation("Z", 1) == pytest.approx(1.0, abs=1e-9)


def test_magic_strategy_recommends_explicit_clifford_t_execution_modes():
    stream = [
        ("h", 0), ("cnot", 0, 1), ("t", 0),
        ("tdg", 1), ("rz", np.pi / 4, 0),
    ]

    default = MpsStabOptimizer.recommend_magic_strategy(stream)
    assert default["recommended_mode"] == "immediate"
    assert default["is_clifford_t_like"]
    assert default["injectable_entries"] == 3
    assert default["deferred_ancillas_required"] == 3
    assert default["deferred_feasible"] is None
    assert "with_injection" in default["message"]

    deferred = MpsStabOptimizer.recommend_magic_strategy(
        stream, ancilla_budget=3, prioritize_peak_bond=True
    )
    assert deferred["recommended_mode"] == "deferred"
    assert deferred["deferred_feasible"]
    assert "with_deferred_injection" in deferred["message"]

    constrained = MpsStabOptimizer.recommend_magic_strategy(
        stream, ancilla_budget=1, prioritize_peak_bond=True
    )
    assert constrained["recommended_mode"] == "immediate"
    assert not constrained["deferred_feasible"]

    queued = MpsStabOptimizer(2, gates=stream)
    assert queued.queued_magic_strategy()["message"] == default["message"]
    assert len(queued._queue) == len(stream)


def test_magic_strategy_identifies_direct_and_mixed_streams():
    direct = MpsStabOptimizer.recommend_magic_strategy(
        [("h", 0), ("rxx", 0.31, 0, 1)]
    )
    assert direct["recommended_mode"] == "direct"
    assert direct["other_nonclifford_entries"] == 1
    assert "exact_cooling=True" in direct["message"]

    mixed = MpsStabOptimizer.recommend_magic_strategy(
        [("t", 0), ("rx", 0.31, 1)]
    )
    assert mixed["recommended_mode"] == "immediate"
    assert mixed["injectable_entries"] == 1
    assert mixed["other_nonclifford_entries"] == 1
    assert not mixed["is_clifford_t_like"]


def test_magic_strategy_recognizes_stim_style_clifford_matrices():
    stim_h = np.asarray(_H, dtype=np.complex64)
    report = MpsStabOptimizer.recommend_magic_strategy([(stim_h, 0), ("t", 0)])

    assert report["recommended_mode"] == "immediate"
    assert report["clifford_entries"] == 1
    assert report["injectable_entries"] == 1
    assert report["is_clifford_t_like"]


def test_stream_analysis_summarizes_pepsy_native_design():
    stream = [
        ("h", 0),
        ("cnot", 0, 1),
        ("t", 0),
        ("tdg", 1),
        ("rz", np.pi / 4, 0),
        ("rx", 0.31, 2),
        ("measure", "Z", 0),
        ("reset", 1),
        ("measure_reset", "X", 2),
    ]

    analysis = MpsStabOptimizer.analyze_stream(stream, n_qubits=3)

    assert isinstance(analysis, StreamAnalysisRecord)
    assert analysis.total_entries == 9
    assert analysis["injectable_entries"] == 3
    assert analysis.other_nonclifford_entries == 1
    assert analysis.measurement_entries == 1
    assert analysis.reset_entries == 1
    assert analysis.measure_reset_entries == 1
    assert analysis.touched_qubits == (0, 1, 2)
    assert analysis.estimated_qubits == 3
    assert analysis.is_clifford_t_like is False


def test_stream_analysis_identifies_dense_matrices_as_cost_drivers():
    nonunitary = np.array([[1.0, 0.0], [0.25, 0.0]], dtype=complex)
    stream = [(np.asarray(_H, dtype=np.complex64), 0), (nonunitary, 1)]

    analysis = MpsStabOptimizer.analyze_stream(stream, n_qubits=2)

    assert analysis.clifford_entries == 1
    assert analysis.dense_matrix_entries == 2
    assert analysis.unitary_matrix_entries == 1
    assert analysis.nonunitary_matrix_entries == 1
    assert analysis.opaque_entries == 1
    assert any("Non-unitary" in warning for warning in analysis.warnings)


def test_recommend_settings_wraps_magic_strategy_and_settings():
    stream = [("h", 0), ("cnot", 0, 1), ("t", 0), ("tdg", 1)]

    advice = MpsStabOptimizer.recommend_settings(
        stream,
        n_qubits=2,
        ancilla_budget=2,
        prioritize_peak_bond=True,
        goal="benchmark",
    )

    assert isinstance(advice, StabilizerMpsSettingsAdvice)
    assert advice.recommended_mode == "deferred"
    assert advice.execution_method == "with_deferred_injection"
    assert advice.settings["chi"] == 64
    assert advice.settings["layout"] == "auto"
    assert advice.settings["layout_report"] is False
    assert advice["settings"]["track_infidelity"] is True
    assert advice.deferred_ancillas_required == 2
    assert advice.analysis.injectable_entries == 2
    assert advice.magic_strategy["recommended_mode"] == "deferred"
    assert "Benchmark direct" in " ".join(advice.warnings)


def test_recommend_settings_validate_goal_prefers_exact_reference():
    advice = MpsStabOptimizer.recommend_settings(
        [("t", 0), ("rx", 0.31, 1)],
        n_qubits=2,
        goal="validate",
    )

    assert advice.settings["chi"] is None
    assert advice.settings["track_infidelity"] is False
    assert advice.recommended_mode == "immediate"
    assert advice.execution_method == "with_injection"


def test_queued_recommend_settings_does_not_consume_queue():
    stream = [("h", 0), ("t", 0), ("rx", 0.31, 1)]
    sim = MpsStabOptimizer(2, gates=stream)

    analysis = sim.queued_stream_analysis()
    advice = sim.queued_recommend_settings(ancilla_budget=1)

    assert analysis.estimated_qubits == 2
    assert advice.analysis.total_entries == len(stream)
    assert advice.recommended_mode == "immediate"
    assert len(sim._queue) == len(stream)


def test_run_stabilizer_mps_stream_direct_matches_stim_clifford():
    stream = [("h", 0), ("cnot", 0, 1), ("s", 1), ("cz", 0, 1)]

    result = run_stabilizer_mps_stream(stream, n_qubits=2)

    assert isinstance(result, StabilizerMpsRunResult)
    assert result.mode == "direct"
    assert result.execution_method == "apply"
    assert result.remaining_queue == 0
    assert result.final_bond == 1
    assert result.peak_bond == 1
    assert result["settings"]["n_qubits"] == 2
    assert _fidelity(
        result.simulator.to_statevector(),
        _stim_reference_state(2, stream),
    ) == pytest.approx(1.0, abs=1e-6)


def test_run_stabilizer_mps_stream_immediate_matches_direct_magic():
    stream = [("h", 0), ("cnot", 0, 1), ("t", 0), ("tdg", 1)]

    direct = run_stabilizer_mps_stream(stream, n_qubits=2, mode="direct")
    injected = run_stabilizer_mps_stream(
        stream,
        n_qubits=2,
        mode="immediate",
        n_ancilla=1,
        seed=4,
    )

    assert injected.mode == "immediate"
    assert injected.execution_method == "with_injection"
    assert injected.injection_report.n_injections == 2
    assert injected.projection_elapsed_s >= 0.0
    assert len(injected.immediate_projection_events) == 2
    assert _fidelity(
        injected.simulator.to_statevector(),
        _data_marginal_ref(direct.simulator, 1),
    ) == pytest.approx(1.0, abs=1e-6)


def test_run_stabilizer_mps_stream_deferred_matches_direct_magic():
    stream = [("h", 0), ("cnot", 0, 1), ("t", 0), ("tdg", 1)]

    direct = run_stabilizer_mps_stream(stream, n_qubits=2, mode="direct")
    deferred = run_stabilizer_mps_stream(
        stream,
        n_qubits=2,
        mode="deferred",
        n_ancilla=2,
        run_options={"outcomes": (1, -1), "projection_order": "input"},
    )

    assert deferred.mode == "deferred"
    assert deferred.execution_method == "with_deferred_injection"
    assert deferred.injection_report.n_injections == 2
    assert deferred.injection_report.projection_order == "input"
    assert deferred.projection_elapsed_s >= 0.0
    assert len(deferred.deferred_projection_events) == 2
    assert _fidelity(
        deferred.simulator.to_statevector(),
        _data_marginal_ref(direct.simulator, 2),
    ) == pytest.approx(1.0, abs=1e-6)


def test_run_stabilizer_mps_stream_recommended_mode_is_explicit():
    stream = [("h", 0), ("t", 0)]

    default = run_stabilizer_mps_stream(stream, n_qubits=1)
    recommended = run_stabilizer_mps_stream(
        stream,
        n_qubits=1,
        mode="recommended",
        n_ancilla=1,
        seed=2,
    )

    assert default.mode == "direct"
    assert recommended.requested_mode == "recommended"
    assert recommended.mode == "immediate"
    assert recommended.injection_report.n_injections == 1
    assert _fidelity(
        recommended.simulator.to_statevector(),
        _data_marginal_ref(default.simulator, 1),
    ) == pytest.approx(1.0, abs=1e-6)


def test_from_stim_queue_can_use_settings_advice_and_runner():
    def add_t_after_stim_prefix(stream):
        return [*stream[:-1], ("t", 0)]

    sim = MpsStabOptimizer.from_stim(
        "H 0\nM 0",
        seed=7,
        stream_transform=add_t_after_stim_prefix,
    )
    analysis = sim.queued_stream_analysis()
    advice = sim.queued_recommend_settings(goal="validate")

    result = sim.run_queued_stream(mode="direct", goal="validate")
    expected = MpsStabOptimizer(1).apply([("h", 0), ("t", 0)])

    assert analysis.total_entries == 2
    assert advice.recommended_mode == "immediate"
    assert result.mode == "direct"
    assert result.remaining_queue == 0
    assert sim.queued_stream_analysis().total_entries == analysis.total_entries
    assert _fidelity(
        result.simulator.to_statevector(),
        expected.to_statevector(),
    ) == pytest.approx(1.0, abs=1e-6)


def test_run_with_injection_records_projection_costs():
    sim = MpsStabOptimizer(3)
    sim.run_with_injection([("t", 0), ("tdg", 1)], ancillas=[2])

    report = sim.last_immediate_injection_report
    assert isinstance(report, ImmediateInjectionReport)
    assert report["n_injections"] == 2
    assert len(sim.immediate_projection_events) == 2
    assert isinstance(sim.immediate_projection_events[0], ImmediateProjectionRecord)
    assert sim.immediate_projection_events[0].ancilla == 2
    assert report["projection_elapsed_s"] == pytest.approx(sum(
        event["elapsed_s"] for event in sim.immediate_projection_events
    ))
    assert report["projection_peak_bond"] >= 1


@pytest.mark.parametrize("projection_order", ["input", "middle_out", "min_span"])
def test_deferred_injection_matches_direct_circuit(projection_order):
    n_data = 3
    stream = [
        ("h", 0), ("cnot", 0, 1), ("t", 2), ("tdg", 0),
        ("rz", np.pi / 4, 1), ("cnot", 1, 2), ("t", 1),
        ("rz", np.pi / 2, 0),
    ]
    outcomes = [+1, -1, +1, -1]
    direct = MpsStabOptimizer(n_data).apply(stream)
    deferred = MpsStabOptimizer.with_deferred_injection(
        n_data,
        stream,
        outcomes=outcomes,
        projection_order=projection_order,
    )

    assert deferred.n == n_data + len(outcomes)
    assert _fidelity(
        deferred.to_statevector(), _data_marginal_ref(direct, len(outcomes))
    ) == pytest.approx(1.0, abs=1e-6)
    assert [event["outcome"] for event in sorted(
        deferred.deferred_projection_events, key=lambda event: event["index"]
    )] == outcomes
    assert isinstance(deferred.deferred_projection_events[0], DeferredProjectionRecord)
    assert isinstance(deferred.last_deferred_injection_report, DeferredInjectionReport)
    assert deferred.last_deferred_injection_report["n_injections"] == len(outcomes)
    assert deferred.last_deferred_injection_report["projection_elapsed_s"] >= 0.0
    for ancilla in range(n_data, deferred.n):
        assert deferred.expectation("Z", ancilla) == pytest.approx(1.0, abs=1e-9)


def test_deferred_injection_auto_layout_matches_direct_circuit():
    n_data = 3
    stream = [
        ("h", 0), ("cnot", 0, 1), ("t", 2), ("tdg", 0),
        ("rz", np.pi / 4, 1), ("cnot", 1, 2), ("t", 1),
    ]
    outcomes = [+1, -1, +1, -1]
    direct = MpsStabOptimizer(n_data).apply(stream)
    deferred = MpsStabOptimizer.with_deferred_injection(
        n_data,
        stream,
        outcomes=outcomes,
        projection_order="input",
        layout="auto",
        layout_report=False,
    )

    assert deferred.layout_plan is not None
    assert deferred.layout_plan["source"] == "queued_frame_supports"
    assert _fidelity(
        deferred.to_statevector(),
        _data_marginal_ref(direct, len(outcomes)),
    ) == pytest.approx(1.0, abs=1e-6)


def test_deferred_injection_accepts_an_explicit_projection_order():
    n_data = 2
    stream = [("h", 0), ("t", 0), ("tdg", 1), ("t", 1)]
    direct = MpsStabOptimizer(n_data).apply(stream)
    ancillas = [2, 3, 4]
    deferred = MpsStabOptimizer(n_data + len(ancillas))
    deferred.run_with_deferred_injection(
        stream,
        ancillas=ancillas,
        outcomes=[-1, +1, -1],
        projection_order=list(reversed(ancillas)),
    )

    assert [event["ancilla"] for event in deferred.deferred_projection_events] == list(
        reversed(ancillas)
    )
    assert _fidelity(
        deferred.to_statevector(), _data_marginal_ref(direct, len(ancillas))
    ) == pytest.approx(1.0, abs=1e-6)


def test_deferred_injection_middle_out_projects_each_odd_register_ancilla_once():
    stream = [("h", 0), ("t", 0), ("tdg", 1), ("t", 1)]
    deferred = MpsStabOptimizer.with_deferred_injection(
        2,
        stream,
        outcomes=[+1, -1, +1],
        projection_order="middle_out",
    )

    assert [event["index"] for event in deferred.deferred_projection_events] == [1, 0, 2]
    assert len({event["ancilla"] for event in deferred.deferred_projection_events}) == 3


def test_deferred_injection_requires_a_fresh_ancilla_per_gate():
    with pytest.raises(ValueError, match="ancilla per injectable gate"):
        MpsStabOptimizer.with_deferred_injection(
            2, [("t", 0), ("t", 1)], n_ancilla=1
        )


def test_deferred_injection_rejects_ordinary_entry_touching_reserved_pool():
    with pytest.raises(ValueError, match="ordinary stream entry"):
        MpsStabOptimizer.with_deferred_injection(
            2, [("h", 2), ("t", 0)], n_ancilla=1
        )


# --------------------------------------------------------------------------- #
# Backend / GPU: |nu> and gates on a torch backend (skips without torch)
# --------------------------------------------------------------------------- #
def _torch_backend():
    torch = pytest.importorskip("torch")
    import pepsy as py
    return py.backend_torch(dtype=torch.complex128, device="cpu")


def test_torch_backend_matches_numpy():
    tb = _torch_backend()
    stream = [("h", 0), ("cnot", 0, 1), ("t", 2), ("rz", np.pi / 4, 1),
              ("ry", 0.7, 3), ("cnot", 2, 3), ("tdg", 0)]
    cpu = MpsStabOptimizer(4, seed=0).apply(stream)
    gpu = MpsStabOptimizer(4, seed=0, to_backend=tb).apply(stream)
    # |nu> tensors live on the torch backend
    assert type(gpu.state.p[0].data).__module__.split(".")[0] == "torch"
    assert _fidelity(cpu.to_statevector(), gpu.to_statevector()) == pytest.approx(1.0, abs=1e-6)
    for axis in ("X", "Y", "Z"):
        for q in range(4):
            assert gpu.expectation(axis, q) == pytest.approx(cpu.expectation(axis, q), abs=1e-6)
    assert gpu.norm() == pytest.approx(1.0, abs=1e-9)


def test_torch_backend_absorb_measure_matches_numpy():
    tb = _torch_backend()
    circ = [("h", 0), ("cnot", 0, 1), ("rz", 0.7, 1), ("ry", 0.9, 2)]
    for m in (+1, -1):
        cpu = MpsStabOptimizer(3).apply(circ)
        cpu.measure("Z", 1, outcome=m, absorb_basis=True)
        gpu = MpsStabOptimizer(3, to_backend=tb).apply(circ)
        gpu.measure("Z", 1, outcome=m, absorb_basis=True)
        assert _fidelity(cpu.to_statevector(), gpu.to_statevector()) == pytest.approx(1.0, abs=1e-6)


def test_torch_backend_matrix_gate_input():
    tb = _torch_backend()
    import torch
    # A torch-native (non-unitary) gate matrix passed as an explicit
    # (matrix, where) entry is materialized on the CPU for classification.
    coin = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.complex128)
    sim = MpsStabOptimizer(2, to_backend=tb).apply([(coin, 0)])
    ref = _apply_gate_dense(
        np.array([1, 0, 0, 0], complex),
        np.array([[0.9, 0.1], [0.1, 0.9]], complex), (0,), 2,
    )
    assert _fidelity(sim.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


def test_native_mps_backend_is_inferred_and_foreign_payloads_are_diagnosed():
    torch = pytest.importorskip("torch")
    backend = _torch_backend()
    p = qtn.MatrixProductState.from_dense(
        np.array([1, 0, 0, 0], dtype=complex), dims=[2, 2]
    )
    p.apply_to_arrays(backend)

    sim = MpsStabOptimizer.from_mps(p)
    assert sim.backend_info() == {
        "backend": "torch",
        "dtype": "complex128",
        "device": "cpu",
    }
    assert sim.backend == "torch"
    assert sim.backend_dtype == "complex128"
    assert sim.backend_device == "cpu"

    with pytest.warns(UserWarning, match="gate payload"):
        sim.apply([(np.diag([1.0, np.exp(0.1j)]), 0)])
    assert isinstance(sim.p[0].data, torch.Tensor)
    sim.cap(0, [1.0, 0.0])
    assert sim.backend_info()["backend"] == "torch"
    assert isinstance(sim.p[0].data, torch.Tensor)


def test_native_mps_submpo_conversion_does_not_mutate_source():
    backend = _torch_backend()
    p = qtn.MatrixProductState.from_dense(
        np.array([1, 0, 0, 0], dtype=complex), dims=[2, 2]
    )
    p.apply_to_arrays(backend)
    sim = MpsStabOptimizer.from_mps(p)
    mpo = pauli_rotation_mpo(0.2, ["X", "Z"])
    source_types = tuple(type(tensor.data) for tensor in mpo.tensors)

    with pytest.warns(UserWarning, match="sub-MPO payload"):
        sim.apply([("submpo", mpo, (0, 1))])

    assert tuple(type(tensor.data) for tensor in mpo.tensors) == source_types
    assert "torch" in type(sim.p[0].data).__module__


def test_torch_backend_injection_and_sampling():
    tb = _torch_backend()
    # injection on the torch backend reproduces T
    sim = MpsStabOptimizer(2, to_backend=tb)
    sim.state.h(0)
    sim.prepare_magic(1)
    sim.inject_t(0, 1, outcome=+1)
    ref = MpsStabOptimizer(1).apply([("h", 0), ("t", 0)])
    full = sim.to_statevector().reshape(2, 2)
    dv = full[:, 0] if np.linalg.norm(full[:, 0]) > np.linalg.norm(full[:, 1]) else full[:, 1]
    assert _fidelity(dv, ref.to_statevector()) == pytest.approx(1.0, abs=1e-6)
    # probability_bits and sampling work off the backend state
    circ = [("h", 0), ("cnot", 0, 1), ("t", 1)]
    gpu = MpsStabOptimizer(2, to_backend=tb).apply(circ)
    cpu = MpsStabOptimizer(2).apply(circ)
    for k in range(4):
        b = format(k, "02b")
        assert gpu.probability_bits(b) == pytest.approx(cpu.probability_bits(b), abs=1e-6)
    s = gpu.sample_bits(64, seed=0)
    assert s.shape == (64, 2) and set(np.unique(s)).issubset({0, 1})


def _jax_backend():
    jax = pytest.importorskip("jax")
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:  # pragma: no cover - depends on JAX import state
        pass
    import jax.numpy as jnp
    import pepsy as py
    return py.backend_jax(device="cpu", dtype=jnp.complex128)


def _cupy_backend():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy installed but no CUDA device is available.")
    except Exception as exc:  # pragma: no cover - driver/runtime dependent
        pytest.skip(f"CuPy CUDA runtime is unavailable: {exc}")
    import pepsy as py
    return py.backend_cupy(dtype=cp.complex128)


@pytest.mark.parametrize(
    ("backend_name", "backend_factory", "module_token"),
    [
        ("jax", _jax_backend, "jax"),
        ("cupy", _cupy_backend, "cupy"),
    ],
)
def test_optional_array_backends_match_numpy_for_stn_paths(
    backend_name,
    backend_factory,
    module_token,
):
    backend = backend_factory()
    stream = [
        ("h", 0),
        ("cnot", 0, 1),
        ("rz", 0.7, 1),
        ("ry", 0.4, 2),
        ("t", 0),
    ]
    cpu = MpsStabOptimizer(3, seed=0).apply(stream)
    other = MpsStabOptimizer(3, seed=0, to_backend=backend).apply(stream)
    assert module_token in type(other.state.p[0].data).__module__
    assert _fidelity(cpu.to_statevector(), other.to_statevector()) == pytest.approx(
        1.0, abs=1e-6
    )

    for outcome in (+1, -1):
        cpu_m = MpsStabOptimizer(3).apply(stream)
        other_m = MpsStabOptimizer(3, to_backend=backend).apply(stream)
        cpu_m.measure("Z", 1, outcome=outcome, absorb_basis=True)
        other_m.measure("Z", 1, outcome=outcome, absorb_basis=True)
        assert _fidelity(
            cpu_m.to_statevector(), other_m.to_statevector()
        ) == pytest.approx(1.0, abs=1e-6)

    inj = MpsStabOptimizer(2, to_backend=backend)
    inj.state.h(0)
    inj.prepare_magic(1)
    inj.inject_t(0, 1, outcome=+1)
    ref = MpsStabOptimizer(1).apply([("h", 0), ("t", 0)])
    full = inj.to_statevector().reshape(2, 2)
    data_vec = (
        full[:, 0]
        if np.linalg.norm(full[:, 0]) >= np.linalg.norm(full[:, 1])
        else full[:, 1]
    )
    assert _fidelity(data_vec, ref.to_statevector()) == pytest.approx(1.0, abs=1e-6)

    samples = other.sample_bits(16, seed=0)
    assert samples.shape == (16, 3)
    assert set(np.unique(samples)).issubset({0, 1})


# --------------------------------------------------------------------------- #

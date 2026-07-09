"""Tests for the stabilizer-tensor-network state container and Clifford update.

Covers Phase 1 (state container + statevector reconstruction) and Phase 2
(Clifford update: basis-only, |nu> unchanged) of the STN build
(arXiv:2403.08724).
"""

import numpy as np
import pytest

stim = pytest.importorskip("stim")

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
from pepsy.optimizers.stabilizer_tn import MpsStabOptimizer, pauli_rotation_mpo  # noqa: E402

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


def test_simulator_submpo_event_in_nu_frame():
    # A sub-MPO event acts directly on the coefficient MPS p; from |0...0> the
    # basis is identity so it also equals the physical operator.
    sim = MpsStabOptimizer(3)
    mpo = pauli_rotation_mpo(0.8, ["X", "I", "Z"], sign=1.0)
    exact = mpo.apply(sim.state.p).to_dense().reshape(-1)
    sim.apply([("submpo", mpo, (0, 1, 2))])
    got = np.asarray(sim.state.p.to_dense()).reshape(-1)
    assert _fidelity(got, exact) == pytest.approx(1.0, abs=1e-9)


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
    assert all(inf >= 0.0 for inf in sim.infidelities)
    assert max(sim.infidelities) > 0.0  # truncation actually occurred


def test_simulator_two_qubit_nonclifford_matrix_supported():
    # A dense non-Clifford 2q matrix is now applied via Pauli decomposition.
    sim = MpsStabOptimizer(3).apply([("h", 0), ("t", 2)])
    psi = sim.to_statevector()
    u = _rzz(0.5)  # non-Clifford 2q matrix
    sim.apply([(u, (0, 1))])
    ref = _apply_gate_dense(psi, u, (0, 1), 3)
    assert _fidelity(sim.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


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


def test_from_tableau_and_nu_roundtrip():
    src = MpsStabOptimizer(3).apply([("h", 0), ("cnot", 0, 1), ("rz", 0.6, 2), ("t", 1)])
    # Primary API is from_tableau_and_state / .p; the *_nu names remain aliases.
    rebuilt = MpsStabOptimizer.from_tableau_and_state(src.state._sim.copy(), src.state.p.copy())
    assert _fidelity(rebuilt.to_statevector(), src.to_statevector()) == pytest.approx(1.0, abs=1e-6)
    alias = MpsStabOptimizer.from_tableau_and_nu(src.state._sim.copy(), src.state.nu.copy())
    assert _fidelity(alias.to_statevector(), src.to_statevector()) == pytest.approx(1.0, abs=1e-6)


def test_norm_preserved_after_circuit():
    sim = MpsStabOptimizer(3).apply(
        [("h", 0), ("cnot", 0, 1), ("rz", 0.7, 1), ("ry", 0.9, 2), ("t", 0)]
    )
    assert sim.norm() == pytest.approx(1.0, abs=1e-9)
    # and after a measurement, still normalized
    sim.measure("Z", 0)
    assert sim.norm() == pytest.approx(1.0, abs=1e-9)


def test_run_progbar_smoke():
    pytest.importorskip("tqdm")
    sim = MpsStabOptimizer(3)
    sim.set_gates([("h", 0), ("cnot", 0, 1), ("rz", 0.5, 2), ("t", 1)]).run(progbar=True)
    assert sim.state.max_bond() >= 1


def test_stabilizermps_backward_alias():
    from pepsy.optimizers.stabilizer_tn import StabilizerMps
    assert StabilizerMps is MpsStabOptimizer


def test_optimizers_namespace_exports():
    from pepsy.optimizers import MpsStabOptimizer as M, STNState as S
    assert M is MpsStabOptimizer
    assert S is STNState


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


def test_probability_bits_does_not_mutate_state():
    sim = MpsStabOptimizer(3).apply([("h", 0), ("cnot", 0, 1), ("t", 2)])
    before = sim.to_statevector()
    sim.probability_bits("010")
    assert _fidelity(sim.to_statevector(), before) == pytest.approx(1.0, abs=1e-9)


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
    with pytest.raises(ValueError, match="0 probability"):
        sim.measure("Z", 1, outcome=-1, absorb_basis=True)


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


# --------------------------------------------------------------------------- #
# Benchmark smoke test (keeps benchmarks/stabilizer_tn_magic_scaling.py working)
# --------------------------------------------------------------------------- #
def _load_magic_scaling_benchmark():
    import importlib.util
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "benchmarks" / "stabilizer_tn_magic_scaling.py")
    spec = importlib.util.spec_from_file_location("stn_magic_scaling_bench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_magic_scaling_benchmark_smoke():
    bench = _load_magic_scaling_benchmark()
    # circuit generator produces exactly t T-gates, independent of n
    stream = bench.random_clifford_t_circuit(n=5, t=3, depth=4, seed=0)
    assert sum(1 for e in stream if e[0] == "t") == 3
    # both modes keep the |nu> bond bounded by 2^t
    for mode in ("direct", "injection"):
        res = bench.run_case(n=5, t=3, depth=4, seed=0, chi=None,
                             mode=mode, to_backend=None, rank_max_n=8)
        assert res["max_nu_bond"] <= 2 ** 3
        assert res["pseudo_stabilizer_rank"] is not None


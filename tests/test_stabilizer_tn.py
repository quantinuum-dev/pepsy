"""Tests for the stabilizer-tensor-network state container and Clifford update.

Covers Phase 1 (state container + statevector reconstruction) and Phase 2
(Clifford update: basis-only, |nu> unchanged) of the STN build
(arXiv:2403.08724).
"""

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from pepsy.stabilizer_tn import STNState


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
    # |nu> = |000> = e0
    np.testing.assert_allclose(st.nu_dense(), np.eye(8)[0])
    # |psi> = C|nu> = |000>
    assert _fidelity(st.to_statevector(), np.eye(8)[0]) == pytest.approx(1.0)
    assert st.pseudo_stabilizer_rank() == 1


def test_bell_state_ground_truth():
    """H(0), CNOT(0,1) prepares (|00> + |11>)/sqrt(2), independent of stim."""
    st = STNState(2).h(0).cnot(0, 1)
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert _fidelity(st.to_statevector(), expected) == pytest.approx(1.0)
    # Clifford gates do not touch |nu>.
    assert st.max_bond() == 1
    np.testing.assert_allclose(st.nu_dense(), np.eye(4)[0])


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
from pepsy.stabilizer_tn import MpsStabOptimizer, pauli_rotation_mpo  # noqa: E402

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
    # A sub-MPO event acts directly on |nu>; from |0...0> the basis is identity
    # so it also equals the physical operator.
    sim = MpsStabOptimizer(3)
    mpo = pauli_rotation_mpo(0.8, ["X", "I", "Z"], sign=1.0)
    exact = mpo.apply(sim.state.nu).to_dense().reshape(-1)
    sim.apply([("submpo", mpo, (0, 1, 2))])
    got = np.asarray(sim.state.nu.to_dense()).reshape(-1)
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


def test_simulator_two_qubit_nonclifford_matrix_rejected():
    sim = MpsStabOptimizer(2)
    bad = _rzz(0.5)  # non-Clifford 2q matrix
    with pytest.raises(NotImplementedError):
        sim.apply([(bad, (0, 1))])


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
    np.testing.assert_allclose(base.nu_dense(), np.eye(8)[0])
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
    rebuilt = MpsStabOptimizer.from_tableau_and_nu(src.state._sim.copy(), src.state.nu.copy())
    assert _fidelity(rebuilt.to_statevector(), src.to_statevector()) == pytest.approx(1.0, abs=1e-6)


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
    from pepsy.stabilizer_tn import StabilizerMps
    assert StabilizerMps is MpsStabOptimizer


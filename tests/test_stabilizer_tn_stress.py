"""Stress / differential tests for :class:`MpsStabOptimizer`.

These validate challenging cases against *independent* oracles:

* ``quimb.tensor.Circuit`` — an independent statevector simulator, used for
  random and long-range unitary circuits and per-bitstring amplitudes.
* ``stim`` — deep pure-Clifford circuits.
* a small dense simulator — mid-circuit (forced) measurements.

Conventions were verified to match (fidelity 1.0) for H/S/SDG/X/Y/Z/CNOT/CZ/
SWAP/RX/RY/RZ/RXX/RYY/RZZ/T/TDG and the big-endian amplitude ordering.
"""

import numpy as np
import pytest

stim = pytest.importorskip("stim")
qtn = pytest.importorskip("quimb.tensor")

from pepsy.optimizers import MpsStabOptimizer  # noqa: E402

# --------------------------------------------------------------------------- #
# Dense building blocks
# --------------------------------------------------------------------------- #
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_P1 = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def _fid(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def _rot(axis, theta):
    return np.cos(theta / 2) * _I - 1j * np.sin(theta / 2) * _P1[axis]


def _apply_gate_dense(psi, u, where, n):
    where = list(where)
    k = len(where)
    t = psi.reshape([2] * n)
    u = u.reshape([2] * k + [2] * k)
    t = np.tensordot(u, t, axes=(list(range(k, 2 * k)), where))
    t = np.moveaxis(t, list(range(k)), where)
    return t.reshape(-1)


def _pauli_op(pauli, where, n):
    axes = list(pauli)
    if where is None:
        where = tuple(range(n))
    elif isinstance(where, int):
        where = (where,)
    else:
        where = tuple(where)
    full = [np.eye(2, dtype=complex) for _ in range(n)]
    for ax, q in zip(axes, where):
        full[q] = _P1[ax]
    out = full[0]
    for m in full[1:]:
        out = np.kron(out, m)
    return out


_CNOT4 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)
_CZ4 = np.diag([1, 1, 1, -1]).astype(complex)
_SWAP4 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], complex)


def _gate_mat(e):
    nm = e[0]
    one = {
        "h": _H, "x": _X, "y": _Y, "z": _Z,
        "s": np.diag([1, 1j]).astype(complex),
        "sdg": np.diag([1, -1j]).astype(complex),
        "t": _rot("Z", np.pi / 4), "tdg": _rot("Z", -np.pi / 4),
    }
    if nm in one:
        return one[nm], (e[1],)
    if nm in ("cnot", "cx"):
        return _CNOT4, (e[1], e[2])
    if nm == "cz":
        return _CZ4, (e[1], e[2])
    if nm == "swap":
        return _SWAP4, (e[1], e[2])
    if nm in ("rx", "ry", "rz"):
        return _rot({"rx": "X", "ry": "Y", "rz": "Z"}[nm], e[1]), (e[2],)
    if nm in ("rxx", "ryy", "rzz"):
        ax = {"rxx": "X", "ryy": "Y", "rzz": "Z"}[nm]
        pp = np.kron(_P1[ax], _P1[ax])
        return np.cos(e[1] / 2) * np.eye(4) - 1j * np.sin(e[1] / 2) * pp, (e[2], e[3])
    raise AssertionError(f"unhandled gate {nm}")


def _dense_run(n, stream):
    """Dense statevector, supporting forced measurements ('measure', p, where, m)."""
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    for e in stream:
        if e[0] == "measure":
            _, pauli, where, m = e
            proj = (np.eye(2 ** n) + m * _pauli_op(pauli, where, n)) / 2
            psi = proj @ psi
            psi = psi / np.linalg.norm(psi)
        else:
            u, where = _gate_mat(e)
            psi = _apply_gate_dense(psi, u, where, n)
    return psi


# --------------------------------------------------------------------------- #
# quimb.tensor.Circuit oracle
# --------------------------------------------------------------------------- #
_Q1 = {"h": "H", "s": "S", "sdg": "SDG", "x": "X", "y": "Y", "z": "Z", "t": "T", "tdg": "TDG"}
_Q2 = {"cnot": "CNOT", "cx": "CNOT", "cz": "CZ", "cy": "CY", "swap": "SWAP"}


def _quimb_circuit(n, stream):
    circ = qtn.Circuit(n)
    for e in stream:
        nm = e[0]
        if nm in _Q1:
            circ.apply_gate(_Q1[nm], e[1])
        elif nm in _Q2:
            circ.apply_gate(_Q2[nm], e[1], e[2])
        elif nm in ("rx", "ry", "rz"):
            circ.apply_gate(nm.upper(), e[1], e[2])
        elif nm in ("rxx", "ryy", "rzz"):
            circ.apply_gate(nm.upper(), e[1], e[2], e[3])
        else:
            raise AssertionError(f"quimb oracle cannot handle {nm}")
    return circ


def _quimb_sv(n, stream):
    return np.asarray(_quimb_circuit(n, stream).to_dense()).reshape(-1)


# --------------------------------------------------------------------------- #
# Random circuit generator (challenging: long-range + non-Clifford)
# --------------------------------------------------------------------------- #
def _pair(n, rng, long_range):
    if long_range and rng.random() < 0.5:
        return 0, n - 1
    a, b = rng.choice(n, size=2, replace=False)
    return int(a), int(b)


def _random_stream(n, depth, rng, *, long_range=True, non_clifford=True):
    one_c = ["h", "s", "sdg", "x", "y", "z"]
    two_c = ["cnot", "cz", "swap"]
    stream = []
    for _ in range(depth):
        r = rng.random()
        if r < 0.25:
            a, b = _pair(n, rng, long_range)
            stream.append((str(rng.choice(two_c)), a, b))
        elif r < 0.5 or not non_clifford:
            stream.append((str(rng.choice(one_c)), int(rng.integers(n))))
        elif r < 0.7:
            ax = str(rng.choice(["rx", "ry", "rz"]))
            stream.append((ax, float(rng.uniform(0.15, 1.4)), int(rng.integers(n))))
        elif r < 0.85:
            a, b = _pair(n, rng, long_range)
            stream.append((str(rng.choice(["rxx", "ryy", "rzz"])), float(rng.uniform(0.15, 1.4)), a, b))
        else:
            stream.append((str(rng.choice(["t", "tdg"])), int(rng.integers(n))))
    return stream


# --------------------------------------------------------------------------- #
# Differential tests vs quimb Circuit (exact mode)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(8))
def test_random_circuit_matches_quimb(seed):
    rng = np.random.default_rng(seed)
    n = 6
    stream = _random_stream(n, 80, rng)
    sim = MpsStabOptimizer(n).apply(stream)  # exact (chi=None)
    assert _fid(sim.to_statevector(), _quimb_sv(n, stream)) == pytest.approx(1.0, abs=1e-5)
    assert sim.norm() == pytest.approx(1.0, abs=1e-8)


def test_long_range_rotations_match_quimb():
    n = 6
    stream = [("h", q) for q in range(n)] + [("cnot", i, i + 1) for i in range(n - 1)]
    stream += [("rzz", 0.9, 0, n - 1), ("rxx", 0.7, 0, n - 1),
               ("ry", 0.5, 0), ("rz", 0.4, n - 1), ("rzz", 1.1, 1, n - 2)]
    sim = MpsStabOptimizer(n).apply(stream)
    assert _fid(sim.to_statevector(), _quimb_sv(n, stream)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("seed", [1, 4, 9])
def test_amplitude_probabilities_match_quimb(seed):
    rng = np.random.default_rng(seed)
    n = 4
    stream = _random_stream(n, 45, rng)
    sim = MpsStabOptimizer(n).apply(stream)
    circ = _quimb_circuit(n, stream)
    for k in range(2 ** n):
        bits = format(k, f"0{n}b")
        assert sim.probability(bits) == pytest.approx(abs(circ.amplitude(bits)) ** 2, abs=1e-5)


def test_high_entanglement_clifford_plus_T_vs_quimb():
    # A random Clifford circuit produces a highly entangled physical state, yet
    # |nu> stays low-bond until the T gates: bond(|nu>) <= 2^t.
    n = 8
    rng = np.random.default_rng(2)
    clifford = _random_stream(n, 120, rng, non_clifford=False)
    n_t = 2
    tgates = [("t", int(rng.integers(n))) for _ in range(n_t)]
    stream = clifford + tgates
    sim = MpsStabOptimizer(n).apply(stream)
    assert _fid(sim.to_statevector(), _quimb_sv(n, stream)) == pytest.approx(1.0, abs=1e-4)
    assert sim.state.max_bond() <= 2 ** n_t


# --------------------------------------------------------------------------- #
# Deep Clifford vs stim
# --------------------------------------------------------------------------- #
def test_deep_clifford_matches_stim_and_stays_chi_one():
    n = 8
    rng = np.random.default_rng(123)
    stream = _random_stream(n, 250, rng, non_clifford=False)
    sim = MpsStabOptimizer(n).apply(stream)
    assert sim.state.max_bond() == 1  # Clifford-only never entangles |nu>
    ssim = stim.TableauSimulator()
    ssim.set_num_qubits(n)
    meth = {"h": "h", "s": "s", "sdg": "s_dag", "x": "x", "y": "y", "z": "z",
            "cnot": "cnot", "cz": "cz", "swap": "swap"}
    for e in stream:
        getattr(ssim, meth[e[0]])(*e[1:])
    ref = np.asarray(ssim.state_vector(endian="big")).reshape(-1)
    assert _fid(sim.to_statevector(), ref) == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# Truncation convergence to the exact result
# --------------------------------------------------------------------------- #
def test_truncation_converges_to_exact():
    n = 6
    rng = np.random.default_rng(7)
    stream = [("h", q) for q in range(n)] + [("cnot", i, i + 1) for i in range(n - 1)]
    stream += [("rz", float(rng.uniform(0.3, 1.2)), q) for q in range(n)]
    stream += [("cnot", i, i + 1) for i in range(n - 1)]
    stream += [("rx", float(rng.uniform(0.3, 1.2)), q) for q in range(n)]

    exact = MpsStabOptimizer(n).apply(stream)
    chi_exact = exact.state.max_bond()
    assert chi_exact > 1  # |nu> is genuinely entangled here
    sv_exact = exact.to_statevector()

    chis = [1, 2, 4, chi_exact]
    fids = []
    for chi in chis:
        approx = MpsStabOptimizer(n, chi=chi, track_infidelity=True).apply(stream)
        fids.append(_fid(approx.to_statevector(), sv_exact))
        assert approx.state.max_bond() <= chi
        # recorded infidelities are valid probabilities
        assert all(0.0 <= inf <= 1.0 for inf in approx.infidelities)

    # Non-decreasing with chi and exact at full bond.
    for lo, hi in zip(fids, fids[1:]):
        assert lo <= hi + 1e-6
    assert fids[-1] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Mid-circuit measurements vs dense (consistent forced outcomes)
# --------------------------------------------------------------------------- #
def test_midcircuit_measurements_match_dense():
    n = 4
    rng = np.random.default_rng(5)
    stream = (
        _random_stream(n, 20, rng)
        + [("measure", "Z", 1)]
        + _random_stream(n, 12, rng)
        + [("measure", "XZ", (0, 2))]
        + _random_stream(n, 12, rng)
        + [("measure", "Y", 3)]
    )
    sim = MpsStabOptimizer(n, seed=0).apply(stream)
    assert sim.norm() == pytest.approx(1.0, abs=1e-8)

    outcomes = iter(m for (_, _, m) in sim.measurements)
    dense_stream = []
    for e in stream:
        if e[0] == "measure":
            dense_stream.append(("measure", e[1], e[2], next(outcomes)))
        else:
            dense_stream.append(e)
    psi = _dense_run(n, dense_stream)
    assert _fid(sim.to_statevector(), psi) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Born-rule sampling statistics
# --------------------------------------------------------------------------- #
def test_born_sampling_frequencies_match_expectation():
    n = 3
    sim = MpsStabOptimizer(n, seed=0).apply(
        [("h", 0), ("cnot", 0, 1), ("rz", 0.7, 2), ("ry", 0.9, 2), ("rx", 0.4, 0)]
    )
    for pauli, where in [("Z", 0), ("Z", 2), ("X", 1), ("Y", 2)]:
        exp = sim.expectation(pauli, where)
        outs = sim.sample(pauli, where, shots=4000)
        assert set(np.unique(outs)).issubset({-1, 1})
        assert outs.mean() == pytest.approx(exp, abs=0.06)


# --------------------------------------------------------------------------- #
# General dense k-qubit gates (unitary + non-unitary) via Pauli decomposition
# --------------------------------------------------------------------------- #
def _haar_unitary(dim, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def test_general_two_qubit_unitary_matches_dense():
    n = 4
    stream = [("h", 0), ("cnot", 0, 1), ("t", 2), ("rx", 0.7, 3), ("cz", 1, 2)]
    sim = MpsStabOptimizer(n).apply(stream)
    psi = sim.to_statevector()
    U = _haar_unitary(4, 1)
    sim.apply([(U, (1, 3))])  # non-adjacent 2q unitary
    ref = _apply_gate_dense(psi, U, (1, 3), n)
    assert _fid(sim.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


def test_general_two_qubit_nonunitary_matches_dense():
    n = 4
    stream = [("h", 0), ("cnot", 0, 1), ("t", 2), ("ry", 0.4, 3)]
    sim = MpsStabOptimizer(n).apply(stream)
    psi = sim.to_statevector()
    rng = np.random.default_rng(3)
    G = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    sim.apply([(G, (0, 2))])  # arbitrary non-unitary 2q gate
    ref = _apply_gate_dense(psi, G, (0, 2), n)
    got = sim.to_statevector()
    assert _fid(got, ref) == pytest.approx(1.0, abs=1e-6)
    # non-unitary gate: represented-state norm tracks |G|psi>| (not renormalized)
    assert np.linalg.norm(got) == pytest.approx(np.linalg.norm(ref), rel=1e-6)


def test_general_three_qubit_unitary_matches_dense():
    n = 5
    stream = [("h", 0), ("cnot", 0, 1), ("t", 2), ("rx", 0.3, 3), ("cz", 2, 4)]
    sim = MpsStabOptimizer(n).apply(stream)
    psi = sim.to_statevector()
    U3 = _haar_unitary(8, 7)
    sim.apply([(U3, (1, 2, 4))])
    ref = _apply_gate_dense(psi, U3, (1, 2, 4), n)
    assert _fid(sim.to_statevector(), ref) == pytest.approx(1.0, abs=1e-6)


def test_single_qubit_nonunitary_matches_dense():
    n = 3
    sim = MpsStabOptimizer(n).apply([("h", 0), ("cnot", 0, 1), ("t", 2)])
    psi = sim.to_statevector()
    rng = np.random.default_rng(11)
    G = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    sim.apply([(G, 2)])
    ref = _apply_gate_dense(psi, G, (2,), n)
    got = sim.to_statevector()
    assert _fid(got, ref) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(got) == pytest.approx(np.linalg.norm(ref), rel=1e-6)

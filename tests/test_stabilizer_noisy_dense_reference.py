"""Conditioned dense references for noise, magic, reset and classical feedback."""

import numpy as np
import pytest

from pepsy import build_backend
from pepsy.optimizers import StabilizerMpsSimulator

pytest.importorskip("stim")


def _apply(vector, gate, sites):
    order = tuple(sites) + tuple(i for i in range(3) if i not in sites)
    block = vector.reshape((2,) * 3).transpose(order).reshape(2**len(sites), -1)
    return (gate @ block).reshape((2,) * 3).transpose(np.argsort(order)).ravel()


X = np.array([[0., 1.], [1., 0.]])
Z = np.diag([1., -1.])
H = np.array([[1., 1.], [1., -1.]]) / 2**.5
T = np.diag([1., np.exp(1j * np.pi / 4)])
CX = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
               [0., 0., 0., 1.], [0., 0., 1., 0.]])


def _condition(vector, operator, site, outcome):
    selected = (vector + outcome * _apply(vector, operator, (site,))) / 2
    probability = float(np.vdot(selected, selected).real)
    assert probability > 0
    return selected / probability**.5, probability


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("strategy", ["independent", "coalesced", "auto"])
@pytest.mark.parametrize("gamma", [.17, 1e-30])
def test_noisy_magic_feedback_matches_dense_conditional_states_and_weights(backend, strategy, gamma):
    converter = None
    if backend == "torch":
        torch = pytest.importorskip("torch")
        converter = build_backend(device="cpu", dtype=torch.complex128, set_default=False)
    stream = [
        ("measure", "Z", 2),
        ("h", 0), ("cnot", 0, 1), ("t", 0),
        ("amplitude_damping", gamma, 0), ("measure", "X", 0, None, True),
        ("if", -1, 1, ("z", 1)), ("measure_reset", "Z", 0), ("t", 1),
        ("x_error", .07, 2), ("cnot", 1, 2), ("measure", "X", 1),
        ("if", -1, 1, ("x", 2)), ("reset", 1, "X"), ("measure", "Z", 2),
    ]
    engine = StabilizerMpsSimulator(3, chi=8, exact_cooling=False, to_backend=converter)
    engine.apply_layout((2, 0, 1), layout_report=False).compile(stream)

    def proposal(event_index, labels, target, optimizer):
        assert optimizer is not None
        return np.full(len(labels), 1 / len(labels))

    result = engine.run(shots=32, strategy=strategy, workers=2, seed=29,
                        progress=False, importance_sampling=proposal)
    assert sum(result.counts) == 32
    jump_seen = False
    for sim, records, weight in zip(result.optimizers, result.records, result.weights):
        vector = np.zeros(8, dtype=complex)
        vector[0] = 1
        vector = _apply(_apply(_apply(vector, H, (0,)), CX, (0, 1)), T, (0,))
        kraus, flip = records
        assert kraus.event_index == 4
        assert flip.event_index == 9
        if kraus.label == "jump":
            jump_seen = True
            matrix = np.array([[0., gamma**.5], [0., 0.]])
        else:
            assert kraus.label == "no_jump"
            matrix = np.diag([1., (1 - gamma)**.5])
        vector = _apply(vector, matrix, (0,))
        target = float(np.vdot(vector, vector).real)
        assert kraus.probability == pytest.approx(target, rel=1e-12, abs=0.)
        vector /= target**.5
        expected_weight = target / .5
        outcomes = [r.outcome for r in sim.measurements]
        assert len(outcomes) == 5
        assert outcomes[0] == 1
        outcomes = outcomes[1:]
        probabilities = [1.]
        vector, p = _condition(vector, X, 0, outcomes[0])
        probabilities.append(p)
        if outcomes[0] < 0:
            vector = _apply(vector, Z, (1,))
        vector, p = _condition(vector, Z, 0, outcomes[1])
        probabilities.append(p)
        if outcomes[1] < 0:
            vector = _apply(vector, X, (0,))
        vector = _apply(vector, T, (1,))
        if flip.label == "X":
            vector = _apply(vector, X, (2,))
            target = .07
        else:
            assert flip.label == "I"
            target = .93
        assert flip.probability == pytest.approx(target)
        expected_weight *= target / .5
        assert weight == pytest.approx(expected_weight, rel=1e-12, abs=0.)
        assert all(r.proposal_probability == .5 for r in records)
        vector = _apply(vector, CX, (1, 2))
        vector, p = _condition(vector, X, 1, outcomes[2])
        probabilities.append(p)
        if outcomes[2] < 0:
            vector = _apply(vector, X, (2,))
            vector = _apply(vector, Z, (1,))  # deterministic X-basis reset
        vector, p = _condition(vector, Z, 2, outcomes[3])
        probabilities.append(p)
        actual = sim.to_statevector()
        assert np.linalg.norm(actual) == pytest.approx(1., abs=1e-12)
        assert abs(np.vdot(actual, vector))**2 == pytest.approx(1., abs=1e-12)
        visible = [e for e in sim.norm_events if e.kind in {"measure", "measure_absorb"}]
        np.testing.assert_allclose([e.branch_probability for e in visible], probabilities, atol=1e-12)
        assert sim.logical_order == [2, 0, 1]
        routing = sim.measurement_routing_diagnostics()
        assert routing["total"] >= 5
        assert routing["counts"]["stim"] >= 1
    assert jump_seen  # importance sampling must retain the extremely rare branch

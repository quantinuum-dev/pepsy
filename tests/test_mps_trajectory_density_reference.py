"""Dense noisy-ensemble references and separation of sampling/compression error.

These are small-system integration tests, not hardware benchmarks. Statistical
checks use fixed seeds and six standard errors computed from an enumerated
proposal law, never ESS or a tolerance fitted to the observed deviation.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy

pytestmark = pytest.mark.integration

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1., -1.]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)


def _ry(angle):
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return np.array([[c, -s], [s, c]], complex)


def _embed(matrix, sites, n):
    """Independent big-endian basis construction, including nonlocal supports."""
    sites = tuple(sites)
    result = np.zeros((2**n, 2**n), complex)
    for column in range(2**n):
        bits = [(column >> (n - 1 - j)) & 1 for j in range(n)]
        local_column = sum(bits[j] << (len(sites) - 1 - k) for k, j in enumerate(sites))
        for local_row in range(2**len(sites)):
            output = bits.copy()
            for k, j in enumerate(sites):
                output[j] = (local_row >> (len(sites) - 1 - k)) & 1
            row = sum(bit << (n - 1 - j) for j, bit in enumerate(output))
            result[row, column] = matrix[local_row, local_column]
    return result


def _ad(gamma):
    return (np.diag([1., np.sqrt(1 - gamma)]),
            np.array([[0., np.sqrt(gamma)], [0., 0.]]))


@dataclass
class _Branch:
    state: np.ndarray
    target: float = 1.
    proposal: float = 1.


def _reference(operations, n, importance=False):
    """Propagate a density matrix and independently enumerate pure branches."""
    state = np.zeros(2**n, complex)
    state[0] = 1
    rho = np.outer(state, state.conj())
    branches = [_Branch(state)]
    for kind, matrices, sites in operations:
        operators = [_embed(matrix, sites, n) for matrix in matrices]
        rho = sum(operator @ rho @ operator.conj().T for operator in operators)
        children = []
        for branch in branches:
            vectors = [operator @ branch.state for operator in operators]
            target = np.array([np.vdot(v, v).real for v in vectors])
            proposal = target.copy()
            if importance and kind == "channel":
                positive = target > 0
                proposal = .5 * target + .5 * positive / positive.sum()
            for v, p, q in zip(vectors, target, proposal):
                if p > 0:
                    children.append(_Branch(v / np.sqrt(p), branch.target * p,
                                            branch.proposal * q))
        branches = children
    np.testing.assert_allclose(np.trace(rho), 1., atol=1e-12)
    np.testing.assert_allclose(sum(b.target for b in branches), 1., atol=1e-12)
    np.testing.assert_allclose(sum(b.proposal for b in branches), 1., atol=1e-12)
    enumerated = sum(b.target * np.outer(b.state, b.state.conj()) for b in branches)
    np.testing.assert_allclose(enumerated, rho, atol=1e-12)
    return rho, branches


def _expectation(state, observable):
    return float(np.vdot(state, observable @ state).real)


def _standard_error(branches, observable, mean, shots):
    variance = sum(
        b.proposal * (b.target / b.proposal * _expectation(b.state, observable) - mean)**2
        for b in branches
    )
    return np.sqrt(max(0., variance) / shots)


def _state(optimizer):
    state = optimizer.to_dense().reshape(-1)
    np.testing.assert_allclose(np.linalg.norm(state), 1., atol=2e-10)
    return state


def test_dense_reference_has_analytic_amplitude_damping_and_feedforward_limits():
    rho = np.diag([0., 1.])
    np.testing.assert_allclose(sum(k @ rho @ k.conj().T for k in _ad(.3)),
                               np.diag([.3, .7]))
    p0, p1 = np.diag([1., 0.]), np.diag([0., 1.])
    operations = [("unitary", (H,), (0,)), ("unitary", (CX,), (0, 1)),
                  ("measurement", (np.kron(p0, I), np.kron(p1, X)), (0, 1))]
    rho, _ = _reference(operations, 2)
    np.testing.assert_allclose(rho, np.diag([.5, 0., .5, 0.]), atol=1e-12)
    np.testing.assert_array_equal(_embed(CX, (0, 2), 4)[:, 8], np.eye(16)[:, 10])


@pytest.mark.parametrize("mode", ["direct", "dmrg2"])
@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
@pytest.mark.parametrize("importance", [False, True])
def test_noisy_observables_match_density_matrix_with_feedforward(mode, strategy, importance):
    p0, p1 = np.diag([1., 0.]), np.diag([0., 1.])
    phase = (np.sqrt(.87) * I, np.sqrt(.13) * Z)
    operations = [
        ("unitary", (_ry(1.1),), (0,)), ("unitary", (_ry(.7),), (1,)),
        ("unitary", (CX,), (0, 1)), ("unitary", (H,), (0,)),
        ("channel", _ad(.2), (1,)), ("channel", phase, (0,)),
        ("measurement", (np.kron(p0, I), np.kron(p1, X)), (0, 1)),
        ("channel", _ad(.3), (1,)),
    ]
    rho, branches = _reference(operations, 2, importance)
    stream = [(_ry(1.1), 0), (_ry(.7), 1), (CX, (0, 1)), (H, 0),
              pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.amplitude_damping(.2), 1),
              pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.mixture(
                  (("I", .87, I), ("Z", .13, Z))), 0),
              ("measure", "Z", 0), ("if", -1, 1, (X, 1)),
              pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.amplitude_damping(.3), 1)]

    def proposal(event_index, labels, target, optimizer):
        assert optimizer is not None
        positive = target > 0
        return .5 * target + .5 * positive / positive.sum()

    shots = 512 if strategy == "independent" else 32768
    result = pepsy.run_trajectory_shots(
        lambda: pepsy.MpsOptimizer(qtn.MPS_computational_state("00", dtype="complex128"),
                                  chi=4, mode=mode),
        stream, shots=shots, seed=73, strategy=strategy,
        importance_sampling=proposal if importance else None,
        run_kwargs={"progbar": False, "cutoff": 0., "n_iter": 4,
                    "fit_init_strategy": "guess-direct"},
    )
    counts = result.counts if strategy == "coalesced" else [1] * shots
    states = [_state(opt) for opt in result.optimizers]
    observed_rho = sum(count * weight * np.outer(v, v.conj())
                       for count, weight, v in zip(counts, result.weights, states)) / shots
    # The weight sum itself is an observable; do not self-normalize the result.
    # All sixteen Pauli products form a complete density-matrix basis.
    observables = [np.kron(a, b) for a in (I, X, Y, Z) for b in (I, X, Y, Z)]
    for observable in observables:
        exact = float(np.trace(rho @ observable).real)
        values = [_expectation(v, observable) for v in states]
        estimate = result.estimate(values)
        np.testing.assert_allclose(estimate, np.trace(observed_rho @ observable).real,
                                   atol=1e-12)
        tolerance = 6 * _standard_error(branches, observable, exact, shots) + 2e-10
        assert abs(estimate - exact) <= tolerance, (mode, strategy, importance, estimate, exact)
    if strategy == "coalesced":
        assert sum(result.counts) == shots


def test_compression_bias_is_separate_from_sampling_error():
    operations = [
        ("unitary", (_ry(1.1),), (0,)), ("unitary", (CX,), (0, 2)),
        ("unitary", (_ry(.7),), (1,)), ("unitary", (CX,), (1, 3)),
        ("channel", _ad(.2), (3,)),
        ("channel", (np.sqrt(.85) * I, np.sqrt(.15) * Z), (0,)),
    ]
    exact_rho, _ = _reference(operations, 4)
    stream = [(_ry(1.1), 0), (CX, (0, 2)), (_ry(.7), 1), (CX, (1, 3)),
              pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.amplitude_damping(.2), 3),
              pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.mixture(
                  (("I", .85, I), ("Z", .15, Z))), 0)]
    observable = _embed(np.kron(X, X), (0, 2), 4)
    errors = {}
    low_rank_bias = []
    sampling_errors = []
    # The central cut has Schmidt rank four. An absolute cutoff .4 deliberately
    # removes a nonzero Schmidt value even when chi can hold the exact state.
    for chi, cutoff, shots in [(1, 0., 65536), (2, 0., 8192),
                               (2, 0., 65536), (4, 0., 65536), (4, .4, 65536)]:
        result = pepsy.run_trajectory_shots(
            lambda: pepsy.MpsOptimizer(qtn.MPS_computational_state("0000", dtype="complex128"),
                                      chi=chi, mode="svd"),
            stream, shots=shots, seed=91, strategy="coalesced",
            run_kwargs={"progbar": False, "cutoff": cutoff, "cutoff_mode": "abs"},
        )
        if chi == 4 and cutoff == 0.:
            assert result.branches == 4
        states = [_state(opt) for opt in result.optimizers]
        probabilities = np.array([np.prod([r.probability for r in leaf.records])
                                  for leaf in result.leaves])
        np.testing.assert_allclose(probabilities.sum(), 1., atol=1e-12)
        # Ignore random counts to recover this compressed model's infinite-shot
        # ensemble. Its difference from the dense oracle is compression bias.
        model_rho = sum(p * np.outer(v, v.conj()) for p, v in zip(probabilities, states))
        error = np.linalg.norm(model_rho - exact_rho)
        errors[(chi, cutoff)] = error
        values = np.array([_expectation(v, observable) for v in states])
        mean = float(probabilities @ values)
        variance = float(probabilities @ (values - mean)**2)
        se = np.sqrt(variance / shots)
        assert abs(result.estimate(values) - mean) < 6 * se + 2e-10
        if chi == 2:
            low_rank_bias.append(error)
            assert variance > .01
            sampling_errors.append(se)
    np.testing.assert_allclose(low_rank_bias[0], low_rank_bias[1], atol=1e-12)
    assert sampling_errors[0] / sampling_errors[1] == pytest.approx(np.sqrt(8))
    assert errors[(1, 0.)] > errors[(2, 0.)] > .05
    assert errors[(4, 0.)] < 1e-10
    assert errors[(4, .4)] > .05

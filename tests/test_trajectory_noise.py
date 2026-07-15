"""Regression tests for user-defined MPS quantum-trajectory channels."""

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy


_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_I = np.eye(2, dtype=complex)


def _statevector(optimizer):
    if isinstance(optimizer, pepsy.MpsStabOptimizer):
        return optimizer.to_statevector().reshape(-1)
    return optimizer.to_dense().reshape(-1)


def _factory(kind):
    if kind == "mps":
        return lambda: pepsy.MpsOptimizer(
            qtn.MPS_computational_state("0"), chi=4, mode="mpo"
        )
    return lambda: pepsy.MpsStabOptimizer(1, chi=4)


def _run_kwargs(kind):
    return {"progbar": False} if kind == "mps" else {}


def test_random_unitary_channel_samples_directly_in_a_gate_stream():
    channel = pepsy.TrajectoryChannel.mixture(
        (("identity", 0.0, _I), ("bit_flip", 1.0, _X))
    )
    stream = [("h", 0), pepsy.TrajectoryEvent(channel, 0)]
    sample = pepsy.sample_trajectory_stream(stream, seed=5)

    assert sample.records == (
        pepsy.TrajectoryRecord(1, (0,), "bit_flip", 1.0),
    )
    assert sample.gate_stream[0] == ("h", 0)
    np.testing.assert_allclose(sample.gate_stream[1][0], _X)
    assert sample.gate_stream[1][1] == 0

    with pytest.raises(ValueError, match="State-dependent Kraus"):
        pepsy.sample_trajectory_stream(
            [pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.amplitude_damping(0.2), 0)]
        )


def test_trajectory_channel_validates_probabilities_dimensions_and_kraus_sum():
    with pytest.raises(ValueError, match="sum to one"):
        pepsy.TrajectoryChannel.mixture((("I", 0.8, _I), ("X", 0.3, _X)))
    with pytest.raises(ValueError, match="unitary"):
        pepsy.TrajectoryChannel.mixture((("not_unitary", 1.0, 0.5 * _I),))
    with pytest.raises(ValueError, match=r"sum\(K\^dagger K\) = I"):
        pepsy.TrajectoryChannel.kraus((("bad", 0.5 * _I),))
    channel = pepsy.TrajectoryChannel.depolarizing(0.1)
    with pytest.raises(ValueError, match="support size"):
        pepsy.TrajectoryEvent(channel, (0, 1))


@pytest.mark.parametrize("kind", ("mps", "stn"))
def test_amplitude_damping_trajectory_replays_and_normalizes_on_both_optimizers(kind):
    stream = [
        (_X, 0),
        pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.amplitude_damping(1.0), 0),
        (_X, 0),
    ]
    result = pepsy.run_trajectory_shots(
        _factory(kind), stream, shots=3, seed=4, run_kwargs=_run_kwargs(kind)
    )

    assert result.shots == 3
    assert all(
        records == (pepsy.TrajectoryRecord(1, (0,), "jump", 1.0),)
        for records in result.records
    )
    assert all(
        len(gate_stream) == 3 and not isinstance(gate_stream[1], pepsy.TrajectoryEvent)
        for gate_stream in result.gate_streams
    )
    for optimizer in result.optimizers:
        np.testing.assert_allclose(_statevector(optimizer), [0.0, 1.0], atol=1e-8)
        assert abs(optimizer.p.norm()) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("kind", ("mps", "stn"))
def test_state_dependent_kraus_branches_are_sampled_from_the_current_state(kind):
    stream = [
        (_X, 0),
        pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.amplitude_damping(0.5), 0),
    ]
    result = pepsy.run_trajectory_shots(
        _factory(kind), stream, shots=12, seed=6, run_kwargs=_run_kwargs(kind)
    )

    labels = {records[0].label for records in result.records}
    assert labels == {"jump", "no_jump"}
    assert all(records[0].probability == pytest.approx(0.5) for records in result.records)
    for optimizer, records in zip(result.optimizers, result.records):
        expected = [1.0, 0.0] if records[0].label == "jump" else [0.0, 1.0]
        np.testing.assert_allclose(_statevector(optimizer), expected, atol=1e-8)
        assert abs(optimizer.p.norm()) == pytest.approx(1.0, abs=1e-8)

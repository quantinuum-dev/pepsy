"""Regression tests for user-defined MPS quantum-trajectory channels."""

import importlib.util
from pathlib import Path

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


def test_kraus_trajectory_starts_a_fresh_stn_norm_diagnostic_segment():
    stream = [
        ("rxx", 0.8, 0, 1),
        (_X, 0),
        pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.amplitude_damping(1.0), 0),
        ("rxx", 0.8, 0, 1),
    ]
    result = pepsy.run_trajectory_shots(
        lambda: pepsy.MpsStabOptimizer(2, chi=1, track_infidelity=True),
        stream,
        shots=1,
        seed=7,
    )

    simulator = result.optimizers[0]
    event = simulator.norm_events[-1]
    diagnostics = simulator.norm_diagnostics()
    assert event["kind"] == "trajectory_kraus"
    assert event["valid"] is True
    assert event["branch_probability"] == pytest.approx(result.records[0][0].probability)
    assert event["post_norm"] == pytest.approx(1.0, abs=1e-10)
    assert diagnostics["current_valid"] is True
    assert diagnostics["total_survival_proxy"] is not None
    assert diagnostics["total_norm_proxy"] == pytest.approx(
        diagnostics["total_survival_proxy"] ** 0.5
    )


def test_coalesced_pauli_ensemble_reuses_the_ideal_no_error_state():
    """A zero-rate ensemble should build one optimizer for all represented shots."""
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return pepsy.MpsOptimizer(
            qtn.MPS_computational_state("0"), chi=4, mode="mpo"
        )

    result = pepsy.run_coalesced_noisy_shots(
        factory,
        [(_X, 0)],
        pepsy.PauliErrorModel(),
        shots=64,
        seed=8,
        run_kwargs={"progbar": False},
    )

    assert calls == 1
    assert result.shots == 64
    assert result.branches == 1
    assert result.counts == (64,)
    assert result.leaves[0].faults == ()
    np.testing.assert_allclose(_statevector(result.leaves[0].optimizer), [0.0, 1.0])


def test_coalesced_trajectory_branches_mid_circuit_measurements_by_count():
    """A shared ancilla-like measurement should use exact binomial branch counts."""
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return pepsy.MpsOptimizer(
            qtn.MPS_computational_state("0"), chi=4, mode="mpo"
        )

    result = pepsy.run_coalesced_trajectory_shots(
        factory,
        [
            (
                np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0),
                0,
            ),
            ("measure", "Z", 0),
        ],
        shots=64,
        seed=9,
        run_kwargs={"progbar": False},
    )

    assert calls == 1
    assert result.shots == 64
    assert result.branches == 2
    assert {leaf.measurements[0].outcome for leaf in result.leaves} == {-1, 1}
    assert all(leaf.measurements[0].probability == pytest.approx(0.5) for leaf in result.leaves)
    assert all(leaf.gate_stream[-1][0] == "measure" for leaf in result.leaves)


def test_coalesced_kraus_ensemble_uses_one_copy_per_nonempty_outcome():
    """State-dependent channel probabilities are evaluated once per live node."""
    result = pepsy.run_coalesced_trajectory_shots(
        lambda: pepsy.MpsStabOptimizer(1, chi=4),
        [
            (_X, 0),
            pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.amplitude_damping(0.5), 0),
        ],
        shots=64,
        seed=10,
    )

    assert result.shots == 64
    assert result.branches == 2
    leaves = {leaf.records[0].label: leaf for leaf in result.leaves}
    assert set(leaves) == {"jump", "no_jump"}
    assert all(leaf.records[0].probability == pytest.approx(0.5) for leaf in leaves.values())
    np.testing.assert_allclose(_statevector(leaves["jump"].optimizer), [1.0, 0.0])
    np.testing.assert_allclose(_statevector(leaves["no_jump"].optimizer), [0.0, 1.0])


def test_coalesced_terminal_sampling_reads_each_mps_leaf_in_one_batch():
    """Terminal samples expand rows, never the represented optimizer state."""
    result = pepsy.run_coalesced_noisy_shots(
        lambda: pepsy.MpsOptimizer(
            qtn.MPS_computational_state("0"), chi=4, mode="mpo"
        ),
        [(_X, 0)],
        pepsy.PauliErrorModel(),
        shots=12,
        seed=4,
        run_kwargs={"progbar": False},
    )

    samples = result.sample_bits(seed=12, shuffle=False)

    assert samples.shots == 12
    assert samples.branches == 1
    np.testing.assert_array_equal(samples.configs, np.ones((12, 1), dtype=np.int8))
    np.testing.assert_array_equal(samples.leaf_indices, np.zeros(12, dtype=np.int64))
    np.testing.assert_allclose(samples.probs, np.ones(12))


def test_coalesced_terminal_sampling_uses_stn_tree_sampler_without_probs():
    """STN leaves retain their scalable bit sampler and avoid dense probabilities."""
    result = pepsy.run_coalesced_noisy_shots(
        lambda: pepsy.MpsStabOptimizer(2, chi=4),
        [],
        pepsy.PauliErrorModel(),
        shots=10,
        seed=4,
    )

    samples = pepsy.sample_coalesced_bits(result, seed=13, shuffle=False)

    assert samples.probs is None
    np.testing.assert_array_equal(samples.configs, np.zeros((10, 2), dtype=np.int8))


def test_coalesced_trajectory_benchmark_reports_branch_reduction():
    """The benchmark remains a lightweight, JSON-ready downstream harness."""
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "coalesced_trajectory_scaling.py"
    spec = importlib.util.spec_from_file_location("coalesced_trajectory_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    row = module.run_case(
        n=2,
        depth=1,
        shots=4,
        probability=0.0,
        chi=2,
        seed=4,
    )

    assert row["represented_shots"] == 4
    assert row["independent_states"] == 4
    assert row["coalesced_states"] == 1
    assert row["state_reduction"] == pytest.approx(4.0)
    assert row["expected_faults"] == 0.0

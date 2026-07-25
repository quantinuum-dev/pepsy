"""Regression tests for user-defined MPS quantum-trajectory channels."""

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy


_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_I = np.eye(2, dtype=complex)


def _statevector(optimizer):
    if isinstance(optimizer, (pepsy.MpsStabOptimizer, pepsy.TreeStabOptimizer)):
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


def test_native_stochastic_pauli_entries_are_gate_stream_events():
    stream = [
        ("h", 0),
        ("x_error", 1.0, 0),
        ("pauli_channel1", {"z": 1.0}, 0),
    ]

    sample = pepsy.sample_trajectory_stream(stream, seed=5)

    assert sample.records == (
        pepsy.TrajectoryRecord(1, (0,), "X", 1.0),
        pepsy.TrajectoryRecord(2, (0,), "Z", 1.0),
    )
    assert sample.gate_stream[0] == ("h", 0)
    np.testing.assert_allclose(sample.gate_stream[1][0], _X)
    np.testing.assert_allclose(sample.gate_stream[2][0], np.diag([1.0, -1.0]))


def test_native_stochastic_entries_use_trajectory_runner_not_external_macro():
    stream = [("x_error", 0.1, 0)]

    with pytest.raises(ValueError, match="Stream-local stochastic entries"):
        pepsy.sample_noisy_gate_stream(stream, pepsy.PauliErrorModel())

    with pytest.raises(ValueError, match="Stream-local stochastic entries"):
        pepsy.run_noisy_shots(
            lambda: pepsy.MpsStabOptimizer(1),
            stream,
            pepsy.PauliErrorModel(),
            shots=1,
        )


def test_leakage_entries_use_trajectory_runner_not_external_macro():
    stream = [("leakage", 0.1, 0)]

    with pytest.raises(ValueError, match="Stateful leakage entries"):
        pepsy.sample_noisy_gate_stream(stream, pepsy.PauliErrorModel())

    with pytest.raises(ValueError, match="Stateful leakage entries"):
        pepsy.sample_trajectory_stream(stream)


@pytest.mark.parametrize("kind", ("mps", "stn"))
def test_leakage_suppresses_gates_and_measure_leaked_reports_two(kind):
    stream = [(_X, 0), ("leakage", 1.0, 0), (_X, 0), ("measure_leaked", 0)]

    result = pepsy.run_trajectory_shots(
        _factory(kind), stream, shots=1, seed=5, run_kwargs=_run_kwargs(kind)
    )

    records = result.leakage_records[0]
    assert records[0] == pepsy.LeakageRecord(
        event_index=1,
        kind="leakage",
        site=0,
        probability=1.0,
        occurred=True,
        initially_leaked=False,
        finally_leaked=True,
        branch="leaked",
    )
    assert records[1].kind == "measure_leaked"
    assert records[1].measurement == 2
    assert records[1].finally_leaked is True
    np.testing.assert_allclose(_statevector(result.optimizers[0]), [1.0, 0.0], atol=1e-8)


@pytest.mark.parametrize("kind", ("mps", "stn"))
def test_reset_clears_leakage_before_later_gates(kind):
    stream = [
        ("leakage", 1.0, 0),
        ("reset", 0),
        (_X, 0),
        ("measure_leaked", 0),
    ]

    result = pepsy.run_trajectory_shots(
        _factory(kind), stream, shots=1, seed=5, run_kwargs=_run_kwargs(kind)
    )

    assert result.leakage_records[0][0].finally_leaked is True
    assert result.leakage_records[0][1].measurement == 1
    assert result.leakage_records[0][1].finally_leaked is False
    np.testing.assert_allclose(_statevector(result.optimizers[0]), [0.0, 1.0], atol=1e-8)


@pytest.mark.parametrize("kind", ("mps", "stn"))
def test_leakage_return_unleaks_to_a_computational_branch(kind):
    stream = [("leakage", 1.0, 0), ("leakage_return", 1.0, 0), ("measure_leaked", 0)]

    result = pepsy.run_trajectory_shots(
        _factory(kind), stream, shots=1, seed=9, run_kwargs=_run_kwargs(kind)
    )

    records = result.leakage_records[0]
    assert records[1].kind == "leakage_return"
    assert records[1].occurred is True
    assert records[1].initially_leaked is True
    assert records[1].finally_leaked is False
    assert records[1].branch in {"return_0", "return_1"}
    assert records[2].kind == "measure_leaked"
    assert records[2].measurement in {0, 1}


def test_leak2depolar_replaces_later_leakage_with_pauli_approximation():
    result = pepsy.run_trajectory_shots(
        lambda: pepsy.MpsStabOptimizer(1, chi=4),
        [("leak2depolar", True), ("leakage", 1.0, 0), ("measure_leaked", 0)],
        shots=1,
        seed=4,
    )

    records = result.leakage_records[0]
    assert records[0].kind == "leak2depolar"
    assert records[1].kind == "leakage_depolarize"
    assert records[1].occurred is True
    assert records[1].finally_leaked is False
    assert records[2].measurement in {0, 1}


def test_stateful_leakage_auto_strategy_stays_independent_for_now():
    result = pepsy.run_trajectory_shots(
        lambda: pepsy.MpsStabOptimizer(1, chi=4),
        [("leakage", 1.0, 0)],
        shots=3,
        seed=4,
        strategy="auto",
    )

    assert isinstance(result, pepsy.TrajectoryShotResult)
    with pytest.raises(NotImplementedError, match="stateful leakage"):
        pepsy.run_trajectory_shots(
            lambda: pepsy.MpsStabOptimizer(1, chi=4),
            [("leakage", 1.0, 0)],
            shots=3,
            seed=4,
            strategy="coalesced",
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
def test_native_amplitude_damping_entry_replays_as_kraus_channel(kind):
    stream = [(_X, 0), ("amplitude_damping", 1.0, 0), (_X, 0)]

    result = pepsy.run_trajectory_shots(
        _factory(kind), stream, shots=2, seed=4, run_kwargs=_run_kwargs(kind)
    )

    assert all(
        records == (pepsy.TrajectoryRecord(1, (0,), "jump", 1.0),)
        for records in result.records
    )
    for optimizer in result.optimizers:
        np.testing.assert_allclose(_statevector(optimizer), [0.0, 1.0], atol=1e-8)


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


def test_coalesced_native_pauli_channel2_uses_stream_local_noise():
    result = pepsy.run_coalesced_trajectory_shots(
        lambda: pepsy.MpsStabOptimizer(2, chi=4),
        [("pauli_channel2", {"XI": 1.0}, 0, 1)],
        shots=16,
        seed=11,
    )

    assert result.shots == 16
    assert result.branches == 1
    assert result.leaves[0].records == (
        pepsy.TrajectoryRecord(0, (0, 1), "XI", 1.0),
    )
    expected = np.zeros(4, dtype=complex)
    expected[2] = 1.0
    np.testing.assert_allclose(_statevector(result.leaves[0].optimizer), expected)


def test_trajectory_runner_strategy_setting_can_select_coalescing():
    result = pepsy.run_trajectory_shots(
        lambda: pepsy.MpsStabOptimizer(1, chi=4),
        [("x_error", 1.0, 0)],
        shots=12,
        seed=12,
        strategy="coalesced",
        max_branches=4,
    )

    assert isinstance(result, pepsy.CoalescedTrajectoryResult)
    assert result.shots == 12
    assert result.branches == 1
    assert result.leaves[0].records == (
        pepsy.TrajectoryRecord(0, (0,), "X", 1.0),
    )
    np.testing.assert_allclose(_statevector(result.leaves[0].optimizer), [0.0, 1.0])


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


def test_tree_state_dependent_kraus_branches_are_sampled_from_the_current_state():
    """Tree trajectories compute Kraus probabilities from copied TTNs."""
    stream = [
        (_X, 0),
        pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.amplitude_damping(0.5), 0),
    ]
    result = pepsy.run_trajectory_shots(
        lambda: pepsy.TreeOptimizer(None, n=2, chi=4, run=False),
        stream,
        shots=12,
        seed=6,
    )

    labels = {records[0].label for records in result.records}
    assert labels == {"jump", "no_jump"}
    assert all(records[0].probability == pytest.approx(0.5) for records in result.records)
    for optimizer, records in zip(result.optimizers, result.records):
        expected = [1.0, 0.0, 0.0, 0.0] if records[0].label == "jump" else [0.0, 0.0, 1.0, 0.0]
        np.testing.assert_allclose(_statevector(optimizer), expected, atol=1e-8)
        assert optimizer.norm() == pytest.approx(1.0, abs=1e-8)


def test_tree_stab_state_dependent_kraus_branches_use_tree_normalization():
    stream = [
        (_X, 0),
        pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.amplitude_damping(0.5), 0),
    ]
    result = pepsy.run_trajectory_shots(
        lambda: pepsy.TreeStabOptimizer(1), stream, shots=12, seed=6
    )

    labels = {records[0].label for records in result.records}
    assert labels == {"jump", "no_jump"}
    assert all(records[0].probability == pytest.approx(0.5) for records in result.records)
    for optimizer, records in zip(result.optimizers, result.records):
        expected = [1.0, 0.0] if records[0].label == "jump" else [0.0, 1.0]
        np.testing.assert_allclose(_statevector(optimizer), expected, atol=1e-8)
        assert optimizer.norm() == pytest.approx(1.0, abs=1e-8)


def test_tree_stab_random_unitary_depolarizing_channel_replays_branches():
    result = pepsy.run_trajectory_shots(
        lambda: pepsy.TreeStabOptimizer(1),
        [pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.depolarizing(1.0), 0)],
        shots=16,
        seed=4,
    )

    assert {records[0].label for records in result.records} == {"X", "Y", "Z"}
    assert all(records[0].probability == pytest.approx(1.0 / 3.0) for records in result.records)
    assert all(optimizer.norm() == pytest.approx(1.0) for optimizer in result.optimizers)


def test_coalesced_tree_stab_measurement_and_terminal_sampling():
    hadamard = np.array(
        [[1.0, 1.0], [1.0, -1.0]], dtype=complex
    ) / np.sqrt(2.0)
    result = pepsy.run_coalesced_trajectory_shots(
        lambda: pepsy.TreeStabOptimizer(1),
        [(hadamard, 0), ("measure", "Z", 0)],
        shots=32,
        seed=9,
    )

    assert result.shots == 32
    assert result.branches == 2
    assert {leaf.measurements[0].outcome for leaf in result.leaves} == {-1, 1}
    samples = result.sample_bits(seed=12, shuffle=False)
    assert samples.shots == 32
    assert samples.configs.shape == (32, 1)
    assert set(samples.configs[:, 0]) <= {0, 1}


def test_coalesced_feed_forward_replays_per_measurement_leaf():
    hadamard = np.array(
        [[1.0, 1.0], [1.0, -1.0]], dtype=complex
    ) / np.sqrt(2.0)
    result = pepsy.run_coalesced_trajectory_shots(
        lambda: pepsy.TreeStabOptimizer(2),
        [
            (hadamard, 0),
            ("measure", "Z", 0),
            ("if", -1, 1, ("x", 1)),
        ],
        shots=32,
        seed=13,
    )

    assert result.branches == 2
    for leaf in result.leaves:
        bit = int(leaf.optimizer.measurements[0].outcome < 0)
        samples = leaf.optimizer.sample_bits(8, seed=4, shuffle=False)
        assert np.all(samples[:, 0] == bit)
        assert np.all(samples[:, 1] == bit)


def test_tree_stab_terminal_sampling_does_not_require_dense_readout():
    optimizer = pepsy.TreeStabOptimizer(2, max_dense_sample_qubits=1)
    samples = optimizer.sample_bits(8, seed=3)
    assert samples.shape == (8, 2)
    assert set(samples.ravel()) <= {0, 1}


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


def test_coalesced_tree_trajectory_branches_mid_circuit_measurements_by_count():
    """Tree expectations support exact coalesced measurement branching."""
    hadamard = np.array(
        [[1.0, 1.0], [1.0, -1.0]], dtype=complex
    ) / np.sqrt(2.0)
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return pepsy.TreeOptimizer(None, n=2, chi=4, run=False)

    result = pepsy.run_coalesced_trajectory_shots(
        factory,
        [(hadamard, 0), ("measure", "Z", 0)],
        shots=64,
        seed=9,
    )

    assert calls == 1
    assert result.shots == 64
    assert result.branches == 2
    assert {leaf.measurements[0].outcome for leaf in result.leaves} == {-1, 1}
    assert all(
        leaf.measurements[0].probability == pytest.approx(0.5)
        for leaf in result.leaves
    )


def test_importance_sampling_records_likelihood_ratios_and_estimates_rare_branch():
    channel = pepsy.TrajectoryChannel.mixture(
        (("I", 0.99, _I), ("X", 0.01, _X))
    )
    policy = pepsy.ImportanceSamplingPolicy({0: {"I": 0.5, "X": 0.5}})
    result = pepsy.run_trajectory_shots(
        lambda: pepsy.MpsStabOptimizer(1, chi=4),
        [pepsy.TrajectoryEvent(channel, 0)],
        shots=2_000,
        seed=19,
        importance_sampling=policy,
    )

    assert {record[0].proposal_probability for record in result.records} == {0.5}
    assert {record[0].likelihood_ratio for record in result.records} == {0.02, 1.98}
    estimate = result.estimate(
        [int(records[0].label == "X") for records in result.records]
    )
    assert estimate == pytest.approx(0.01, abs=0.003)
    assert result.effective_sample_size < result.shots


def test_coalesced_importance_sampling_weights_leaves_and_honors_branch_budget():
    channel = pepsy.TrajectoryChannel.mixture(
        (("I", 0.99, _I), ("X", 0.01, _X))
    )
    policy = pepsy.ImportanceSamplingPolicy({0: {"I": 0.5, "X": 0.5}})
    result = pepsy.run_coalesced_trajectory_shots(
        lambda: pepsy.MpsStabOptimizer(1, chi=4),
        [pepsy.TrajectoryEvent(channel, 0)],
        shots=2_000,
        seed=19,
        importance_sampling=policy,
        parallel_workers=2,
    )

    assert result.shots == 2_000
    assert result.weights == pytest.approx((1.98, 0.02))
    estimate = result.estimate(
        [int(leaf.records[0].label == "X") for leaf in result.leaves]
    )
    assert estimate == pytest.approx(0.01, abs=0.003)
    with pytest.raises(RuntimeError, match="per-event branch budget"):
        pepsy.run_coalesced_trajectory_shots(
            lambda: pepsy.MpsStabOptimizer(1, chi=4),
            [pepsy.TrajectoryEvent(channel, 0)],
            shots=10,
            seed=19,
            max_branch_factor=1,
        )


def test_parallel_independent_trajectory_seed_streams_are_worker_count_invariant():
    channel = pepsy.TrajectoryChannel.mixture(
        (("I", 0.8, _I), ("X", 0.2, _X))
    )
    stream = [pepsy.TrajectoryEvent(channel, 0)]
    serial = pepsy.run_trajectory_shots(
        lambda: pepsy.MpsStabOptimizer(1, chi=4), stream, shots=32, seed=23
    )
    parallel = pepsy.run_trajectory_shots(
        lambda: pepsy.MpsStabOptimizer(1, chi=4),
        stream,
        shots=32,
        seed=23,
        parallel_workers=4,
    )
    assert parallel.weights == serial.weights
    assert [records[0].label for records in parallel.records] == [
        records[0].label for records in serial.records
    ]


def test_stim_importance_sampling_and_parallel_seed_streams():
    stim = pytest.importorskip("stim")
    circuit = stim.Circuit("X_ERROR(0.01) 0\nM 0")
    policy = pepsy.ImportanceSamplingPolicy({0: {"I": 0.5, "X": 0.5}})

    serial = pepsy.run_stim_shots(
        lambda: pepsy.MpsStabOptimizer(1),
        circuit,
        shots=1_000,
        seed=31,
        importance_sampling=policy,
    )
    parallel = pepsy.run_stim_shots(
        lambda: pepsy.MpsStabOptimizer(1),
        circuit,
        shots=1_000,
        seed=31,
        importance_sampling=policy,
        parallel_workers=4,
    )

    assert serial.weights == parallel.weights
    assert [sample.faults for sample in serial.samples] == [
        sample.faults for sample in parallel.samples
    ]
    measured = [int(records[0].outcome < 0) for records in serial.measurements]
    assert serial.estimate(measured) == pytest.approx(0.01, abs=0.003)


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


@pytest.mark.parametrize("kind", ("mps_stn", "tree_stn"))
def test_seeded_stn_trajectories_reproduce_structured_measurements(kind):
    """The optimizer and channel RNG streams are both reproducible."""
    optimizer_cls = (
        pepsy.MpsStabOptimizer if kind == "mps_stn" else pepsy.TreeStabOptimizer
    )
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    stream = [(hadamard, 0), ("measure", "Z", 0)]

    first = pepsy.run_trajectory_shots(
        lambda: optimizer_cls(1), stream, shots=24, seed=31
    )
    second = pepsy.run_trajectory_shots(
        lambda: optimizer_cls(1), stream, shots=24, seed=31
    )

    assert first.measurements == second.measurements
    assert all(
        record.event_index == 1
        and record.pauli == "Z"
        and record.where == (0,)
        and record.outcome in {-1, 1}
        for shot in first.measurements
        for record in shot
    )


@pytest.mark.parametrize("optimizer_cls", (pepsy.MpsStabOptimizer, pepsy.TreeStabOptimizer))
def test_tree_and_mps_stn_kraus_weights_do_not_need_optimizer_copies(optimizer_cls):
    """Kraus probabilities use the exact local Gram path on both STNs."""
    channel = pepsy.TrajectoryChannel.amplitude_damping(0.5)

    def factory():
        optimizer = optimizer_cls(1)

        def forbidden_copy():
            raise AssertionError("Kraus probability evaluation copied the optimizer")

        optimizer.copy = forbidden_copy
        return optimizer

    result = pepsy.run_trajectory_shots(
        factory,
        [(_X, 0), pepsy.TrajectoryEvent(channel, 0)],
        shots=16,
        seed=17,
    )
    assert {records[0].label for records in result.records} == {"jump", "no_jump"}
    assert all(records[0].probability == pytest.approx(0.5) for records in result.records)


@pytest.mark.parametrize("optimizer_cls", (pepsy.MpsStabOptimizer, pepsy.TreeStabOptimizer))
def test_coalesced_reset_is_trace_preserving_and_does_not_duplicate_leaves(optimizer_cls):
    result = pepsy.run_coalesced_trajectory_shots(
        lambda: optimizer_cls(1),
        [("h", 0), ("reset", 0)],
        shots=100,
        seed=19,
    )

    assert result.branches == 1
    assert result.counts == (100,)
    assert result.leaves[0].measurements == ()
    np.testing.assert_allclose(_statevector(result.leaves[0].optimizer), [1.0, 0.0])


@pytest.mark.parametrize("optimizer_cls", (pepsy.MpsStabOptimizer, pepsy.TreeStabOptimizer))
def test_immediate_and_deferred_magic_trajectories_match_direct_state(optimizer_cls):
    """MAST-style projection and recycled immediate injection preserve the state."""
    stream = [("h", 0), ("t", 0)]
    direct = pepsy.run_trajectory_shots(
        lambda: optimizer_cls(1), stream, shots=1, seed=23
    ).optimizers[0]
    direct_state = _statevector(direct)

    for strategy in ("immediate", "deferred"):
        result = pepsy.run_trajectory_shots(
            lambda: optimizer_cls(2),
            stream,
            shots=1,
            seed=23,
            magic_strategy=strategy,
            magic_ancillas=(1,),
        )
        state = _statevector(result.optimizers[0])
        data_state = state[::2]
        assert abs(np.vdot(direct_state, data_state)) == pytest.approx(1.0, abs=1e-7)
        assert result.optimizers[0].measurements


def test_stim_detector_and_observable_records_are_resolved_for_both_replay_modes():
    """Stim rec annotations become structured syndrome records, including leaves."""
    pytest.importorskip("stim")
    circuit = """
    H 0
    M 0
    DETECTOR(1, 2) rec[-1]
    OBSERVABLE_INCLUDE(0) rec[-1]
    """
    for optimizer_cls in (pepsy.MpsStabOptimizer, pepsy.TreeStabOptimizer):
        result = pepsy.run_stim_shots(
            lambda optimizer_cls=optimizer_cls: optimizer_cls(1),
            circuit,
            shots=8,
            seed=29,
        )
        assert len(result.syndromes) == 8
        assert len(result.observables) == 8
        assert {record.value for shot in result.syndromes for record in shot} == {False, True}
        assert result.plan.detectors[0].coordinates == (1.0, 2.0)

        coalesced = pepsy.run_coalesced_stim_shots(
            lambda optimizer_cls=optimizer_cls: optimizer_cls(1),
            circuit,
            shots=8,
            seed=29,
        )
        assert coalesced.plan is result.plan or coalesced.plan.detectors == result.plan.detectors
        assert sum(coalesced.counts) == 8
        assert len(coalesced.syndromes) == coalesced.branches


def test_coherent_crosstalk_helper_is_seeded_and_uses_pepsy_rotation_convention():
    model = pepsy.CoherentCrosstalkModel(0.125, sign_mode="random_sign")
    stream = [("cnot", 0, 1), ("cnot", 1, 2)]
    first = model.transform(stream, seed=37)
    second = model.transform(stream, seed=37)

    assert first == second
    assert first[1][0] == "rzz"
    assert first[1][1] in {-0.25, 0.25}
    assert first[3][0] == "rzz"
    assert first[3][1] in {-0.25, 0.25}


@pytest.mark.parametrize("optimizer_cls", (pepsy.MpsStabOptimizer, pepsy.TreeStabOptimizer))
def test_stn_truncation_convergence_reports_reference_and_observable(optimizer_cls):
    rows = optimizer_cls.truncation_convergence(
        2,
        [("h", 0), ("cnot", 0, 1), ("t", 0)],
        chi_values=(1, None),
        observable=lambda optimizer: optimizer.expectation("Z", 0),
    )

    assert [row["chi"] for row in rows] == [1, None]
    assert all(row["max_bond"] >= 1 for row in rows)
    assert all(np.isfinite(row["norm"]) for row in rows)
    assert all("norm_diagnostics" in row and "observable" in row for row in rows)

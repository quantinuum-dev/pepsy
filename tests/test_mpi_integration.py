"""Real multi-process MPI coverage.

Run explicitly with, for example::

    mpiexec -n 2 python -m pytest -q -o addopts='' tests/test_mpi_integration.py
"""

import numpy as np
import pytest

import pepsy


pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402


class _MPIProbeOptimizer:
    def __init__(self):
        self._rng = None
        self.value = None

    def set_gates(self, gates):
        self.gates = tuple(gates)

    def run(self, **kwargs):
        del kwargs
        self.value = float(self._rng.random())


def _probe_factory():
    return _MPIProbeOptimizer()


@pytest.mark.integration
def test_real_mpi_partition_reduction_and_streaming():
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("run this test under mpiexec -n 2 or more")

    runner = pepsy.MPIShotRunner(
        _probe_factory,
        [(np.eye(2), 0)],
        comm=comm,
    )
    result = runner.run(9, seed=31, retain="final")

    assert result.world_size == comm.Get_size()
    assert result.local_shots == len(result.local_result.optimizers)
    assert result.reduce_sum(result.local_shots) == 9
    estimate = result.reduce_mean(lambda optimizer: optimizer.value)
    assert 0.0 <= estimate <= 1.0
    vector_estimate = result.reduce_mean(
        lambda optimizer: np.asarray([optimizer.value, 2.0 * optimizer.value])
    )
    assert vector_estimate.shape == (2,)
    assert vector_estimate[1] == pytest.approx(2.0 * vector_estimate[0])
    matrix_estimate = result.reduce_mean(
        lambda optimizer: np.asarray(
            [
                [optimizer.value + 1j, 2.0 * optimizer.value],
                [3.0 * optimizer.value, 4.0 * optimizer.value - 1j],
            ]
        )
    )
    assert matrix_estimate.shape == (2, 2)
    assert matrix_estimate[0, 1] == pytest.approx(2.0 * matrix_estimate[1, 0] / 3.0)
    totals = result.reduce_sum(
        np.asarray([result.local_shots, 2 * result.local_shots], dtype=np.int64)
    )
    np.testing.assert_array_equal(totals, np.asarray([9, 18], dtype=np.int64))

    streamed = runner.run(
        9,
        seed=31,
        observable=lambda optimizer: optimizer.value,
        chunk_size=2,
    )
    assert streamed.local_result is None
    assert streamed.reduce_mean() == pytest.approx(estimate)

    sparse = runner.run(1, seed=31, retain="final")
    sparse_mean = sparse.reduce_mean(
        lambda optimizer: np.asarray([optimizer.value, 2.0 * optimizer.value])
    )
    assert sparse_mean.shape == (2,)
    assert sparse_mean[1] == pytest.approx(2.0 * sparse_mean[0])

    empty = runner.run(0, seed=31, retain="final")
    assert empty.reduce_sum(empty.local_shots) == 0
    assert np.isnan(empty.reduce_mean(lambda _optimizer: 1.0))

    records_result = runner.run(9, seed=31, retain="all")
    root_records = records_result.gather_records(root=0)
    last_records = records_result.gather_records(root=comm.Get_size() - 1)
    if comm.Get_rank() == 0:
        assert root_records == ((),) * 9
    if comm.Get_rank() == comm.Get_size() - 1:
        assert last_records == ((),) * 9


@pytest.mark.integration
def test_real_mpi_gathers_trajectory_records_in_global_order():
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("run this test under mpiexec -n 2 or more")

    identity = np.eye(2)
    flip = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    channel = pepsy.TrajectoryChannel.mixture(
        (("I", 0.5, identity), ("X", 0.5, flip))
    )
    runner = pepsy.MPIShotRunner(
        lambda: pepsy.MpsStabOptimizer(1, chi=4),
        [pepsy.TrajectoryEvent(channel, 0)],
        comm=comm,
    )
    result = runner.run(9, seed=53, retain="all")
    serial = pepsy.MPIShotRunner(
        lambda: pepsy.MpsStabOptimizer(1, chi=4),
        [pepsy.TrajectoryEvent(channel, 0)],
        comm=MPI.COMM_SELF,
    ).run(9, seed=53, retain="all")
    gathered = result.gather_records(root=0)
    if comm.Get_rank() == 0:
        gathered_labels = tuple(record[0].label for record in gathered)
        serial_labels = tuple(record[0].label for record in serial.local_result.records)
        assert gathered_labels == serial_labels


@pytest.mark.integration
@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
def test_real_mpi_importance_reduction_matches_shot_estimator(strategy):
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("run this test under mpiexec -n 2 or more")

    identity = np.eye(2)
    flip = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    channel = pepsy.TrajectoryChannel.mixture(
        (("I", 0.99, identity), ("X", 0.01, flip))
    )
    policy = pepsy.ImportanceSamplingPolicy({0: {"I": 0.5, "X": 0.5}})
    result = pepsy.MPIShotRunner(
        lambda: pepsy.MpsStabOptimizer(1, chi=4),
        [pepsy.TrajectoryEvent(channel, 0)],
        comm=comm,
    ).run(
        256,
        seed=19,
        strategy=strategy,
        importance_sampling=policy,
        retain="all",
    )
    values = [int(records[0].label == "X") for records in result.local_result.records]
    factors = np.asarray(result.local_result.counts) * np.asarray(
        result.local_result.weights
    )
    local_numerator = float(np.dot(values, factors))
    expected = comm.allreduce(local_numerator, op=MPI.SUM) / result.shots
    by_optimizer = {
        id(optimizer): value
        for optimizer, value in zip(result.local_result.optimizers, values)
    }
    actual = result.reduce_mean(lambda optimizer: by_optimizer[id(optimizer)])
    assert actual == pytest.approx(expected)


@pytest.mark.integration
def test_real_mpi_independent_shot_seeds_match_mpi_self():
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("run this test under mpiexec -n 2 or more")

    gates = [(np.eye(2), 0)]
    distributed = pepsy.MPIShotRunner(
        _probe_factory,
        gates,
        comm=comm,
    ).run(7, seed=41, retain="final")
    serial = pepsy.MPIShotRunner(
        _probe_factory,
        gates,
        comm=MPI.COMM_SELF,
    ).run(7, seed=41, retain="final")
    local_pairs = list(
        zip(
            distributed.shot_range,
            (optimizer.value for optimizer in distributed.local_result.optimizers),
        )
    )
    gathered = comm.gather(local_pairs, root=0)
    if comm.Get_rank() == 0:
        distributed_values = [
            value
            for pairs in sorted(gathered, key=lambda items: items[0][0] if items else 7)
            for _shot_id, value in pairs
        ]
        serial_values = [
            optimizer.value for optimizer in serial.local_result.optimizers
        ]
        assert distributed_values == pytest.approx(serial_values)


@pytest.mark.integration
def test_real_mpi_synchronizes_factory_failures():
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("run this test under mpiexec -n 2 or more")

    def failing_factory():
        raise RuntimeError("integration factory failure")

    with pytest.raises(pepsy.MPIShotError, match="integration factory failure"):
        pepsy.MPIShotRunner(
            failing_factory,
            [(np.eye(2), 0)],
            comm=comm,
        ).run(3, seed=2)


@pytest.mark.integration
def test_real_mpi_coalesces_within_each_rank():
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("run this test under mpiexec -n 2 or more")

    gate = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    result = pepsy.MPIShotRunner(
        lambda: pepsy.MpsStabOptimizer(1, chi=4),
        [(gate, 0)],
        comm=comm,
    ).run(
        8,
        seed=17,
        error_model=pepsy.PauliErrorModel.bit_flip(0.1),
        strategy="coalesced",
        retain="final",
    )

    assert result.strategy == "coalesced"
    assert result.reduce_sum(result.local_result.shots) == 8

    streamed = pepsy.MPIShotRunner(
        lambda: pepsy.MpsStabOptimizer(1, chi=4),
        [(gate, 0)],
        comm=comm,
    ).run(
        8,
        seed=17,
        error_model=pepsy.PauliErrorModel.bit_flip(0.1),
        strategy="coalesced",
        observable=lambda _optimizer: 1.0,
        chunk_size=2,
    )
    assert streamed.reduce_mean() == pytest.approx(1.0)

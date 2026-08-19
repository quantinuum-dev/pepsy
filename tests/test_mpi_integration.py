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

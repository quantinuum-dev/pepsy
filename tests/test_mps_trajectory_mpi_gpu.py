"""Opt-in real multi-rank GPU replay (ranks may share one CUDA device)."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import MpsOptimizer

pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

pytestmark = [pytest.mark.optional, pytest.mark.integration]


@pytest.mark.parametrize("backend", ["torch", "cupy"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
def test_real_gpu_ranks_match_cpu_and_streaming(backend, dtype, strategy):
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires mpiexec -n 2 or more")
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        device = comm.rank % torch.cuda.device_count()
        convert = lambda x: torch.tensor(x, device=f"cuda:{device}")
    else:
        cp = pytest.importorskip("cupy")
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
        cp.cuda.Device(comm.rank % cp.cuda.runtime.getDeviceCount()).use()
        convert = cp.asarray
    initial = qtn.MPS_rand_state(8, 4, seed=201, dtype=dtype)
    gpu = initial.copy()
    gpu.apply_to_arrays(convert)
    stream = [("x_error", .3, 1), ("cnot", 2, 3),
              ("amplitude_damping", .25, 3), ("cnot", 3, 4)]
    options = dict(shots=37, seed=82, mpi=comm, strategy=strategy,
                   workers=1, progress=False, retain="all")
    actual = MpsOptimizer(gpu, stream, chi=4, mode="swap").run(**options)
    expected = MpsOptimizer(initial, stream, chi=4, mode="swap").run(**options)
    assert actual.reduce_sum(actual.local_shots) == 37
    assert actual.local_result.counts == expected.local_result.counts
    tol = 7e-6 if dtype == "complex64" else 2e-12
    for a, b in zip(actual.local_result.optimizers, expected.local_result.optimizers):
        np.testing.assert_allclose(ar.to_numpy(a.to_dense()), b.to_dense(), atol=tol, rtol=tol)
        assert a.backend_info()["backend"] == backend
    if strategy == "coalesced":
        assert actual.local_result.diagnostics.max_gate_parent_batch >= 2

    def observable(opt):
        vector = ar.to_numpy(opt.to_dense()).ravel()
        return float(np.sum(abs(vector.reshape(2, -1)[1])**2))

    expected_mean = actual.reduce_mean(observable)
    options.update(retain="none", observable=observable)
    streamed = MpsOptimizer(gpu, stream, chi=4, mode="swap").run(**options)
    assert streamed.local_result is None
    assert streamed.reduce_mean() == pytest.approx(expected_mean, abs=tol)

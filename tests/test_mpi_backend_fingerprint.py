"""MPI configuration identity excludes process-local tensor storage IDs."""

import hashlib
import pickle

import numpy as np
import pytest

from pepsy.optimizers import mpi, noise


def test_torch_plan_fingerprint_uses_values_and_keeps_gradients():
    torch = pytest.importorskip("torch")
    matrix = torch.eye(4, dtype=torch.complex128, requires_grad=True)

    def plan(gate):
        return noise.compile_trajectory_stream([(gate, (0, 1)), ("x_error", .1, 0)])

    a = mpi._configuration_fingerprint(plan(matrix))
    assert a == mpi._configuration_fingerprint(plan(matrix.clone()))
    assert a != mpi._configuration_fingerprint(plan(matrix * 2))
    assert a != mpi._configuration_fingerprint(plan(matrix.to(torch.complex64)))
    if torch.cuda.is_available():
        assert a == mpi._configuration_fingerprint(plan(matrix.cuda()))
    assert matrix.requires_grad


def test_numpy_plan_preserves_existing_checkpoint_fingerprint():
    plan = noise.compile_trajectory_stream([(np.eye(4), (0, 1)), ("x_error", .1, 0)])
    expected = ("pickle", type(plan).__module__, type(plan).__qualname__,
                hashlib.sha256(pickle.dumps(plan, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest())
    assert mpi._fingerprint_value(plan) == expected

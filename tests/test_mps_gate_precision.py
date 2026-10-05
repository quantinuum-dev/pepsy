"""Small operator factorization accuracy and differentiation contracts."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.backends.convert import _small_matrix_precision
from pepsy.optimizers.mps.compression import _apply_dense_gate_with_method


@pytest.mark.parametrize("backend", ["torch", "cuda", "jax", "jax-x64"])
def test_small_gate_factorization_preserves_gradient_and_state_dtype(backend):
    if backend in {"torch", "cuda"}:
        torch = pytest.importorskip("torch")
        if backend == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        device = "cuda" if backend == "cuda" else "cpu"
        convert = lambda x: torch.tensor(x, dtype=torch.complex64, device=device)
    else:
        jax = pytest.importorskip("jax")
        convert = lambda x: jax.numpy.asarray(x, dtype=jax.numpy.complex64)
    rng = np.random.default_rng(173)
    matrix = convert(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    perturbation = convert(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    initial = qtn.MPS_rand_state(2, 2, seed=7, dtype="complex64")
    initial.canonize(0)
    vector = convert(initial.to_dense().ravel())

    def loss(theta, compressed):
        gate = matrix + theta * perturbation
        if compressed:
            state = initial.copy()
            state.apply_to_arrays(convert)
            _apply_dense_gate_with_method(state, gate, (0, 1), dims=(2, 2), chi=2,
                                          method="direct", cutoff=0., cutoff_mode="rsum2",
                                          info={"cur_orthog": (0, 0)})
            assert all(ar.get_dtype_name(t.data) == "complex64" for t in state.tensors)
            result = ar.do("reshape", state.to_dense(), (-1,))
        else:
            result = gate @ vector
        return ar.do("sum", ar.do("abs", result) ** 2)

    if backend in {"torch", "cuda"}:
        theta = torch.tensor(.17, requires_grad=True, device=device)
        actual = loss(theta, True)
        expected = loss(theta, False)
        gradient = torch.autograd.grad(actual, theta)[0]
        reference = torch.autograd.grad(expected, theta)[0]
    else:
        previous = jax.config.x64_enabled
        # Match MpsOptimizer's existing scoped accumulation policy, including
        # the backward contractions. A temporary x64 scope in eager replay
        # must never leak into the caller's transformation policy.
        with jax.default_matmul_precision("highest"), jax.enable_x64(backend == "jax-x64"):
            actual, gradient = jax.value_and_grad(lambda t: loss(t, True))(.17)
            expected, reference = jax.value_and_grad(lambda t: loss(t, False))(.17)
        assert jax.config.x64_enabled == previous
    np.testing.assert_allclose(ar.to_numpy(actual), ar.to_numpy(expected), rtol=3e-6)
    np.testing.assert_allclose(ar.to_numpy(gradient), ar.to_numpy(reference), rtol=1e-5)


def test_jax_operator_precision_restored_on_failure():
    jax = pytest.importorskip("jax")
    array = jax.numpy.ones((4, 4), dtype="complex64")
    previous = jax.config.x64_enabled
    with pytest.raises(RuntimeError, match="test"):
        with _small_matrix_precision(array) as working:
            assert ar.get_dtype_name(working) == "complex128"
            raise RuntimeError("test")
    assert jax.config.x64_enabled == previous
    assert ar.get_dtype_name(array) == "complex64"


def test_large_operator_does_not_allocate_double_workspace():
    array = np.ones((32, 32), dtype="complex64")
    with _small_matrix_precision(array) as working:
        assert working is array

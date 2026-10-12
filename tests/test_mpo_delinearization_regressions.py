"""Independent operator, tangent and dtype checks for exact MPO reduction."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.operators import MPOAutomaton, ham_tn
from pepsy.operators._structural_compression import _delinearize_mpo


def _convert(backend, dtype):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        return lambda a: torch.tensor(a, dtype=getattr(torch, dtype))
    if backend == "jax":
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        if dtype == "float64":
            jax.config.update("jax_enable_x64", True)
        return lambda a: jax.device_put(jnp.asarray(a, dtype=dtype), jax.devices("cpu")[0])
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CUDA unavailable")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA unavailable")
        return lambda a: cp.asarray(a, dtype=dtype)
    return lambda a: np.asarray(a, dtype=dtype)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "cupy"])
@pytest.mark.parametrize("dtype,small", [("float64", 1e-15), ("float32", 1e-6)])
@pytest.mark.parametrize("length", [2, 4])
def test_unbalanced_product_terms_survive_both_mpo_reduction_routes(backend, dtype, small, length):
    """A small suffix/prefix channel can represent an order-one global term."""
    import autoray as ar

    convert = _convert(backend, dtype)
    x = np.array([[0., 1.], [1., 0.]])
    z = np.diag([1., -1.])
    identity = np.eye(2)
    expected = np.kron(x, np.eye(2 ** (length - 1)))
    expected += np.kron(np.kron(z, np.eye(2 ** (length - 2))), z)
    for reverse in (False, True):
        operators = (small*z, z/small) if not reverse else (z/small, small*z)
        automaton = MPOAutomaton.from_product_terms(length, [
            ((0, length-1), (convert(x), convert(identity)), 1.),
            ((0, length-1), tuple(map(convert, operators)), 1.),
        ])
        direct = automaton.to_mpo()
        raw = automaton.to_mpo(delinearize=False)
        _delinearize_mpo(raw)
        for mpo in (direct, raw):
            np.testing.assert_allclose(ar.to_numpy(mpo.to_dense()), expected,
                                       rtol=2e-6 if dtype == "float32" else 1e-12, atol=1e-12)


@pytest.mark.parametrize("initial", [0., .1])
@pytest.mark.parametrize("route", ["arrays", "tensors"])
def test_trainable_term_preserves_independent_derivative_at_zero(initial, route):
    torch = pytest.importorskip("torch")
    x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    z = torch.diag(torch.tensor([1., -1.], dtype=torch.float64))
    theta = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    automaton = MPOAutomaton.from_product_terms(2, [
        ((0, 1), (x, x), 1.), ((0, 1), (z, z), theta),
    ])
    mpo = automaton.to_mpo(delinearize=route == "arrays")
    report = mpo.pepsy_delinearization if route == "arrays" else _delinearize_mpo(mpo)
    dense = mpo.to_dense()
    torch.testing.assert_close(dense, torch.kron(x, x) + theta*torch.kron(z, z))
    value = (dense * torch.kron(z, z)).sum() / 4
    torch.testing.assert_close(torch.autograd.grad(value, theta)[0], torch.ones_like(theta))
    assert "trainable" in report["skipped_reason"]


@pytest.mark.parametrize("route", ["arrays", "tensors"])
def test_float16_mpo_preserves_dtype_without_unsupported_svd(route):
    torch = pytest.importorskip("torch")
    x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float16)
    automaton = MPOAutomaton.from_product_terms(2, [((0, 1), (x, x), 1.)])
    mpo = automaton.to_mpo(delinearize=route == "arrays")
    if route == "tensors":
        _delinearize_mpo(mpo)
    assert all(t.data.dtype == torch.float16 for t in mpo)
    torch.testing.assert_close(mpo.to_dense(), torch.kron(x, x))


def test_reduction_keeps_derivatives_carried_only_by_the_other_endpoint():
    """Constant tensors must not choose a rank for a trainable partner."""
    torch = pytest.importorskip("torch")
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    z = torch.diag(torch.tensor([1., -1.], dtype=torch.float64))
    mpo = qtn.MatrixProductOperator([
        torch.stack((x, z)), torch.stack((x, theta*z)),
    ], shape="lrud")
    report = _delinearize_mpo(mpo)
    assert not report["changed"]
    value = (mpo.to_dense()*torch.kron(z, z)).sum()/4
    torch.testing.assert_close(torch.autograd.grad(value, theta)[0], torch.ones_like(theta))


@pytest.mark.parametrize("mode", ["term", "automaton"])
@pytest.mark.parametrize("delinearize", [False, True])
def test_hamiltonian_builder_preserves_unbalanced_terms_without_numerical_compression(mode, delinearize):
    x = np.array([[0., 1.], [1., 0.]])
    z = np.diag([1., -1.])
    builder = ham_tn(shape=4, max_bond=None)
    mpo = builder.to_mpo([
        ((x,), (0,), 1.), ((1e-15*z, 1e15*z), (0, 3), 1.),
    ], compress=False, mode=mode, delinearize=delinearize)
    expected = np.kron(x, np.eye(8)) + np.kron(np.kron(z, np.eye(4)), z)
    np.testing.assert_allclose(mpo.to_dense(), expected, rtol=1e-12, atol=1e-12)


def test_hamiltonian_builder_keeps_torch_float16_without_numerical_compression():
    import pepsy

    torch = pytest.importorskip("torch")
    x = np.array([[0., 1.], [1., 0.]])
    builder = ham_tn(shape=2, max_bond=None, to_backend=pepsy.backend_torch(dtype=torch.float16))
    mpo = builder.to_mpo([((x, x), (0, 1), 1.)], compress="automaton")
    assert all(t.data.dtype == torch.float16 for t in mpo)
    torch.testing.assert_close(mpo.to_dense(), torch.tensor(np.kron(x, x), dtype=torch.float16))

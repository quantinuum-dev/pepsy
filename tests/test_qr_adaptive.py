"""Adaptive QR policy must survive scope restoration and batch fallbacks."""

import pytest

import pepsy


@pytest.fixture(autouse=True)
def _native_torch_policy():
    """Keep process-global registration tests independent of suite ordering."""
    pytest.importorskip("torch")
    pepsy.reset_linalg_registrations(backend="torch")
    try:
        yield
    finally:
        pepsy.reset_linalg_registrations(backend="torch")


@pytest.mark.parametrize("complex_input", [False, True])
def test_adaptive_qr_preserves_finite_native_vjp_after_scope_exit(complex_input):
    torch = pytest.importorskip("torch")
    import autoray as ar

    dtype = torch.complex128 if complex_input else torch.float64
    matrix = torch.tensor([[1., 1., 1.], [0., 1e-8, 1.]], dtype=dtype, requires_grad=True)
    weight_q = torch.tensor([[0.2, -0.1], [0.5, 0.3]], dtype=dtype)
    weight_r = torch.ones((2, 3), dtype=dtype)
    previous = pepsy.get_torch_linalg_config()
    with pepsy.TorchLinalgConfig(
        mode="complex" if complex_input else "real",
        stabilized=True, qr_rank_policy="adaptive",
    ).activated():
        q, r = ar.do("linalg.qr", matrix)
    assert pepsy.get_torch_linalg_config() == (previous or pepsy.TorchLinalgConfig())
    actual = torch.autograd.grad(
        (q.conj() * weight_q).real.sum() + (r.conj() * weight_r).real.sum(), matrix,
    )[0]
    q, r = torch.linalg.qr(matrix)
    expected = torch.autograd.grad(
        (q.conj() * weight_q).real.sum() + (r.conj() * weight_r).real.sum(), matrix,
    )[0]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-12)


@pytest.mark.parametrize("complex_input", [False, True])
def test_adaptive_qr_regularizes_only_nonfinite_batch_members(complex_input):
    torch = pytest.importorskip("torch")
    import autoray as ar

    dtype = torch.complex128 if complex_input else torch.float64
    matrix = torch.tensor([
        [[1., 0., 1.], [0., 1., 1.]],
        [[1., 1., 1.], [0., 0., 1.]],
    ], dtype=dtype, requires_grad=True)
    weight = torch.tensor([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        [[-0.2, 0.1, 0.5], [0.6, 0.3, 0.4]],
    ], dtype=dtype)
    with pepsy.TorchLinalgConfig(
        mode="complex" if complex_input else "real",
        stabilized=True, qr_rank_policy="adaptive",
    ).activated():
        q, r = ar.do("linalg.qr", matrix)
    with pytest.warns(RuntimeWarning, match="adaptive backward"):
        gradient = torch.autograd.grad(((q @ r).conj() * weight).real.sum(), matrix)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, weight, atol=1e-7, rtol=1e-5)

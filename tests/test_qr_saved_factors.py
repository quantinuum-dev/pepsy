"""Saved-factor QR pullbacks match independent native Torch differentiation."""

import pytest


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype_name', ['float64', 'complex128', 'float32', 'complex64'])
@pytest.mark.parametrize('shape', [(9, 4), (4, 4), (3, 7), (2, 3, 9, 4), (2, 3, 3, 7)])
@pytest.mark.parametrize('cotangent', ['both', 'q', 'r'])
def test_saved_qr_vjp_matches_native(device, dtype_name, shape, cotangent):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from pepsy.backends.linalg_torch import _qr_backward_from_factors

    generator = torch.Generator(device=device).manual_seed(491)
    dtype = getattr(torch, dtype_name)
    a = torch.randn(shape, dtype=dtype, device=device, generator=generator, requires_grad=True)
    q, r = torch.linalg.qr(a)
    dq = torch.randn(q.shape, dtype=dtype, device=device, generator=generator)
    dr = torch.randn(r.shape, dtype=dtype, device=device, generator=generator)
    if cotangent == 'q':
        dr = None
    elif cotangent == 'r':
        dq = None
    expected = torch.autograd.grad((q, r), a,
        (torch.zeros_like(q) if dq is None else dq, torch.zeros_like(r) if dr is None else dr))[0]
    actual = _qr_backward_from_factors(q.detach(), r.detach(), dq, dr)
    assert not actual.is_conj()
    tolerance = 3e-5 if dtype in (torch.float32, torch.complex64) else 2e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('complex_input', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_eager_adaptive_backward_does_not_repeat_qr(monkeypatch, complex_input, device):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    import autoray as ar
    from pepsy import TorchLinalgConfig

    dtype = torch.complex128 if complex_input else torch.float64
    a = torch.randn((7, 3), dtype=dtype, device=device,
                    generator=torch.Generator(device=device).manual_seed(773), requires_grad=True)
    weight = torch.randn(a.shape, dtype=dtype, device=device,
                         generator=torch.Generator(device=device).manual_seed(21))
    with TorchLinalgConfig(mode='complex' if complex_input else 'real', stabilized=True,
                           qr_rank_policy='adaptive').activated():
        q, r = ar.do('linalg.qr', a)

    def repeated_qr(*args, **kwargs):
        raise AssertionError('Backward must reuse forward QR factors')

    monkeypatch.setattr(torch.linalg, 'qr', repeated_qr)
    gradient = torch.autograd.grad(((q @ r).conj()*weight).real.sum(), a)[0]
    torch.testing.assert_close(gradient, weight, atol=1e-12, rtol=1e-12)

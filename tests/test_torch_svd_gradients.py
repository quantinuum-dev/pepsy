"""Smooth truncated-SVD derivatives and bounded singular-case extensions."""

import pytest

torch = pytest.importorskip("torch")

from pepsy.backends.linalg_torch import SVD, SVD_real, _svd_reciprocal


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("shape", [(5, 4), (4, 5), (2, 4, 4)])
@pytest.mark.parametrize("scale", [1.e-100, 1., 1.e100])
def test_resolved_truncated_svd_vjp_matches_native(dtype, shape, scale):
    """Small but resolved singular values must not acquire damping bias."""
    generator = torch.Generator().manual_seed(714)
    seed = torch.randn(shape, dtype=dtype, generator=generator)
    u, _, vh = torch.linalg.svd(seed, full_matrices=False)
    spectrum = torch.tensor([3., .003, .0002, .0001], dtype=torch.float64)
    # Keep scale outside the leaf: the checked gradient remains O(1) even
    # when the decomposition itself operates on extreme spectral scales.
    matrix = ((u * spectrum) @ vh).requires_grad_()
    weight = torch.randn(shape, dtype=dtype, generator=generator)

    def loss(svd, spectral_scale):
        left, values, right = svd(matrix * spectral_scale)
        truncated = (left[..., :2] * (values[..., :2] / spectral_scale).unsqueeze(-2)) @ right[..., :2, :]
        return (weight.conj() * truncated).real.sum()

    function = SVD_real if dtype == torch.float64 else SVD
    actual = torch.autograd.grad(loss(function.apply, scale), matrix)[0]
    # Use an ordinary-scale native reference: its squared-spectrum formula
    # can itself overflow for huge complex spectra.
    expected = torch.autograd.grad(
        loss(lambda x: torch.linalg.svd(x, full_matrices=False), 1.), matrix,
    )[0]
    torch.testing.assert_close(actual, expected, atol=3.e-10, rtol=1.e-9)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("shape", [(5, 2), (2, 5), (2, 2)])
def test_small_resolved_singular_value_preserves_reconstruction_gradient(dtype, shape):
    """Rectangular and complex-phase reciprocals use numerical rank, not gaps."""
    matrix = torch.zeros(shape, dtype=dtype)
    matrix[0, 0], matrix[1, 1] = 1., 1.e-8
    matrix.requires_grad_()
    generator = torch.Generator().manual_seed(107)
    weight = torch.randn(shape, dtype=dtype, generator=generator)
    function = SVD_real if dtype == torch.float64 else SVD
    u, s, vh = function.apply(matrix)
    loss = (weight.conj() * ((u * s) @ vh)).real.sum()
    gradient = torch.autograd.grad(loss, matrix)[0]
    torch.testing.assert_close(gradient, weight, atol=1.e-12, rtol=1.e-12)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("shape", [(5, 3), (3, 5)])
def test_truncated_svd_gradcheck(dtype, shape):
    """Finite differences through a genuinely rank-reduced reconstruction."""
    generator = torch.Generator().manual_seed(2026)
    matrix = torch.randn(shape, dtype=dtype, generator=generator, requires_grad=True)
    function = SVD_real if dtype == torch.float64 else SVD

    def truncate(x):
        u, s, vh = function.apply(x)
        return (u[:, :2] * s[:2]) @ vh[:2, :]

    assert torch.autograd.gradcheck(truncate, (matrix,), atol=2.e-7, rtol=2.e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize("spectrum", [[0., 0., 0.], [2., 2., .5], [2., 1., 0.]])
def test_singular_svd_vjp_remains_finite(dtype, spectrum):
    """The extension stays finite; individual vectors have no derivative at ties."""
    matrix = torch.diag(torch.tensor(spectrum, dtype=dtype)).requires_grad_()
    function = SVD if matrix.is_complex() else SVD_real
    u, s, vh = function.apply(matrix)
    generator = torch.Generator().manual_seed(517)
    loss = sum(
        (value.conj() * torch.randn(value.shape, dtype=value.dtype, generator=generator)).real.sum()
        for value in (u, s, vh)
    )
    assert torch.isfinite(torch.autograd.grad(loss, matrix)[0]).all()


@pytest.mark.parametrize("scale", [0., 1.e-200, 1., 1.e200])
def test_svd_reciprocal_has_compact_finite_stabilization(scale):
    spectrum_scale = torch.tensor(scale, dtype=torch.float64)
    # Include both join points and zero; this also catches unsafe evaluation
    # of an unselected branch of torch.where.
    multiples = torch.tensor([-2., -1., -.5, 0., .5, 1., 2.], dtype=torch.float64)
    threshold = max(scale * 1.e-6, torch.finfo(torch.float64).tiny)
    x = (multiples * threshold).requires_grad_()
    actual = _svd_reciprocal(x, spectrum_scale)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual * threshold,
                               torch.tensor([-.5, -1., -.875, 0., .875, 1., .5], dtype=x.dtype))
    if scale == 1.:
        slope = torch.autograd.grad(actual.sum(), x)[0]
        torch.testing.assert_close(slope[[1, 5]], -1 / x[[1, 5]]**2)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_configured_gap_tolerance_is_captured_for_backward(dtype):
    """A resolved narrow truncation gap must retain its native derivative."""
    import autoray as ar
    from pepsy.backends import TorchLinalgConfig

    generator = torch.Generator().manual_seed(142)
    seed = torch.randn((4, 4), dtype=dtype, generator=generator)
    left, _, right = torch.linalg.svd(seed, full_matrices=False)
    spectrum = torch.tensor([1., 1. - 2.e-7, .4, .1], dtype=torch.float64)
    matrix = ((left * spectrum) @ right).detach().requires_grad_()
    weight = torch.randn((4, 4), dtype=dtype, generator=generator)

    def loss(svd):
        u, s, vh = svd(matrix)
        return (weight.conj() * ((u[:, :1] * s[:1]) @ vh[:1])).real.sum()

    expected = torch.autograd.grad(loss(lambda a: torch.linalg.svd(a, full_matrices=False)), matrix)[0]
    mode = "complex" if matrix.is_complex() else "real"
    with TorchLinalgConfig(mode=mode, stabilized=True).activated():
        default_value = loss(lambda a: ar.do("linalg.svd", a))
    default_gradient = torch.autograd.grad(default_value, matrix)[0]
    assert torch.linalg.vector_norm(default_gradient - expected) > .1 * torch.linalg.vector_norm(expected)
    with TorchLinalgConfig(mode=mode, stabilized=True, svd_gap_rtol=1.e-8).activated():
        value = loss(lambda a: ar.do("linalg.svd", a))
    # Backward runs after the scoped registration has been restored.
    actual = torch.autograd.grad(value, matrix)[0]
    torch.testing.assert_close(actual, expected, atol=1.e-3, rtol=2.e-8)


@pytest.mark.parametrize("tolerance", [0., -1., float("nan"), float("inf")])
def test_invalid_svd_gap_tolerance_rejected(tolerance):
    from pepsy.backends import TorchLinalgConfig

    with pytest.raises(ValueError, match="svd_gap_rtol"):
        TorchLinalgConfig(svd_gap_rtol=tolerance)

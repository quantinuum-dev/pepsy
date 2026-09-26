"""Scale extraction must preserve the represented gauge and its derivative."""

import numpy as np
import pytest

from pepsy.operators import renorm_gauge
from pepsy.tensors import id_to_pepo


@pytest.mark.parametrize("scale", [0.0, 1e-200, 1e-12, 1.0, 1e200])
@pytest.mark.parametrize("smudge", [0.0, 1e-12, 0.1])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_gauge_scale_reconstruction_and_derivative(scale, smudge, backend):
    network = id_to_pepo(2, 1, chi=2)
    bond = next(iter(network["I0,0"].bonds(network["I1,0"])))
    original = np.array([scale, scale / 2], dtype=np.float64)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        values = torch.tensor(original, requires_grad=True)
    else:
        values = original.copy()
    gauges = {bond: values}
    renorm_gauge(network, gauges, ((0, 0), (1, 0)), smudge=smudge)
    reconstructed = 10.0**network.exponent * gauges[bond]
    if backend == "torch":
        assert torch.isfinite(network.exponent)
        reconstructed.sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values), atol=1e-12, rtol=1e-12)
        actual = reconstructed.detach().numpy()
    else:
        assert np.isfinite(network.exponent)
        actual = reconstructed
    # Absolute tolerance zero matters for the subnormal-scale regressions.
    np.testing.assert_allclose(actual, original, rtol=2e-12, atol=0)


@pytest.mark.parametrize("smudge", [-1.0, np.inf, np.nan])
def test_gauge_scale_rejects_invalid_floor(smudge):
    with pytest.raises(ValueError, match="smudge"):
        renorm_gauge(id_to_pepo(2, 1), {}, ((0, 0), (1, 0)), smudge=smudge)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_smallest_nonzero_gauge_keeps_positive_scale_without_floor(backend):
    network = id_to_pepo(2, 1, chi=8)
    bond = next(iter(network["I0,0"].bonds(network["I1,0"])))
    original = np.zeros(8, dtype=np.float64)
    original[0] = np.nextafter(0., 1.)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        values = torch.tensor(original, requires_grad=True)
    else:
        values = original.copy()
    gauges = {bond: values}
    renorm_gauge(network, gauges, ((0, 0), (1, 0)), smudge=0.)
    reconstructed = 10.0**network.exponent * gauges[bond]
    if backend == "torch":
        assert torch.isfinite(network.exponent)
        assert torch.isfinite(gauges[bond]).all()
        reconstructed.sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values))
        actual = reconstructed.detach().numpy()
    else:
        assert np.isfinite(network.exponent)
        assert np.isfinite(gauges[bond]).all()
        actual = reconstructed
    np.testing.assert_array_equal(actual, original)

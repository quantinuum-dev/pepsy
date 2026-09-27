"""Active-block storage guards must work before differentiable materialization."""

from dataclasses import replace
from math import prod

import pytest

from pepsy.operators import PauliPEPOBasis


@pytest.mark.optional
@pytest.mark.integration
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_torch_active_storage_metadata_preserves_autodiff(dtype_name):
    torch = pytest.importorskip("torch")
    theta = torch.tensor([.2, -.3], dtype=torch.float64, requires_grad=True)
    active = PauliPEPOBasis.compile(2, 2, [("onsite", "X"), ("edge", "ZZ")], order=2).exp(
        -.1j, coefficients=theta, materialize=False)
    # Test storage metadata for both block dtypes, independently of the
    # builder's single-precision operator-bank policy.
    if dtype_name == "float32":
        active = replace(active, blocks={site: {key: block.to(torch.complex64)
                          for key, block in blocks.items()} for site, blocks in active.blocks.items()})
    expected = active.dense_nbytes
    assert active.active_nbytes > 0
    pepo = active.to_pepo()
    assert expected == sum(t.data.numel() * t.data.element_size() for t in pepo)
    value = torch.trace(pepo.to_dense()).real
    gradient, = torch.autograd.grad(value, theta, retain_graph=True)
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 1e-4
    network = active.to_trace_network()
    assert not network.outer_inds()
    assert sum(t.data.numel() * t.data.element_size() for t in network) == active.trace_nbytes == expected // 4
    actual = network.contract(all, optimize="greedy").real
    actual_gradient, = torch.autograd.grad(actual, theta)
    tolerance = 2e-6 if dtype_name == "float32" else 2e-12
    torch.testing.assert_close(actual, value, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_gradient, gradient, atol=tolerance, rtol=tolerance)


@pytest.mark.optional
@pytest.mark.integration
def test_localized_joint_trace_jax_and_periodic_physical_closure():
    import numpy as np

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from pepsy.operators import PauliPEPOTerm

    basis = PauliPEPOBasis.compile(2, 2, [PauliPEPOTerm("onsite", "X", where=(0, 0)),
        PauliPEPOTerm("edge", "ZZ", where=((0, 0), (0, 1)), direction="r")],
        cyclic=True, order=2)
    with jax.enable_x64(True):
        def evaluate(theta, trace_first):
            active = basis.exp(-.1j, coefficients=theta, materialize=False)
            if trace_first:
                return active.to_trace_network().contract(all, optimize="greedy").real
            return jnp.trace(active.to_pepo().to_dense()).real

        theta = jnp.array([.2, -.3], dtype=jnp.float64)
        value, grad = jax.value_and_grad(lambda x: evaluate(x, True))(theta)
        expected, expected_grad = jax.value_and_grad(lambda x: evaluate(x, False))(theta)
        np.testing.assert_allclose(value, expected, atol=2e-12)
        np.testing.assert_allclose(grad, expected_grad, atol=2e-12)


def test_dense_storage_estimate_uses_unbounded_integer_arithmetic(monkeypatch):
    from pepsy.operators import pepo_active

    active = PauliPEPOBasis.compile(3, 3, [("onsite", "X")], order=1).exp(-.1j, coefficients=[.2])
    maps = {(site, direction): range(10**6)
            for site, directions in active.site_directions.items() for direction in directions}
    monkeypatch.setattr(pepo_active, "_local_bond_sector_maps", lambda _: maps)
    expected = 16 * 4 * sum(prod(10**6 for _ in directions) for directions in active.site_directions.values())
    assert expected > 2**63
    assert active.dense_nbytes == expected


@pytest.mark.optional
@pytest.mark.integration
def test_batched_local_exponentials_match_analytic_onsite_values_and_gradients():
    torch = pytest.importorskip("torch")
    from pepsy.operators import PauliPEPOTerm

    basis = PauliPEPOBasis.compile(2, 2, [PauliPEPOTerm("onsite", "X", where=site)
        for site in ((0, 0), (0, 1), (1, 0), (1, 1))], order=1)
    theta = torch.tensor([.04377333, .032, -.06, .015], dtype=torch.float64, requires_grad=True)
    active = basis.exp(1j, coefficients=theta, materialize=False)
    actual = active.to_trace_network().contract(all, optimize="greedy").real / 16
    expected = theta.cos().prod()
    got, = torch.autograd.grad(actual, theta)
    want, = torch.autograd.grad(expected, theta)
    torch.testing.assert_close(actual, expected, atol=3e-14, rtol=3e-14)
    torch.testing.assert_close(got, want, atol=3e-13, rtol=3e-13)

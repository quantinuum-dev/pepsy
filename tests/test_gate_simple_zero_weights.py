"""Unsmeared simple update preserves exact-zero external gauge support."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.operators import gate_simple


def _fixture(backend, weight=.7):
    state = qtn.MPS_rand_state(4, bond_dim=2, seed=19, dtype="complex128")
    weights = np.array([0., weight])
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
        weights = torch.tensor(weights, dtype=torch.float64)
    gauges = {state.bond(0, 1): weights, state.bond(2, 3): weights}
    return state, gauges


def _vector(state, gauges):
    physical = state.copy()
    # Absorb each full weight once. Splitting sqrt(g) across both ends has
    # an unrelated singular derivative at g=0 even for a smooth contraction.
    for ix, weight in gauges.items():
        tensor = next(t for t in physical.tensors if ix in t.inds)
        tensor.multiply_index_diagonal_(ix, weight)
    return physical.to_dense().reshape(-1)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("weight", [.7, 0., 1e-30])
@pytest.mark.parametrize("where", [(1, 2), (0, 3)])
def test_zero_external_weights_match_dense_and_preserve_source(backend, weight, where):
    state, gauges = _fixture(backend, weight)
    original_weights = dict(gauges)
    original_arrays = [ar.to_numpy(t.data).copy() for t in state]
    before = ar.to_numpy(_vector(state, gauges))
    rng = np.random.default_rng(10)
    gate = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    order = [*where, *(i for i in range(4) if i not in where)]
    expected = (gate @ before.reshape((2,) * 4).transpose(order).reshape(4, -1))
    expected = expected.reshape((2,) * 4).transpose(np.argsort(order)).reshape(-1)
    if backend == "torch":
        import torch
        gate = torch.tensor(gate)
    result = gate_simple(state, gate, where=where, gauges=gauges,
                         cutoff=0., smudge=0., renorm=False, inplace=False)
    scale = np.linalg.norm(expected) or 1.
    np.testing.assert_allclose(ar.to_numpy(_vector(result, gauges)) / scale, expected / scale,
                               rtol=1e-11, atol=1e-12)
    assert all(np.isfinite(ar.to_numpy(t.data)).all() for t in result)
    if where == (1, 2):
        for ix, weight in original_weights.items():
            assert gauges[ix] is weight
    for tensor, original in zip(state, original_arrays):
        np.testing.assert_array_equal(ar.to_numpy(tensor.data), original)


def test_external_weights_are_restored_after_gate_error():
    state, gauges = _fixture("numpy")
    original = dict(gauges)
    with pytest.raises(ValueError):
        gate_simple(state, np.ones((3, 3)), where=(1, 2), gauges=gauges,
                    cutoff=0., smudge=0., renorm=False)
    for ix, weight in original.items():
        assert gauges[ix] is weight


@pytest.mark.parametrize("max_bond", [1, 4])
def test_zero_support_torch_gradient_with_and_without_truncation(max_bond):
    torch = pytest.importorskip("torch")
    from pepsy.backends import TorchLinalgConfig

    def loss(theta):
        state, gauges = _fixture("torch")
        gate = torch.diag(torch.stack((1 + theta, 1 - theta,
                                      2 + theta, 1.2 - theta))).to(torch.complex128)
        gate_simple(state, gate, where=(1, 2), gauges=gauges,
                    max_bond=max_bond, cutoff=0., smudge=0., renorm=False,
                    contract="split")
        assert state.bond_size(1, 2) == max_bond
        return abs(_vector(state, gauges)).square().sum()

    theta = torch.tensor(.23, dtype=torch.float64, requires_grad=True)
    with TorchLinalgConfig(stabilized=True).activated():
        value = loss(theta)
        grad, = torch.autograd.grad(value, theta)
        for step in (1e-4, 1e-5):
            fd = (loss(theta.detach() + step) - loss(theta.detach() - step)) / (2 * step)
            torch.testing.assert_close(grad, fd, rtol=2e-6, atol=1e-9)

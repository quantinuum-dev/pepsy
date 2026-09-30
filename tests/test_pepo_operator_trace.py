"""PEPO traces measure the returned operator, with all selected approximations."""

from unittest.mock import patch

import numpy as np
import pytest

from pepsy.operators import (
    ClusterPlan, MPOParameter, MPOProductTerm, PEPOClusterProductExpansion,
    PauliPEPOBasis, trace_pepo,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]


def builder(kind, *, rank=None):
    terms = [MPOProductTerm.from_pauli((0, 1), "XX", coefficient=MPOParameter("J")),
             MPOProductTerm.from_pauli((0, 1), "ZZ", coefficient=.31)]
    terms += [MPOProductTerm.from_pauli((i,), "Y", coefficient=.2) for i in range(2)]
    if kind == "uniform":
        basis = PauliPEPOBasis(1, 2, [("edge", "XX", MPOParameter("J")),
                                     ("edge", "ZZ", .31), ("onsite", "Y", .2)],
                              order=2, max_tree_rank=rank)
        return PEPOClusterProductExpansion.from_bases([basis])
    if kind == "routed":
        terms = [MPOProductTerm.from_pauli(tuple(3 if s == 1 else s for s in term.sites),
                                           word, coefficient=term.coefficient)
                 for term, word in zip(terms, ("XX", "ZZ", "Y", "Y"))]
    plan = ClusterPlan.from_terms(terms, sites=4 if kind == "routed" else 2,
                                 shape=(2, 2) if kind == "routed" else (1, 2), cluster_size=2)
    return PEPOClusterProductExpansion.from_plan(plan, [terms],
                                                layout="graph" if kind == "graph" else "square",
                                                max_tree_rank=rank)


@pytest.mark.parametrize("kind", ["uniform", "square", "graph", "routed"])
def test_trace_constructs_and_contracts_compressed_pepo(kind):
    source = builder(kind)
    params = {"J": .4}
    compression = source.prepare_compression(-.31j, params, max_tree_rank=1)
    active = source.exp(-.31j, params, compression=compression, materialize=False)
    network = source.exp(-.31j, params, compression=compression, materialize=True)
    if kind == "graph":
        dense = network.to_dense([("graph-bra", i) for i in range(2)],
                                 [("graph-ket", i) for i in range(2)])
    else:
        dense = network.to_dense()
    expected = np.trace(dense)
    np.testing.assert_allclose(active.trace(), expected, atol=2e-13)
    np.testing.assert_allclose(trace_pepo(network), expected, atol=2e-13)
    np.testing.assert_allclose(trace_pepo(network, normalized=True), expected/len(dense), atol=2e-13)
    assert abs(expected-source.partition_trace_exp(-.31j, params)) > 1e-5
    with patch.object(source, "partition_trace_exp", side_effect=AssertionError("scalar shortcut")), \
            patch.object(source, "exp", wraps=source.exp) as build:
        np.testing.assert_allclose(source.trace_exp(-.31j, params, compression=compression), expected,
                                   atol=2e-13)
        assert build.call_count == 1
    np.testing.assert_allclose(source.trace_exp(-.31j, params, compression=compression,
                                               contract_opts={"optimize": "greedy"}),
                               expected, atol=2e-13)
    # Trace the actual result after a caller changes it; no cached scalar target.
    changed = network.multiply(2.0)
    np.testing.assert_allclose(trace_pepo(changed), 2*expected, atol=2e-13)


def test_quimb_compression_is_part_of_pepo_trace():
    source = builder("uniform")
    expected = source.exp(-.31j, {"J": .4}, compress=True, max_bond=1).to_dense()
    np.testing.assert_allclose(source.trace_exp(-.31j, {"J": .4}, compress=True, max_bond=1),
                               np.trace(expected), atol=2e-13)


@pytest.mark.parametrize("symmetry", [None, "C4"])
@pytest.mark.parametrize("order", [2, 4])
def test_short_periodic_axes_use_finite_cluster_pepo(symmetry, order):
    source = PEPOClusterProductExpansion.from_bases([
        PauliPEPOBasis(2, 2, [("onsite", "X", .2), ("edge", "ZZ", .3)],
                       order=order, cyclic=True, factorization="fixed", symmetry=symmetry)])
    active = source.exp(-.1j, materialize=False)
    expected = np.trace(active.to_dense())
    np.testing.assert_allclose(active.trace(), expected, atol=2e-12)
    np.testing.assert_allclose(source.trace_exp(-.1j), expected, atol=2e-12)
    np.testing.assert_allclose(source.partition_trace_exp(-.1j), expected, atol=2e-12)
    basis = source.factors[0].basis
    np.testing.assert_allclose(basis.compile_exp().exp(-.1j).trace(), expected, atol=2e-12)
    assert basis.cache_info["prepared_exp_modes"] == ("localized",)
    if order == 4:
        from scipy.linalg import expm

        x, z, identity = np.array([[0, 1], [1, 0]]), np.diag([1, -1]), np.eye(2)
        def embed(ops):
            matrix = np.ones((1, 1))
            for i in range(4):
                matrix = np.kron(matrix, ops.get(i, identity))
            return matrix
        h = sum(.2*embed({i: x}) for i in range(4))
        # Each edge occurs twice on the two-by-two torus.
        h += sum(.6*embed({i: z, j: z}) for i, j in ((0, 1), (0, 2), (1, 3), (2, 3)))
        np.testing.assert_allclose(active.to_dense(), expm(-.1j*h), atol=2e-12)


@pytest.mark.parametrize("kind", ["square", "graph", "routed"])
def test_rank_capped_pepo_trace_matches_materialization(kind):
    source = builder(kind, rank=1)
    active = source.exp(-.31j, {"J": .4}, materialize=False)
    np.testing.assert_allclose(source.trace_exp(-.31j, {"J": .4}),
                               np.trace(active.to_dense()), atol=2e-12)


@pytest.mark.parametrize("kind", ["uniform", "square", "graph", "routed"])
def test_sparse_trace_torch_gradients_match_compressed_operator(kind):
    torch = pytest.importorskip("torch")
    source = builder(kind)
    compression = source.prepare_compression(-.31j, {"J": .4}, max_tree_rank=2)
    for value in (.3, 0.):
        theta = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        active = source.exp(-.31j, {"J": theta}, compression=compression, materialize=False)
        actual, expected = active.trace(), torch.trace(active.to_dense())
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        ga, = torch.autograd.grad(actual.real+.3*actual.imag, theta, retain_graph=True)
        ge, = torch.autograd.grad(expected.real+.3*expected.imag, theta)
        torch.testing.assert_close(ga, ge, atol=2e-11, rtol=2e-11)


def test_sparse_trace_jax_jit_and_zero_gradients():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    source = builder("routed")
    compression = source.prepare_compression(-.31j, {"J": .4}, max_tree_rank=2)
    with jax.enable_x64(True):
        def loss(theta, dense=False):
            active = source.exp(-.31j, {"J": theta}, compression=compression, materialize=False)
            value = jnp.trace(active.to_dense()) if dense else active.trace()
            return value.real + .3*value.imag
        actual = jax.jit(jax.value_and_grad(loss))
        expected = jax.value_and_grad(lambda theta: loss(theta, dense=True))
        for value in (.3, 0.):
            for got, want in zip(actual(jnp.array(value)), expected(jnp.array(value))):
                np.testing.assert_allclose(got, want, atol=2e-12)


def test_sparse_trace_never_materializes_dense_sites_and_budget_is_explicit():
    active = builder("square").exp(-.31j, {"J": .4}, materialize=False)
    expected = np.trace(active.to_dense())
    with patch.object(type(active), "to_pepo", side_effect=AssertionError("dense allocation")):
        np.testing.assert_allclose(active.trace(), expected, atol=2e-13)
        with pytest.raises(ValueError, match="state_budget"):
            active.trace(state_budget=1)


@pytest.mark.parametrize("kind", ["uniform", "square", "graph", "routed"])
def test_materialized_trace_rejects_ignored_sparse_options(kind):
    source = builder(kind)
    with pytest.raises(ValueError, match="state_budget.*cannot be combined"):
        source.trace_exp(-.31j, {"J": .4}, state_budget=1, contract_opts={})
    with pytest.raises(TypeError, match="contract_opts must be a mapping"):
        source.trace_exp(-.31j, {"J": .4}, contract_opts=[])


def test_compressed_trace_rejects_ignored_sparse_budget():
    source = builder("uniform")
    with pytest.raises(ValueError, match="state_budget.*cannot be combined"):
        source.trace_exp(-.31j, {"J": .4}, state_budget=1, compress=True, max_bond=1)

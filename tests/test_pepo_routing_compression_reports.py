"""Exact virtual routing, explicit projected objectives, and measured work."""

from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import (
    ClusterPlan, MPOClusterFactor, MPOParameter, MPOProductTerm,
    PEPOClusterProductExpansion, PauliPEPOBasis, PauliPEPOTerm,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]
PAULI = {"I": np.eye(2), "X": np.array([[0., 1.], [1., 0.]]),
         "Y": np.array([[0., -1j], [1j, 0.]]), "Z": np.diag([1., -1.])}


def dense_term(term, nsites):
    operators = dict(zip(term.sites, term.operators))
    result = np.ones((1, 1))
    for site in range(nsites):
        result = np.kron(result, operators.get(site, PAULI["I"]))
    return result


@pytest.mark.parametrize("shape,cyclic,specs,size", [
    ((2, 2), False, [((0, 3), "XY"), ((1, 2), "ZZ")], 2),
    ((2, 3), False, [((0, 2), "YX"), ((1,), "Y")], 2),
    ((2, 2), False, [((0, 1, 3), "XYZ"), ((2,), "Y")], 3),
    ((2, 3), (False, True), [((0, 2), "YX"), ((1, 5), "XY")], 2),
])
def test_routed_square_preserves_disjoint_clusters_and_routing_site_operators(shape, cyclic, specs, size):
    nsites = np.prod(shape)
    factors = [MPOClusterFactor([MPOProductTerm.from_pauli(sites, word, coefficient=.2 + i*.1)])
               for i, (sites, word) in enumerate(specs)]
    terms = [term for factor in factors for term in factor.terms]
    plan = ClusterPlan.from_terms(terms, sites=int(nsites), shape=shape, cyclic=cyclic, cluster_size=size)
    builder = PEPOClusterProductExpansion.from_plan(plan, factors, layout="square", factorization="fixed")
    assert builder.cluster_plan is plan
    actual, report = builder.exp(-.09j, materialize=True, return_report=True)
    expected = np.eye(2**nsites, dtype=complex)
    for term in terms:
        expected = expected @ expm(-.09j * term.coefficient * dense_term(term, nsites))
    np.testing.assert_allclose(actual.to_dense(), expected, atol=3e-13)
    assert report["routing"] == "graph-wires"
    assert report["cluster_counts"] == plan.counts
    assert report["dense_nbytes"] == sum(t.data.nbytes for t in actual.tensors)


def pair_builder(*, layout="square", template=False):
    if template:
        basis = PauliPEPOBasis(1, 2, [("onsite", "Y", MPOParameter("h")),
                                     ("edge", "XX", MPOParameter("J")),
                                     ("edge", "ZZ", .31)], order=2, factorization="fixed")
        return PEPOClusterProductExpansion.from_bases([basis])
    terms = [MPOProductTerm.from_pauli((0, 1), "XX", coefficient=MPOParameter("J")),
             MPOProductTerm.from_pauli((0, 1), "ZZ", coefficient=.31)]
    terms += [MPOProductTerm.from_pauli((i,), "Y", coefficient=MPOParameter("h")) for i in range(2)]
    plan = ClusterPlan.from_terms(terms, sites=2, shape=(1, 2), cluster_size=2)
    return PEPOClusterProductExpansion.from_plan(plan, [MPOClusterFactor(terms)],
                                                layout=layout, factorization="fixed")


def dense_result(result):
    return result.to_dense() if hasattr(result, "sites") else result.to_pepo().to_dense()


@pytest.mark.parametrize("kind", ["square", "graph", "template"])
def test_frozen_compression_reduces_bonds_and_differentiates_its_actual_objective(kind):
    torch = pytest.importorskip("torch")
    builder = pair_builder(layout="graph" if kind == "graph" else "square", template=kind == "template")
    reference = {"h": .21, "J": .4}
    compression = builder.prepare_compression(-.17j, reference, max_tree_rank=2)
    exact = dense_result(builder.exp(-.17j, reference, materialize=False))
    projected, report = builder.exp(-.17j, reference, compression=compression,
                                     materialize=False, return_report=True)
    actual = dense_result(projected)
    assert np.linalg.norm(actual - exact) > 1e-7  # This cap really truncates.
    np.testing.assert_allclose(np.linalg.norm(actual-exact), compression.reference_errors[0][1],
                               atol=1e-14)
    assert report["compression"]["method"] == "frozen-projectors"
    assert max(rank for _, ranks in compression.ranks for _, rank in ranks) == 2
    assert projected.bond_dim < builder.exp(-.17j, reference, materialize=False).bond_dim
    weight = np.arange(16).reshape(4, 4) / 16

    def value(theta):
        result = builder.exp(-.17j, dict(zip(("h", "J"), theta)),
                             materialize=False, compression=compression)
        matrix = dense_result(result)
        if isinstance(matrix, np.ndarray):
            return ((matrix.real + .37 * matrix.imag) * weight).sum()
        return ((matrix.real + .37 * matrix.imag) * torch.as_tensor(weight)).sum()

    for values in ([.21, .4], [0., 0.], [-.1, .2]):
        theta = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        gradient, = torch.autograd.grad(value(theta), theta)
        eps = 1e-5
        fd = [(value(np.array(values)+eps*np.eye(2)[i]) - value(np.array(values)-eps*np.eye(2)[i]))/(2*eps)
              for i in range(2)]
        np.testing.assert_allclose(gradient.numpy(), fd, atol=3e-9, rtol=3e-8)
    # Reference bases have no live tensor graph and remain unchanged on replay.
    assert all(isinstance(matrix, np.ndarray) and not matrix.flags.writeable
               for _, matrices in compression.projectors for _, matrix in matrices)


def test_uncapped_projectors_reconstruct_and_geometry_is_checked():
    builder = pair_builder()
    plan = builder.prepare_compression(-.17j, {"h": .2, "J": .4}, max_tree_rank=4)
    for params in ({"h": 0., "J": 0.}, {"h": -.3, "J": .6}):
        np.testing.assert_allclose(
            dense_result(builder.exp(-.17j, params, compression=plan)),
            dense_result(builder.exp(-.17j, params)), atol=2e-14)
    wrong = ClusterPlan.from_supports(3, [(0, 2)], cluster_size=2)
    other = PEPOClusterProductExpansion.from_plan(wrong, [[MPOProductTerm.from_pauli((0, 2), "XX")]])
    with pytest.raises(ValueError, match="different cluster geometry"):
        other.exp(-.1j, compression=plan)
    for rank in (0, -1, True):
        with pytest.raises((ValueError, TypeError)):
            builder.prepare_compression(-.1j, {"h": .2, "J": .4}, max_tree_rank=rank)


def test_square_routing_rejects_charge_metadata():
    term = MPOProductTerm((0, 3), (PAULI["X"], PAULI["X"]), charge=1)
    plan = ClusterPlan.from_terms([term], sites=4, shape=(2, 2), cluster_size=2)
    with pytest.raises(ValueError, match="charge, fermion, or string"):
        PEPOClusterProductExpansion.from_plan(plan, [[term]], layout="square")


def test_projected_square_jax_jit_value_and_gradient():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    builder = pair_builder()
    compression = builder.prepare_compression(-.17j, {"h": .2, "J": .4}, max_tree_rank=2)
    with jax.enable_x64(True):
        def loss(theta):
            matrix = builder.exp(-.17j, dict(zip(("h", "J"), theta)), compression=compression,
                                 materialize=True).to_dense()
            return jnp.sum(matrix.real + .4 * matrix.imag)
        compiled = jax.jit(jax.value_and_grad(loss))
        for values in ([.2, .4], [0., 0.]):
            value, gradient = compiled(jnp.array(values))
            np.testing.assert_allclose(value, loss(jnp.array(values)), atol=2e-13)
            eps = 1e-5
            fd = [(loss(jnp.array(values)+eps*jnp.eye(2)[i]) -
                   loss(jnp.array(values)-eps*jnp.eye(2)[i]))/(2*eps) for i in range(2)]
            np.testing.assert_allclose(gradient, fd, atol=3e-9)


def test_projected_branched_tree_and_graph_report():
    from pepsy.operators import mpo_product

    torch = pytest.importorskip("torch")
    factors = [MPOClusterFactor([
        MPOProductTerm.from_pauli((0, 1), "XY", coefficient=MPOParameter("a")),
        MPOProductTerm.from_pauli((0, 2), "ZZ", coefficient=.31),
    ]), MPOClusterFactor([MPOProductTerm.from_pauli((1, 2), "XZ", coefficient=.17)])]
    plan = ClusterPlan.from_terms([t for f in factors for t in f.terms], sites=3, cluster_size=3)
    builder = PEPOClusterProductExpansion.from_plan(plan, factors, layout="graph", factorization="fixed")
    compression = builder.prepare_compression(-.19j, {"a": .4}, max_tree_rank=2)
    with patch.object(mpo_product, "_matrix_exponential", wraps=mpo_product._matrix_exponential) as exp:
        _, report = builder.exp(-.19j, {"a": .4}, compression=compression, return_report=True)
    assert report["materialization"]["factor_exponentials_evaluated"] == exp.call_count
    assert report["materialization"]["partition_products"] > 0

    def loss(a):
        matrix = builder.exp(-.19j, {"a": a}, compression=compression).to_dense()
        return (matrix.real + .3*matrix.imag).sum()

    for value in (.2, 0.):
        theta = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        grad, = torch.autograd.grad(loss(theta), theta)
        fd = (loss(value+1e-5)-loss(value-1e-5))/2e-5
        np.testing.assert_allclose(grad.numpy(), fd, atol=3e-9)


def test_generic_uniform_report_counts_lower_contractions():
    from pepsy.operators import pepo_basis

    basis = PauliPEPOBasis(1, 5, [("onsite", "X", .2), ("edge", "ZZ", .4)],
                          order=5, factorization="fixed")
    builder = PEPOClusterProductExpansion.from_bases([basis])
    with patch.object(pepo_basis, "_contract_active_support_backend",
                      wraps=pepo_basis._contract_active_support_backend) as lower:
        _, report = builder.exp(-.03j, materialize=False, return_report=True)
    assert report["materialization"]["lower_contractions"] == lower.call_count > 0


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_projection_replay_preserves_torch_dtype(dtype):
    torch = pytest.importorskip("torch")
    terms = [MPOProductTerm((0, 1), (PAULI["X"].astype(dtype), PAULI["Y"].astype(dtype)),
                            MPOParameter("a"))]
    plan = ClusterPlan.from_terms(terms, sites=2, shape=(1, 2), cluster_size=2)
    builder = PEPOClusterProductExpansion.from_plan(plan, [terms], layout="graph", factorization="fixed")
    step = torch.tensor(-.13j, dtype=getattr(torch, dtype))
    reference = torch.tensor(.3, dtype=step.real.dtype, requires_grad=True)
    compression = builder.prepare_compression(step, {"a": reference}, max_tree_rank=2)
    theta = torch.tensor(.2, dtype=reference.dtype, requires_grad=True)
    matrix = builder.exp(step, {"a": theta}, compression=compression).to_dense()
    assert matrix.dtype == step.dtype
    assert matrix.device == step.device
    gradient, unused = torch.autograd.grad(matrix.imag.sum(), (theta, reference), allow_unused=True)
    assert unused is None
    assert torch.isfinite(gradient)


@pytest.mark.parametrize("localized", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_reports_count_actual_exponentials_and_lower_contractions(localized, reuse):
    from pepsy.operators import pepo_basis

    terms = [("onsite", "X", MPOParameter("h")), ("edge", "ZZ", .3)]
    if localized:
        terms = [PauliPEPOTerm("onsite", "X", MPOParameter("h"), where=divmod(i, 2)) for i in range(4)]
        terms += [("edge", "ZZ", .3)]
    basis = PauliPEPOBasis(2, 2, terms, order=4, factorization="fixed", spatial_reuse=reuse)
    builder = PEPOClusterProductExpansion.from_bases([basis])
    original = pepo_basis._backend_expm
    measured = []

    def exponential(matrix):
        measured.append(matrix.shape[0] if matrix.ndim == 3 else 1)
        return original(matrix)

    with patch.object(pepo_basis, "_backend_expm", exponential), \
            patch.object(pepo_basis, "_contract_active_support_backend",
                         wraps=pepo_basis._contract_active_support_backend) as lower:
        _, report = builder.exp(-.07j, {"h": .2}, materialize=False, return_report=True)
    counts = report["materialization"]
    assert counts["exponential_batch_sizes"] == tuple(measured)
    assert counts["factor_exponentials_evaluated"] == sum(measured)
    assert counts["lower_contractions"] == lower.call_count
    assert counts["factor_exponentials_reused"] >= 0
    assert counts["local_products_reused"] >= 0
    # A fresh call owns fresh counters; compilation and previous calls do not accumulate.
    _, again = builder.exp(-.08j, {"h": .1}, materialize=False, return_report=True)
    assert again["materialization"] == counts

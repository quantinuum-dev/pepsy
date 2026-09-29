"""General graph PEPOs preserve ordered matrices, supports, and physical axes."""

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import (
    ClusterPlan,
    GraphClusterExpansionPlan,
    GraphPEPOClusterProductExpansion,
    MPOClusterFactor,
    MPOClusterProductExpansion,
    MPOProductTerm,
    PEPOClusterProductExpansion,
    PauliPEPOBasis,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Y = np.array([[0.0, -1j], [1j, 0.0]])
Z = np.diag([1.0, -1.0])


def test_graph_dense_materialization_preserves_asymmetric_complex_operator():
    plan = ClusterPlan.from_supports(2, [(0, 1)], cluster_size=2)
    builder = GraphClusterExpansionPlan.from_plan(plan, np.kron(X, Y), np.zeros((2, 2)))
    np.testing.assert_allclose(builder.exp(0.07).to_dense(), expm(0.07 * np.kron(X, Y)), atol=1e-12)


@pytest.mark.parametrize("cutoff", [2, 3])
def test_ordered_graph_product_matches_mpo_and_dense(cutoff):
    terms = [MPOProductTerm((0, 2), (X, Y), 0.3), MPOProductTerm((0,), (Z,), 0.17)]
    second = [MPOProductTerm((0, 1), (Z, X), -0.2), MPOProductTerm((1, 2), (Y, Z), 0.23)]
    plan = ClusterPlan.from_terms((*terms, *second), sites=3, cluster_size=cutoff)
    factors = (MPOClusterFactor(terms), MPOClusterFactor(second, 0.8))
    pepo = PEPOClusterProductExpansion.from_plan(plan, factors)
    mpo = MPOClusterProductExpansion.from_plan(plan, factors, graph_assembly="exact", cutoff=0)
    assert isinstance(pepo, GraphPEPOClusterProductExpansion)
    for step in (0.07, -0.11j):
        active, report = pepo.exp(step, return_report=True)
        np.testing.assert_allclose(
            active.to_dense(), mpo.exp(step, materialize=True).to_dense(), atol=1e-12
        )
        np.testing.assert_allclose(pepo.trace_exp(step), np.trace(active.to_dense()), atol=1e-12)
        assert report["cluster_counts"] == plan.counts
        if cutoff == 3:
            a = 0.3 * np.kron(np.kron(X, np.eye(2)), Y) + 0.17 * np.kron(
                np.kron(Z, np.eye(2)), np.eye(2)
            )
            b = -0.2 * np.kron(np.kron(Z, X), np.eye(2)) + 0.23 * np.kron(np.kron(np.eye(2), Y), Z)
            np.testing.assert_allclose(
                active.to_dense(), expm(step * a) @ expm(step * 0.8 * b), atol=1e-12
            )


def test_shared_pauli_basis_factory_and_runtime_slots():
    plan = ClusterPlan.from_supports(2, [(0, 1)], cluster_size=2)
    basis = PauliPEPOBasis.from_plan(plan, [((0, 1), "XY", 0.2), ((0,), "Z", 0.1)])
    assert basis.compile_exp() is basis
    for values in ((0.2, 0.3), (-0.1, 0.4)):
        expected = expm(0.1 * (values[0] * np.kron(X, Y) + values[1] * np.kron(Z, np.eye(2))))
        np.testing.assert_allclose(
            basis.exp(0.1, coefficients=values).to_dense(), expected, atol=1e-12
        )
        np.testing.assert_allclose(
            basis.trace_exp(0.1, coefficients=values), np.trace(expected), atol=1e-12
        )


def test_higher_body_operator_and_disconnected_site_materialize_correctly():
    plan = ClusterPlan.from_supports(4, [(0, 1, 2)], cluster_size=3)
    basis = PauliPEPOBasis.from_plan(plan, [((0, 1, 2), "XYZ", 0.2), ((3,), "Y", 0.1)])
    expected = np.kron(expm(0.2 * np.kron(np.kron(X, Y), Z)), expm(0.1 * Y))
    np.testing.assert_allclose(basis.exp(1.0).to_dense(), expected, atol=1e-12)


def test_public_exports_and_reused_trace_recursion():
    import pepsy.operators as operators
    from pepsy.operators._cluster_trace import compile_trace_plan

    plan = ClusterPlan.from_supports(3, [(0, 1), (1, 2)], cluster_size=3)
    assert operators.ClusterPlan is ClusterPlan
    assert operators.GraphPEPOClusterProductExpansion is GraphPEPOClusterProductExpansion
    first = compile_trace_plan(3, plan.index_clusters, 100000)[1]
    second = compile_trace_plan(3, plan.index_clusters, 100000)[1]
    assert first is second
    assert first["collection_count"] + 1 == plan.collection_count()


@pytest.mark.parametrize("edges", [(), ((0, 1),)])
def test_graph_active_torch_blocks_preserve_dtype_axes_and_gradients(edges):
    torch = pytest.importorskip("torch")
    from pepsy.operators import GraphActivePEPOBlocks

    scale = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    local = scale * torch.tensor([[1., 2j], [0., 3.]], dtype=torch.complex128)
    directions = {site: tuple(i for i, edge in enumerate(edges) if site in edge)
                  for site in (0, 1)}
    active = GraphActivePEPOBlocks(
        sites=(0, 1), edges=edges, bond_dim=1, physical_dim=2,
        site_directions=directions,
        blocks={site: {(0,) * len(directions[site]): local} for site in (0, 1)},
    )
    assert active.dense_nbytes == 2 * 4 * local.element_size()
    network = active.to_tensor_network()
    output_inds = [("graph-bra", site) for site in (0, 1)]
    output_inds += [("graph-ket", site) for site in (0, 1)]
    actual = network.contract(output_inds=output_inds).data.reshape(4, 4)
    expected = torch.kron(local, local)
    torch.testing.assert_close(actual, expected)
    actual_grad, = torch.autograd.grad(actual.abs().square().sum(), scale, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected.abs().square().sum(), scale)
    torch.testing.assert_close(actual_grad, expected_grad)

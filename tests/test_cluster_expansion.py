"""Focused tests for the dense PEPO cluster-expansion vertical slice."""

import numpy as np
import pytest
from scipy.linalg import expm

import pepsy
from pepsy.operators import (
    ActivePEPOBlocks,
    ConnectedClusterShape,
    ClusterExpansionReport,
    ClusterExpansionPlan,
    ClusterModelAdapter,
    CompiledPEPOExp,
    MPOParameter,
    PauliPEPOBasis,
    PauliPEPOTerm,
    adapt_cluster_model,
    build_cluster_expansion_pepo,
    build_model_cluster_expansion_pepo,
    build_itf_cluster_expansion_pepo,
    build_real_time_cluster_expansion_pepo,
    compose_cluster_expansion_pepo,
    compose_pepo_layers,
    generate_connected_cluster_shapes,
)


def _itf_terms():
    z = np.diag([1.0, -1.0])
    onsite = np.array([[0.0, 0.2], [0.2, 0.0]])
    return np.kron(z, z), onsite


def _three_site_itf_hamiltonian(twosite, onesite):
    identity = np.eye(2)
    return (
        np.kron(np.kron(onesite, identity), identity)
        + np.kron(np.kron(identity, onesite), identity)
        + np.kron(np.kron(identity, identity), onesite)
        + np.kron(twosite, identity)
        + np.kron(identity, twosite)
    )


def _four_site_chain_hamiltonian(twosite, onesite):
    identity = np.eye(2)
    result = np.zeros((16, 16))
    for position in range(4):
        factors = [identity] * 4
        factors[position] = onesite
        result += np.kron(np.kron(np.kron(factors[0], factors[1]), factors[2]), factors[3])
    result += np.kron(twosite, np.eye(4))
    result += np.kron(np.kron(identity, twosite), identity)
    result += np.kron(np.eye(4), twosite)
    return result


def _five_site_chain_hamiltonian(twosite, onesite):
    identity = np.eye(2)
    result = np.zeros((32, 32))
    for position in range(5):
        factors = [identity] * 5
        factors[position] = onesite
        result += _kron_all(factors)
    for position in range(4):
        result += np.kron(
            np.kron(np.eye(2**position), twosite),
            np.eye(2 ** (3 - position)),
        )
    return result


def _six_site_chain_hamiltonian(twosite, onesite):
    identity = np.eye(2)
    result = sum(
        (
            _kron_all(
                [onesite if site == position else identity for site in range(6)]
            )
            for position in range(6)
        ),
        start=np.zeros((64, 64), dtype=np.result_type(twosite, onesite)),
    )
    for position in range(5):
        result += np.kron(
            np.kron(np.eye(2**position), twosite),
            np.eye(2 ** (4 - position)),
        )
    return result


def _two_by_two_itf_hamiltonian(onesite):
    """Return the open-boundary square-lattice ITF Hamiltonian."""
    identity = np.eye(2)
    z = np.diag([1.0, -1.0])
    result = np.zeros((16, 16))
    for position in range(4):
        factors = [identity] * 4
        factors[position] = onesite
        result += np.kron(
            np.kron(np.kron(factors[0], factors[1]), factors[2]),
            factors[3],
        )
    for first, second in ((0, 1), (0, 2), (1, 3), (2, 3)):
        factors = [identity] * 4
        factors[first] = factors[second] = z
        result += np.kron(
            np.kron(np.kron(factors[0], factors[1]), factors[2]),
            factors[3],
        )
    return result


def _global_pepo_expectation(pepo, peps):
    """Evaluate a PEPO expectation through Quimb's global PEPS path."""
    acted = pepo.apply(peps, contract=True)
    numerator = (peps.H & acted).contract(all, optimize="auto-hq")
    denominator = (peps.H & peps).contract(all, optimize="auto-hq")
    return numerator / denominator


def _kron_all(factors):
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def _two_by_two_product_hamiltonian(onesite, first_edge, second_edge):
    """Build a 2x2 Hamiltonian for a directed product edge term."""
    identity = np.eye(2)
    result = np.zeros((16, 16), dtype=complex)
    for position in range(4):
        result += _kron_all(
            [onesite if site == position else identity for site in range(4)]
        )
    for first, second in ((0, 1), (0, 2), (1, 3), (2, 3)):
        factors = [identity] * 4
        factors[first] = first_edge
        factors[second] = second_edge
        result += _kron_all(factors)
    return result


def test_cluster_expansion_builder_is_public():
    """The PEPO builder is exposed through both operator namespaces."""
    assert build_cluster_expansion_pepo is pepsy.build_cluster_expansion_pepo
    assert build_cluster_expansion_pepo is pepsy.operators.build_cluster_expansion_pepo
    assert build_itf_cluster_expansion_pepo is pepsy.build_itf_cluster_expansion_pepo
    assert ClusterExpansionPlan is pepsy.ClusterExpansionPlan
    assert ClusterExpansionReport is pepsy.ClusterExpansionReport
    assert ActivePEPOBlocks is pepsy.ActivePEPOBlocks
    assert PauliPEPOBasis is pepsy.PauliPEPOBasis
    assert PauliPEPOTerm is pepsy.PauliPEPOTerm
    assert CompiledPEPOExp is pepsy.CompiledPEPOExp
    assert ConnectedClusterShape is pepsy.ConnectedClusterShape
    assert ClusterModelAdapter is pepsy.ClusterModelAdapter
    assert adapt_cluster_model is pepsy.adapt_cluster_model
    assert build_model_cluster_expansion_pepo is pepsy.build_model_cluster_expansion_pepo
    assert build_real_time_cluster_expansion_pepo is pepsy.build_real_time_cluster_expansion_pepo
    assert generate_connected_cluster_shapes is pepsy.generate_connected_cluster_shapes
    assert "build_cluster_expansion_pepo" in pepsy.operators.__all__


def test_real_time_coefficient_terms_use_the_exact_sum_convention():
    """Coefficient pairs are assembled before the complex cluster solve."""
    twosite, onesite = _itf_terms()
    time = 0.01
    assembled_twosite = 0.4 * twosite + 0.3 * twosite
    assembled_onesite = 0.1 * onesite + 0.2 * onesite
    exact = expm(
        -1j * time * _five_site_chain_hamiltonian(
            assembled_twosite,
            assembled_onesite,
        )
    )

    pepo, report = build_real_time_cluster_expansion_pepo(
        1,
        5,
        time,
        [(0.4, twosite), (0.3, twosite)],
        [(0.1, onesite), (0.2, onesite)],
        order=5,
        fit_steps=8,
        fit_tol=1e-11,
        return_report=True,
    )

    np.testing.assert_allclose(pepo.to_dense(), exact, atol=1e-10)
    assert report.beta == 1j * time
    assert report.cluster_counts["generic_order_5"] == 1
    assert report.residual_norms["generic_order_5"] < 1e-20


def test_connected_cluster_geometry_is_translation_canonical_and_connected():
    """The generic inventory counts fixed square-lattice polyominoes."""
    shapes = generate_connected_cluster_shapes(5)
    counts = [sum(shape.nsites == size for shape in shapes) for size in range(1, 6)]
    assert counts == [1, 2, 6, 19, 63]

    for shape in shapes:
        assert isinstance(shape, ConnectedClusterShape)
        assert min(x for x, _ in shape.sites) == 0
        assert min(y for _, y in shape.sites) == 0
        assert len(shape.edges) >= shape.nsites - 1
        assert shape.loops == len(shape.edges) - shape.nsites + 1
        assert shape.is_tree is (shape.loops == 0)

    plaquettes = [
        shape
        for shape in shapes
        if shape.nsites == 4 and shape.loops == 1
    ]
    assert len(plaquettes) == 1
    assert plaquettes[0].diagonal_edges == ((0, 3), (1, 2))


def test_connected_cluster_geometry_can_quotient_c4_rotations():
    fixed = generate_connected_cluster_shapes(4, min_sites=4)
    rotated = generate_connected_cluster_shapes(
        4,
        min_sites=4,
        quotient_rotations=True,
    )
    assert len(fixed) == 19
    assert len(rotated) == 7
    assert len({shape.sites for shape in rotated}) == 7


def test_cluster_plan_exposes_geometry_through_order_six():
    twosite, onesite = _itf_terms()
    plan = ClusterExpansionPlan(2, 2, twosite, onesite, order=4)
    assert len(plan.connected_cluster_shapes) == 28
    order_six = ClusterExpansionPlan(2, 2, twosite, onesite, order=6)
    assert len(order_six.connected_cluster_shapes) == 307


def test_order_three_is_exact_on_a_three_site_chain():
    """All connected clusters through size three reproduce the local exponential."""
    twosite, onesite = _itf_terms()
    beta = 0.1
    exact = expm(-beta * _three_site_itf_hamiltonian(twosite, onesite))

    order_two = build_cluster_expansion_pepo(
        1,
        3,
        beta,
        twosite,
        onesite,
        order=2,
    )
    order_three = build_cluster_expansion_pepo(
        1,
        3,
        beta,
        twosite,
        onesite,
        order=3,
    )

    error_two = np.linalg.norm(np.asarray(order_two.to_dense()) - exact)
    error_three = np.linalg.norm(np.asarray(order_three.to_dense()) - exact)
    assert error_two > 1e-8
    assert error_three < 1e-12


def test_cluster_expansion_builds_periodic_pepo_without_snake_embedding():
    """The local PEPO construction supports a genuine square-lattice torus."""
    twosite, onesite = _itf_terms()
    pepo = build_cluster_expansion_pepo(
        3,
        3,
        0.03,
        twosite,
        onesite,
        order=3,
        cyclic=True,
        symmetry="C4",
    )

    assert pepo.Lx == 3
    assert pepo.Ly == 3
    assert all(tensor.shape[:4] == (5, 5, 5, 5) for tensor in pepo)


def test_itf_cluster_expansion_convenience_matches_local_builder():
    """The ITF helper uses the same ZZ-plus-X convention as ``ham_tn``."""
    twosite, onesite = _itf_terms()
    direct = build_cluster_expansion_pepo(2, 3, 0.03, twosite, onesite, order=2)
    helper = build_itf_cluster_expansion_pepo(
        2,
        3,
        0.03,
        J=1.0,
        field=0.2,
        order=2,
    )
    np.testing.assert_allclose(direct.to_dense(), helper.to_dense())


def test_plan_reuses_topology_and_keeps_active_blocks_sparse():
    """Plans cache geometry while materialization remains an explicit step."""
    twosite, onesite = _itf_terms()
    plan = ClusterExpansionPlan(2, 3, twosite, onesite, order=3)
    directions = plan.site_directions
    sparse = plan.build(0.03, materialize=False)

    assert isinstance(sparse, ActivePEPOBlocks)
    assert plan.site_directions is directions
    assert sparse.active_nbytes < sparse.dense_nbytes
    np.testing.assert_allclose(sparse.to_pepo().to_dense(), plan.build(0.03).to_dense())


def test_c4_reduction_solves_two_orbits_and_preserves_dense_result():
    """ITF C4 reduction handles one corner and one straight representative."""
    twosite, onesite = _itf_terms()
    plan = ClusterExpansionPlan(
        2,
        3,
        twosite,
        onesite,
        order=3,
        symmetry="C4",
    )
    assert plan.pair_representatives == (("u", "r"), ("u", "d"))

    reduced = plan.build(0.1)
    unreduced = build_cluster_expansion_pepo(
        2,
        3,
        0.1,
        twosite,
        onesite,
        order=3,
    )
    np.testing.assert_allclose(reduced.to_dense(), unreduced.to_dense(), atol=1e-11)


def test_c4_generic_order_five_reuses_rotated_tree_factorizations():
    """Generic finite clusters use C4 transport without dense inflation."""
    active, report = build_itf_cluster_expansion_pepo(
        2,
        3,
        0.003,
        order=5,
        symmetry="C4",
        materialize=False,
        return_report=True,
    )
    assert report.cluster_counts["generic_order_5"] == 6
    assert report.cluster_counts["generic_tree_solved"] == 6
    assert report.residual_norms["generic_order_5"] < 1e-12
    assert active.active_block_count > 0


def test_cluster_expansion_rejects_unimplemented_higher_order():
    """Orders beyond the generic order-nine implementation fail explicitly."""
    twosite, onesite = _itf_terms()
    with pytest.raises(NotImplementedError, match="orders 1 through 9"):
        build_cluster_expansion_pepo(2, 2, 0.1, twosite, onesite, order=10)


def test_order_five_generic_path_is_exact_on_a_five_site_chain():
    """Generic residual subtraction closes the five-site finite-chain case."""
    twosite, onesite = _itf_terms()
    beta = 0.01
    exact = expm(-beta * _five_site_chain_hamiltonian(twosite, onesite))
    pepo, report = build_cluster_expansion_pepo(
        1,
        5,
        beta,
        twosite,
        onesite,
        order=5,
        return_report=True,
    )
    np.testing.assert_allclose(pepo.to_dense(), exact, atol=1e-11)
    assert report.cluster_counts["generic_order_5"] == 1
    assert report.cluster_counts["generic_tree_solved"] == 1
    assert report.residual_norms["generic_order_5"] < 1e-12


def test_quimb_backend_handles_a_five_site_loop_cluster():
    """The opt-in Quimb backend covers cyclic generic cluster topology."""
    twosite, onesite = _itf_terms()
    active, report = build_cluster_expansion_pepo(
        2,
        3,
        1e-4j,
        twosite,
        onesite,
        order=5,
        fit_method="quimb",
        fit_steps=1,
        fit_tol=1e-6,
        fit_solver_maxiter=1,
        max_tree_rank=1,
        max_loop_rank=1,
        materialize=False,
        return_report=True,
    )
    dense = active.to_pepo().to_dense()
    assert report.cluster_counts["generic_order_5"] == 6
    assert np.isfinite(dense).all()
    assert np.isfinite(report.relative_residual_norms["generic_order_5"])


def test_order_five_tree_rank_cap_is_reported():
    """Generic tree SVD truncation keeps finite tensors and reports loss."""
    twosite, onesite = _itf_terms()
    active, report = build_cluster_expansion_pepo(
        1,
        5,
        0.01,
        twosite,
        onesite,
        order=5,
        max_tree_rank=2,
        materialize=False,
        return_report=True,
    )
    assert report.tree_rank <= 2
    assert report.relative_residual_norms["generic_order_5"] > 0.0
    assert np.isfinite(np.asarray(active.to_pepo().to_dense())).all()


def test_order_six_generic_path_is_exact_on_a_six_site_chain():
    """Generic residual subtraction extends to a six-site finite chain."""
    twosite, onesite = _itf_terms()
    beta = 0.006
    exact = expm(-beta * _six_site_chain_hamiltonian(twosite, onesite))
    pepo, report = build_cluster_expansion_pepo(
        1,
        6,
        beta,
        twosite,
        onesite,
        order=6,
        return_report=True,
    )
    np.testing.assert_allclose(pepo.to_dense(), exact, atol=1e-10)
    assert report.cluster_counts["generic_order_5"] == 2
    assert report.cluster_counts["generic_order_6"] == 1
    assert report.cluster_counts["generic_tree_solved"] == 2
    assert report.residual_norms["generic_order_6"] < 1e-10


def test_order_seven_recurses_through_all_lower_generic_levels():
    """P=7 includes the P=5 and P=6 corrections before its own solve."""
    active, report = build_itf_cluster_expansion_pepo(
        1,
        7,
        0.001,
        order=7,
        materialize=False,
        return_report=True,
    )
    assert active.active_block_count > 0
    assert report.cluster_counts["generic_order_5"] == 3
    assert report.cluster_counts["generic_order_6"] == 2
    assert report.cluster_counts["generic_order_7"] == 1
    assert report.cluster_counts["generic_tree_solved"] == 3
    for order in (5, 6, 7):
        assert report.residual_norms[f"generic_order_{order}"] < 1e-12


def test_dense_model_adapter_builds_standard_spin_models():
    """Finite model adapters feed the same dense cluster builder."""
    model = ClusterModelAdapter.ising(J=1.2, field=0.3)
    assert model.name == "ising"
    assert model.local_dim == 2
    assert model.symmetry == "C4"

    direct = build_cluster_expansion_pepo(
        1,
        3,
        0.02,
        model.twosite_op,
        model.onesite_op,
        order=2,
        symmetry="C4",
    )
    adapted = build_model_cluster_expansion_pepo(
        1,
        3,
        0.02,
        model,
        order=2,
    )
    np.testing.assert_allclose(direct.to_dense(), adapted.to_dense())

    recovered = adapt_cluster_model(
        {
            "edge_op": model.twosite_op,
            "onsite_op": model.onesite_op,
            "symmetry": "C4",
        }
    )
    np.testing.assert_allclose(recovered.twosite_op, model.twosite_op)
    np.testing.assert_allclose(recovered.onesite_op, model.onesite_op)


def test_order_four_path_is_exact_and_reports_local_residuals():
    """The four-site path tree closes the finite-chain residual."""
    twosite, onesite = _itf_terms()
    beta = 0.03
    active, report = build_cluster_expansion_pepo(
        1,
        4,
        beta,
        twosite,
        onesite,
        order=4,
        materialize=False,
        return_report=True,
    )

    exact = expm(-beta * _four_site_chain_hamiltonian(twosite, onesite))
    order_three = build_cluster_expansion_pepo(
        1, 4, beta, twosite, onesite, order=3
    ).to_dense()
    order_four = active.to_pepo().to_dense()

    assert isinstance(report, ClusterExpansionReport)
    assert report.order == 4
    assert report.cluster_counts["four_site_path"] == 1
    assert report.residual_norms["four_site_path"] < 1e-12
    assert np.linalg.norm(order_four - exact) < 1e-11
    assert np.linalg.norm(order_four - exact) < np.linalg.norm(order_three - exact)
    assert active.active_nbytes < active.dense_nbytes


def test_order_four_includes_embedded_star_trees():
    """A rectangular lattice exposes the degree-three tree sector."""
    twosite, onesite = _itf_terms()
    active, report = ClusterExpansionPlan(
        2,
        3,
        twosite,
        onesite,
        order=4,
    ).build(0.03, materialize=False, return_report=True)

    assert report.cluster_counts["four_site_star"] > 0
    assert report.residual_norms["four_site_star"] < 1e-10
    assert active.active_block_count > 0


def test_order_four_includes_and_closes_a_four_site_plaquette_loop():
    """The explicit loop sector closes the smallest square cluster."""
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])

    def kron_all(factors):
        result = factors[0]
        for factor in factors[1:]:
            result = np.kron(result, factor)
        return result

    hamiltonian = 0.2 * sum(
        (
            kron_all([x if site == position else identity for site in range(4)])
            for position in range(4)
        ),
        start=np.zeros((16, 16)),
    )
    for first, second in ((0, 1), (0, 2), (2, 3), (1, 3)):
        factors = [identity] * 4
        factors[first] = z
        factors[second] = z
        hamiltonian += kron_all(factors)

    plan = ClusterExpansionPlan(
        2,
        2,
        np.kron(z, z),
        0.2 * x,
        order=4,
    )
    active, report = plan.build(0.01, materialize=False, return_report=True)
    exact = expm(-0.01 * hamiltonian)

    assert report.cluster_counts["four_site_loop"] == 1
    assert report.loop_rank == 16
    assert report.residual_norms["four_site_loop"] > 0.0
    np.testing.assert_allclose(active.to_pepo().to_dense(), exact, atol=1e-11)


def test_pepo_global_expectation_matches_dense_reference_across_orders():
    """Quimb PEPO application preserves global expectations on a PEPS."""
    qtn = pytest.importorskip("quimb.tensor")
    _, onesite = _itf_terms()
    beta = 0.01
    peps = qtn.PEPS.rand(2, 2, bond_dim=2, seed=7, dtype="complex128")
    hamiltonian = _two_by_two_itf_hamiltonian(onesite)
    state_vector = np.asarray(peps.to_dense()).reshape(-1)
    norm = np.vdot(state_vector, state_vector)
    exact_operator = expm(-beta * hamiltonian)
    exact_value = np.vdot(state_vector, exact_operator @ state_vector) / norm

    errors = []
    for order in (1, 2, 3, 4):
        pepo = build_itf_cluster_expansion_pepo(
            2,
            2,
            beta,
            J=1.0,
            field=0.2,
            order=order,
        )
        value = _global_pepo_expectation(pepo, peps)
        dense_value = (
            np.vdot(state_vector, np.asarray(pepo.to_dense()) @ state_vector)
            / norm
        )

        np.testing.assert_allclose(value, dense_value, atol=1e-11, rtol=1e-11)
        errors.append(abs(value - exact_value))

    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-10
    assert errors[3] < 1e-10


def test_quimb_pepo_layer_composition_matches_operator_product():
    """The public composition helper retains Quimb's operator ordering."""
    first = build_itf_cluster_expansion_pepo(2, 2, 0.01, order=2)
    second = build_itf_cluster_expansion_pepo(2, 2, 0.02, order=2)

    composed = compose_pepo_layers((first, second))

    np.testing.assert_allclose(
        composed.to_dense(),
        second.to_dense() @ first.to_dense(),
        atol=1e-11,
        rtol=1e-11,
    )
    assert next(iter(first.tensor_map.values())).shape == (5, 5, 2, 2)
    assert next(iter(second.tensor_map.values())).shape == (5, 5, 2, 2)


def test_yoshida_pepo_composition_improves_order_three_step():
    """Signed fractional P=3 layers cancel the leading local error."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    beta = 0.025
    twosite = np.kron(x, z)
    exact_hamiltonian = _two_by_two_product_hamiltonian(x, x, z)
    exact = expm(-beta * exact_hamiltonian)

    base = build_cluster_expansion_pepo(
        2,
        2,
        beta,
        twosite,
        x,
        order=3,
    )
    composed = compose_cluster_expansion_pepo(
        2,
        2,
        beta,
        twosite,
        x,
        order=3,
    )

    base_error = np.linalg.norm(np.asarray(base.to_dense()) - exact)
    composed_error = np.linalg.norm(np.asarray(composed.to_dense()) - exact)
    assert composed_error < 0.5 * base_error


def test_yoshida_pepo_composition_requires_order_three():
    """The fourth-order composition rejects unsupported elementary orders."""
    twosite, onesite = _itf_terms()
    with pytest.raises(ValueError, match="requires a plan with order=3"):
        ClusterExpansionPlan(2, 2, twosite, onesite, order=4).build_composed(0.01)


def test_order_four_c4_periodic_plan_handles_rotated_path_orbits():
    """Periodic C4 paths build without conflating reversal with rotation."""
    twosite, onesite = _itf_terms()
    active, report = ClusterExpansionPlan(
        3,
        3,
        twosite,
        onesite,
        order=4,
        cyclic=True,
        symmetry="C4",
    ).build(0.03, materialize=False, return_report=True)

    assert report.cluster_counts["four_site_path"] > 0
    assert active.active_block_count > 0
    assert active.active_nbytes < active.dense_nbytes


def _pauli_chain_hamiltonian(nsites, onsite_coefficient, edge_coefficient):
    identity = np.eye(2, dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)

    def kron_all(factors):
        result = factors[0]
        for factor in factors[1:]:
            result = np.kron(result, factor)
        return result

    result = np.zeros((2**nsites, 2**nsites), dtype=complex)
    for site in range(nsites):
        factors = [identity] * nsites
        factors[site] = x
        result += onsite_coefficient * kron_all(factors)
    for site in range(nsites - 1):
        factors = [identity] * nsites
        factors[site] = z
        factors[site + 1] = z
        result += edge_coefficient * kron_all(factors)
    return result


@pytest.mark.parametrize("order, nsites", ((2, 2), (3, 3), (4, 4)))
def test_pauli_pepo_basis_matches_finite_chain_through_tree_order(order, nsites):
    """Fixed Pauli channels close every finite chain tree through order four."""
    basis = PauliPEPOBasis.compile(
        1,
        nsites,
        [
            PauliPEPOTerm("onsite", "X", coefficient=0.2),
            PauliPEPOTerm("edge", "ZZ", coefficient=1.0),
        ],
        order=order,
    )
    pepo = basis.exp(step=-1j * 0.01, materialize=True)
    exact = expm(-1j * 0.01 * _pauli_chain_hamiltonian(nsites, 0.2, 1.0))
    np.testing.assert_allclose(pepo.to_dense(), exact, atol=1e-11)


def test_pauli_pepo_basis_keeps_torch_coefficient_and_time_graph():
    """Coefficient slots and real-time steps remain differentiable."""
    torch = pytest.importorskip("torch")
    basis = PauliPEPOBasis.compile(
        1,
        2,
        [("onsite", "X"), ("edge", "ZZ")],
        order=2,
    )
    coefficients = torch.tensor(
        [0.2, 1.0], dtype=torch.float64, requires_grad=True
    )
    tau = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    active = basis.evaluate(
        tau=tau,
        coefficients=coefficients,
        materialize=False,
    )
    loss = sum(
        block.real.sum()
        for site_blocks in active.blocks.values()
        for block in site_blocks.values()
    )
    coefficient_gradient, time_gradient = torch.autograd.grad(
        loss,
        (coefficients, tau),
    )
    assert any(
        getattr(block, "requires_grad", False)
        for site_blocks in active.blocks.values()
        for block in site_blocks.values()
    )
    assert torch.isfinite(coefficient_gradient).all()
    assert torch.isfinite(time_gradient)


def test_pauli_pepo_basis_resolves_mpo_parameter_references_with_autodiff():
    """PEPO coefficient references follow the MPOBasis parameter contract."""
    torch = pytest.importorskip("torch")
    basis = PauliPEPOBasis.compile(
        1,
        2,
        [
            PauliPEPOTerm("onsite", "X", coefficient=MPOParameter("h")),
            PauliPEPOTerm("edge", "ZZ", coefficient=MPOParameter("J")),
        ],
        order=2,
    )
    h = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    j = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    tau = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    active = basis.compile_exp().exp(
        -1j * tau,
        parameters={"h": h, "J": j},
        materialize=False,
    )
    loss = sum(
        block.real.sum()
        for site_blocks in active.blocks.values()
        for block in site_blocks.values()
    )
    gradients = torch.autograd.grad(loss, (h, j, tau))
    assert all(torch.isfinite(gradient) for gradient in gradients)


def test_pauli_pepo_c4_rotation_preserves_torch_autodiff():
    """C4 active-axis rotations stay inside the backend autodiff graph."""
    torch = pytest.importorskip("torch")
    basis = PauliPEPOBasis.compile(
        2,
        2,
        [("onsite", "X"), ("edge", "ZZ")],
        order=3,
        cyclic=True,
        symmetry="C4",
    )
    coefficients = torch.tensor(
        [0.2, 1.0], dtype=torch.float64, requires_grad=True
    )
    tau = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    active = basis.compile_exp().exp(
        -1j * tau,
        coefficients=coefficients,
        materialize=False,
    )
    loss = sum(
        block.real.sum()
        for site_blocks in active.blocks.values()
        for block in site_blocks.values()
    )
    coefficient_gradient, time_gradient = torch.autograd.grad(
        loss,
        (coefficients, tau),
    )
    assert torch.isfinite(coefficient_gradient).all()
    assert torch.isfinite(time_gradient)


def test_pauli_pepo_plaquette_loop_preserves_torch_autodiff():
    """The fixed-rank loop channel keeps coefficient graphs intact."""
    torch = pytest.importorskip("torch")
    basis = PauliPEPOBasis.compile(
        2,
        2,
        [("onsite", "X"), ("edge", "ZZ")],
        order=4,
    )
    coefficients = torch.tensor(
        [0.2, 1.0], dtype=torch.float64, requires_grad=True
    )
    tau = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    active = basis.compile_exp().exp(
        -1j * tau,
        coefficients=coefficients,
        materialize=False,
    )
    loss = sum(
        block.real.sum()
        for site_blocks in active.blocks.values()
        for block in site_blocks.values()
    )
    coefficient_gradient, time_gradient = torch.autograd.grad(
        loss,
        (coefficients, tau),
    )
    assert torch.isfinite(coefficient_gradient).all()
    assert torch.isfinite(time_gradient)


def test_pauli_pepo_basis_reuses_topology_not_backend_values():
    """Repeated evaluations update slots while retaining compiled geometry."""
    basis = PauliPEPOBasis.compile(
        1,
        2,
        [("onsite", "X"), {"support": "edge", "paulis": "ZZ"}],
        order=2,
    )
    first = basis.evaluate(tau=0.01, coefficients=np.array([0.2, 1.0]))
    second = basis.evaluate(tau=0.02, coefficients=np.array([0.3, 0.7]))
    assert basis.cache_info["builds"] == 2
    assert first.bond_dim == second.bond_dim
    assert first.blocks is not second.blocks


def test_active_pepo_blocks_compact_orphans_and_materialize_native_symmray():
    """Diagonal PEPOs compact inactive channels into native charge blocks."""
    pytest.importorskip("symmray")
    basis = PauliPEPOBasis.compile(
        2,
        2,
        [("onsite", "Z", 0.2), ("edge", "ZZ")],
        order=4,
    )
    active = basis.exp(0.01, materialize=False)
    compact = active.compact()
    assert compact.bond_dim < active.bond_dim
    native = compact.to_symmray_pepo(symmetry="U1")
    assert all(
        hasattr(native[i, j].data, "blocks")
        for i in range(2)
        for j in range(2)
    )

    dense = compact.to_pepo()
    output_inds = dense.outer_inds()
    native_tensor = native.contract(output_inds=output_inds)
    dense_tensor = dense.contract(output_inds=output_inds)
    np.testing.assert_allclose(
        native_tensor.data.to_dense(),
        dense_tensor.data,
        atol=1e-12,
    )


def test_native_symmray_conversion_rejects_mixed_operator_charge():
    """A homogeneous native charge cannot hide a symmetry-breaking slot."""
    pytest.importorskip("symmray")
    basis = PauliPEPOBasis.compile(
        2,
        2,
        [("onsite", "X", 0.2), ("edge", "ZZ")],
        order=2,
    )
    with pytest.raises(ValueError, match="not compatible with Z2 charge"):
        basis.exp(0.01, materialize=False).to_symmray_pepo(symmetry="Z2")


def test_pauli_pepo_compile_exp_is_the_preferred_cached_interface():
    """The PEPO cache uses ``exp`` without an evolution-specific name."""
    basis = PauliPEPOBasis.compile(
        1,
        2,
        [("onsite", "X"), ("edge", "ZZ")],
        order=2,
    )
    compiled = basis.compile_exp()
    assert isinstance(compiled, CompiledPEPOExp)
    assert compiled is basis.compile_exp()
    first = compiled.exp(-1j * 0.01, coefficients=np.array([0.2, 1.0]))
    second = compiled(-1j * 0.02, coefficients=np.array([0.3, 0.7]))
    assert first.bond_dim == second.bond_dim
    assert basis.cache_info["compiled_exp"]


def test_pauli_pepo_compile_exp_prepares_value_independent_cluster_plans():
    """Compiled PEPO exponentials cache geometry but not backend values."""
    basis = PauliPEPOBasis.compile(
        4,
        4,
        [("onsite", "X"), ("edge", "ZZ")],
        order=4,
        symmetry="C4",
    )
    assert basis.cache_info["cluster_embedding_plans"] == 0
    compiled = basis.compile_exp()
    first_plan_count = basis.cache_info["cluster_embedding_plans"]
    assert first_plan_count > 0

    first = compiled.exp(-1j * 0.01, coefficients=np.array([0.2, 1.0]))
    second = compiled.exp(-1j * 0.02, coefficients=np.array([0.3, 0.7]))

    assert basis.cache_info["cluster_embedding_plans"] == first_plan_count
    assert first.blocks is not second.blocks
